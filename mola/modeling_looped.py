"""Looped student architecture for the MoLA-style conversion test.

Student: 2 input blocks + ONE shared loop block + 2 output blocks, all
initialized from Qwen2.5-0.5B teacher weights (loop block = teacher block 12),
plus a small loop-count embedding MLP trained from scratch.

MoLA change (Mixture-of-Loops Transformer, arXiv:2611.34923, §3.2): the
shared loop block's attention at loop r reads the previous loop's depth KV
as addressable memory — keys/values are [prev-loop K/V (positions j <= t),
current-loop K/V (positions j <= t)] in a single shared softmax, with no
separate learned gate. Previous-loop keys keep the writer's rotary
application and are reused without re-rotation. Gradients flow through the
cross-loop KV. Loop 1 has no previous loop and runs standard causal
attention.

Load with: build with the teacher's layers, then load_state_dict() from
student_state.pt (trainable params only, bf16).
"""
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(q, k, cos, sin):
    """RoPE, matching transformers' apply_rotary_pos_emb (cos/sin from the
    model's rotary_emb, shape (B, T, D) or (1, T, D) or (T, D))."""
    if cos.dim() == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _repeat_kv(hidden_states, n_rep):
    """(B, kv_heads, T, D) -> (B, kv_heads*n_rep, T, D), GQA expansion."""
    if n_rep == 1:
        return hidden_states
    return hidden_states.repeat_interleave(n_rep, dim=1)


class LoopCountEmbedding(nn.Module):
    """Small loop-count embedding, trained from scratch (paper: 2-layer MLP)."""

    def __init__(self, hidden_size, max_k=8, emb_dim=64, mlp_dim=256):
        super().__init__()
        self.table = nn.Embedding(max_k + 1, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, hidden_size),
        )
        nn.init.normal_(self.table.weight, std=0.02)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, k):
        idx = torch.tensor([k], device=self.table.weight.device)
        return self.mlp(self.table(idx)).squeeze(0)


class LoopedStudent(nn.Module):
    def __init__(self, teacher):
        super().__init__()
        model = teacher.model
        self.config = teacher.config
        self.embed = model.embed_tokens  # frozen (tied to lm_head)
        # Deepcopy: training the student must NOT touch the teacher.
        self.in_blocks = nn.ModuleList(
            [copy.deepcopy(model.layers[0]), copy.deepcopy(model.layers[1])]
        )
        self.loop_block = copy.deepcopy(model.layers[12])
        self.out_blocks = nn.ModuleList(
            [copy.deepcopy(model.layers[22]), copy.deepcopy(model.layers[23])]
        )
        self.norm = copy.deepcopy(model.norm)
        self.lm_head = teacher.lm_head  # frozen (tied)
        self.rotary_emb = model.rotary_emb
        self.loop_emb = LoopCountEmbedding(self.config.hidden_size)
        # The deepcopies inherit the teacher's frozen flags; the student owns
        # its blocks as trainable. (First version of this script missed this:
        # only the 0.2M loop embedding trained. Fixed here.)
        for blk in list(self.in_blocks) + [self.loop_block] + list(self.out_blocks) + [self.norm]:
            for p in blk.parameters():
                p.requires_grad_(True)

    def _pos_emb(self, hidden, position_ids):
        try:
            return self.rotary_emb(hidden, position_ids)
        except Exception:
            return None

    def _run_block(self, block, hidden, pos_emb, position_ids):
        if pos_emb is not None:
            try:
                out = block(hidden, position_embeddings=pos_emb)
                return out[0] if isinstance(out, tuple) else out
            except TypeError:
                pass
        try:
            out = block(hidden, position_ids=position_ids)
        except TypeError:
            out = block(hidden)
        return out[0] if isinstance(out, tuple) else out

    def _run_loop_block(self, hidden, pos_emb, position_ids, prev_kv):
        """One shared loop-block application with MoLA depth-KV memory.

        prev_kv: (prev_k, prev_v) from the previous loop, post-rotary,
        head-shaped (B, kv_heads, T, D), or None for loop 1. At loop r the
        attention reads [prev K/V (j <= t), current K/V (j <= t)] in one
        shared softmax — no separate gate. Returns (hidden, (k_cur, v_cur))
        where k_cur/v_cur are the post-rotary current-loop K/V for the
        next loop's memory.
        """
        if pos_emb is None:
            raise RuntimeError("loop block needs rotary position embeddings")
        block = self.loop_block
        attn = block.self_attn
        residual = hidden
        h = block.input_layernorm(hidden)
        B, T, _ = h.shape
        # Head counts from config, not the attention module: newer
        # transformers versions renamed Qwen2Attention's attributes
        # (this install has no .num_heads) while the config is stable.
        n_head = self.config.num_attention_heads
        n_kv = getattr(self.config, "num_key_value_heads", n_head)
        head_dim = getattr(self.config, "head_dim",
                           self.config.hidden_size // n_head)
        q = attn.q_proj(h).view(B, T, n_head, head_dim).transpose(1, 2)
        k = attn.k_proj(h).view(B, T, n_kv, head_dim).transpose(1, 2)
        v = attn.v_proj(h).view(B, T, n_kv, head_dim).transpose(1, 2)
        cos, sin = pos_emb
        q, k = _apply_rotary(q, k, cos, sin)
        n_rep = n_head // n_kv
        tril = torch.tril(torch.ones(T, T, dtype=torch.bool, device=h.device))
        if prev_kv is None:
            k_all = _repeat_kv(k, n_rep)
            v_all = _repeat_kv(v, n_rep)
            allowed = tril
        else:
            prev_k, prev_v = prev_kv
            k_all = torch.cat([_repeat_kv(prev_k, n_rep), _repeat_kv(k, n_rep)], dim=2)
            v_all = torch.cat([_repeat_kv(prev_v, n_rep), _repeat_kv(v, n_rep)], dim=2)
            allowed = torch.cat([tril, tril], dim=1)
        mask = torch.where(
            allowed, torch.zeros((), device=h.device, dtype=torch.float32),
            torch.full((), float("-inf"), device=h.device, dtype=torch.float32))
        attn_out = F.scaled_dot_product_attention(
            q, k_all, v_all, attn_mask=mask, scale=1.0 / math.sqrt(head_dim))
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        attn_out = attn.o_proj(attn_out)
        hidden = residual + attn_out
        residual = hidden
        hidden = block.post_attention_layernorm(hidden)
        hidden = block.mlp(hidden)
        hidden = residual + hidden
        return hidden, (k, v)

    def forward(self, input_ids, K, return_per_loop_logits=False):
        """Returns (final_logits, per_loop_logits or None)."""
        device = input_ids.device
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        hidden = self.embed(input_ids)
        pos_emb = self._pos_emb(hidden, position_ids)
        for blk in self.in_blocks:
            hidden = self._run_block(blk, hidden, pos_emb, position_ids)
        per_loop_logits = [] if return_per_loop_logits else None
        prev_kv = None
        for k in range(1, K + 1):
            hidden = hidden + self.loop_emb(k)
            hidden, prev_kv = self._run_loop_block(hidden, pos_emb, position_ids, prev_kv)
            if return_per_loop_logits:
                h = hidden
                for blk in self.out_blocks:
                    h = self._run_block(blk, h, pos_emb, position_ids)
                per_loop_logits.append(self.lm_head(self.norm(h)))
        h = hidden
        for blk in self.out_blocks:
            h = self._run_block(blk, h, pos_emb, position_ids)
        return self.lm_head(self.norm(h)), per_loop_logits
