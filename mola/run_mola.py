#!/usr/bin/env python3
"""
MoLA-style conversion test (Mixture-of-Loops Transformer, arXiv:2611.34923).

Falsifiable next step from the Mixture-of-Loops read: the Dooop falsifier
(2026-11-25) showed a Distill-to-Loop student drifting away from its teacher
past K=2 (KL 2.78 -> 2.76 -> 2.92 -> 3.47). MoLT says the drift is
residual-stream dilution, and that addressable per-layer KV memory fixes it.

This test keeps the falsifier's setup EXACTLY (same teacher, same student
shape, same loss, same data, same steps) and changes ONE thing: the shared
loop block's attention at loop r also reads the previous loop's depth KV
as addressable memory, in a single shared softmax, with no separate learned
gate (MoLA §3.2). Mask form: previous-loop KV at positions j <= t
(MoLT-KV-style prefix memory). Gradients flow through the cross-loop KV
(loop forcing, §3.4, 1:1 loop-loss weights).

Decision rule: if per-loop KL keeps falling past K=2, MoLA rescues the
small-scale LM case and the falsifier's verdict was recipe-specific; if KL
still rises past K=2, the falsifier's verdict stands as scale-specific.

Same shape as run_falsifier.py: saves checkpoints every 100 steps + final
weights + student config + tokenizer + standalone modeling file, all under
--out. Runs a greedy generation probe and records outputs verbatim.
Writes results.json (+ baseline.json) and prints a JSON summary to stdout.
"""
import argparse
import copy
import importlib.util
import json
import math
import os
import shutil
import time
import warnings

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Captured Python warnings (e.g. the HF_TOKEN vault warning) for the record.
WARNINGS_SEEN = []


def _record_warning(message, category, filename, lineno, file=None, line=None):
    WARNINGS_SEEN.append(f"{category.__name__}: {message} ({filename}:{lineno})")


warnings.showwarning = _record_warning
warnings.simplefilter("always")

TRAINABLE_PREFIXES = ("in_blocks", "loop_block", "out_blocks", "norm", "loop_emb")

# Falsifier baselines (2026-11-25, held-out wikitext-103 validation, 48 seqs).
FALSIFIER_BASELINE = {
    "teacher_ppl": 15.72,
    "trained_ppl": {1: 226.25, 2: 222.21, 3: 264.01, 4: 463.53},
    "trained_kl": {1: 2.7795, 2: 2.7555, 3: 2.9194, 4: 3.4702},
    "kl_by_loop_k4": [2.7795, 2.7555, 2.9194, 3.4702],
}

# Standalone copy of the student architecture, saved into the model dir so the
# weights are loadable without this training script. Kept identical to the
# classes below; verified by loading it back at save time (see save_model()
# / verify_saved_model()).
MODELING_SOURCE = '''"""Looped student architecture for the MoLA-style conversion test.

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
'''


# ---------------------------------------------------------------- model ----

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


# ----------------------------------------------------------------- data ----

def load_blocks(tokenizer, split, seq_len, max_blocks):
    """wikitext-103-raw-v1 via the maintained Salesforce/wikitext mirror.

    Streaming read: the non-streaming load builds a 1.8M-row arrow table
    for the whole train split, which is the step the kernel died on in the
    previous attempt (kernel restart during split generation). Streaming
    reads rows until enough text is collected and never builds the table.
    Text is tokenized in one bounded call (~1M tokens), not over the
    whole 103M-token train set.
    """
    from datasets import load_dataset

    need_tokens = max_blocks * seq_len if max_blocks else 1_000_000
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                      split=split, streaming=True)
    parts, n_chars = [], 0
    need_chars = need_tokens * 5
    for row in ds:
        t = row["text"]
        if len(t.strip()) <= 50:
            continue
        parts.append(t)
        n_chars += len(t)
        if n_chars >= need_chars:
            break
    if not parts:
        raise RuntimeError(f"streaming load yielded no usable text for {split}")
    text = "\n\n".join(parts)
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    n = (len(ids) // seq_len) * seq_len
    ids = ids[:n].view(-1, seq_len)
    if max_blocks and len(ids) > max_blocks:
        ids = ids[:max_blocks]
    return ids


# ----------------------------------------------------------------- eval ----

def kl_sum_chunked(s_logits, t_logits, chunk=64):
    """Sum of KL(student || teacher) over next-token positions.

    Computed in sequence chunks with log_target=True so no full-vocab fp32
    softmax/log_softmax tensor is ever materialized. Exact: each chunk's
    KL is a sum over its positions, so the total equals the unchunked sum.
    Returns a tensor (differentiable) for the training loss; call .item()
    for eval. Added after the first training step OOM'd at 22GB: the
    unchunked loss held teacher + 2 student logits in fp32 plus full-size
    softmax/log_softmax temporaries (~10GB for the loss alone).
    """
    total = 0.0
    for i in range(0, s_logits.shape[1], chunk):
        ls = F.log_softmax(s_logits[:, i:i + chunk].float(), dim=-1)
        lt = F.log_softmax(t_logits[:, i:i + chunk].float(), dim=-1)
        total = total + F.kl_div(ls, lt, log_target=True, reduction="sum")
    return total


@torch.no_grad()
def evaluate(teacher, student, data, K_values, device, batch_size=8):
    """Returns {K: {'ppl': float, 'kl': float}, 'kl_by_loop_k4': [...]}."""
    teacher.eval()
    student.eval()
    out = {}
    for K in K_values:
        tot_nll = tot_tok = tot_kl = 0.0
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                t_logits = teacher(input_ids=batch).logits
                _, per_loop = student(batch, K, return_per_loop_logits=True)
            s_logits = per_loop[-1]
            shift_t = t_logits[:, :-1]
            shift_s = s_logits[:, :-1]
            labels = batch[:, 1:]
            tot_nll += F.cross_entropy(
                shift_s.float().reshape(-1, shift_s.size(-1)), labels.reshape(-1),
                reduction="sum").item()
            tot_kl += kl_sum_chunked(shift_s, shift_t).item()
            tot_tok += labels.numel()
        out[K] = {"ppl": math.exp(tot_nll / tot_tok), "kl": tot_kl / tot_tok}
    # KL at each intermediate loop within a single K=4 run (convergence trace).
    kl_by_loop = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            t_logits = teacher(input_ids=batch).logits
            _, per_loop = student(batch, 4, return_per_loop_logits=True)
        shift_t = t_logits[:, :-1]
        for s_logits in per_loop:
            shift_s = s_logits[:, :-1]
            kl_by_loop.append(kl_sum_chunked(shift_s, shift_t).item()
                              / batch[:, 1:].numel())
    n_batches = len(range(0, len(data), batch_size))
    out["kl_by_loop_k4"] = [
        sum(kl_by_loop[j::4]) / n_batches for j in range(4)
    ]
    return out


@torch.no_grad()
def teacher_ppl(teacher, data, device, batch_size=8):
    teacher.eval()
    tot_nll = tot_tok = 0.0
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = teacher(input_ids=batch).logits
        shift = logits[:, :-1].float()
        labels = batch[:, 1:]
        tot_nll += F.cross_entropy(
            shift.reshape(-1, shift.size(-1)), labels.reshape(-1),
            reduction="sum").item()
        tot_tok += labels.numel()
    return math.exp(tot_nll / tot_tok)


@torch.no_grad()
def generation_probe(teacher, student, tokenizer, device, max_new=40):
    """Greedy continuations, teacher vs student (K=2), verbatim.

    Base models do not emit safety warnings; if any safety-warning text
    appears in the outputs it is recorded verbatim here, not dismissed.
    """
    prompts = ["The city of Paris is",
               "To bake bread, first",
               "The theory of relativity states that"]
    out = {}
    for name, K in [("teacher", None), ("student_K2", 2)]:
        outs = []
        for p in prompts:
            ids = tokenizer(p, return_tensors="pt").input_ids.to(device)
            if K is None:
                gen = teacher.generate(
                    ids, max_new_tokens=max_new, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id)
                text = tokenizer.decode(gen[0], skip_special_tokens=True)
            else:
                cur = ids
                for _ in range(max_new):
                    with torch.autocast("cuda", dtype=torch.bfloat16,
                                        enabled=device.type == "cuda"):
                        logits, _ = student(cur, K)
                    nxt = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
                    cur = torch.cat([cur, nxt], dim=1)
                text = tokenizer.decode(cur[0], skip_special_tokens=True)
            outs.append({"prompt": p, "continuation": text[len(p):]})
        out[name] = outs
    return out


# -------------------------------------------------------------- checks ----

@torch.no_grad()
def startup_checks(student, eval_data, device):
    """Two fail-fast checks before any training. A broken loop-block
    attention invalidates the whole run, so these raise on failure.

    (1) Equivalence: the manual MoLA loop-block attention with prev_kv=None
        (loop 1) must reproduce the stock block call. Checked EXACTLY in
        fp32 (threshold 1e-4) — the bf16 comparison is also reported but
        only gated loosely (rel < 0.02): bf16 SDPA backends differ by
        rounding (explicit mask -> math backend vs is_causal -> flash),
        ~0.5% relative on hidden states, which is noise, not a bug
        (verified on 2026-11-26: fp32 max|diff| = 0.0, bf16 rel = 0.0057).
    (2) Causality: perturbing tokens at positions >=200 must not change
        loop-2 logits at positions <100 (the MoLA mask only admits
        previous-loop KV at positions j <= t).
    """
    student.eval()
    batch = eval_data[:8].to(device)
    seq_len = batch.shape[1]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        hidden = student.embed(batch)
        pos_emb = student._pos_emb(hidden, position_ids)
        for blk in student.in_blocks:
            hidden = student._run_block(blk, hidden, pos_emb, position_ids)
        h_stock = student._run_block(student.loop_block, hidden, pos_emb, position_ids)
        h_manual, _ = student._run_loop_block(hidden, pos_emb, position_ids, None)
        eq_diff = (h_stock.float() - h_manual.float()).abs().max().item()
        eq_rel = eq_diff / (h_stock.float().abs().max().item() + 1e-9)
    print(f"[check] loop-block equivalence bf16 (manual vs stock, loop 1) "
          f"max|diff|={eq_diff:.6f} rel={eq_rel:.6f}", flush=True)
    if eq_rel > 0.02:
        raise RuntimeError(
            f"manual loop-block attention disagrees with stock block in bf16 "
            f"(rel={eq_rel:.6f} > 0.02) — refusing to train")
    # Exact check in fp32 (bf16->fp32->bf16 round trip is exact).
    student.loop_block.to(torch.float32)
    try:
        h32 = hidden.float()
        pe32 = student._pos_emb(h32, position_ids)
        h_stock32 = student._run_block(student.loop_block, h32, pe32, position_ids)
        h_manual32, _ = student._run_loop_block(h32, pe32, position_ids, None)
        eq32 = (h_stock32 - h_manual32).abs().max().item()
    finally:
        student.loop_block.to(torch.bfloat16)
    print(f"[check] loop-block equivalence fp32 (manual vs stock, loop 1) "
          f"max|diff|={eq32:.8f}", flush=True)
    if eq32 > 1e-4:
        raise RuntimeError(
            f"manual loop-block attention disagrees with stock block in fp32 "
            f"(max|diff|={eq32:.8f} > 1e-4) — refusing to train")
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        _, per_loop = student(batch, 2, return_per_loop_logits=True)
        base = per_loop[1][:, :100, :].float().clone()
        batch2 = batch.clone()
        batch2[:, 200:] = torch.randint(
            0, 151936, batch2[:, 200:].shape, device=device)
        _, per_loop2 = student(batch2, 2, return_per_loop_logits=True)
        pert = per_loop2[1][:, :100, :].float()
        caus_diff = (base - pert).abs().max().item()
    print(f"[check] causality (perturb tokens>=200, loop-2 logits pos<100) "
          f"max|diff|={caus_diff:.6f}", flush=True)
    if caus_diff > 0.05:
        raise RuntimeError(
            f"causality violated: future tokens changed early loop-2 logits "
            f"(max|diff|={caus_diff:.6f} > 0.05) — refusing to train")
    return {"equivalence_bf16_max_abs_diff": eq_diff,
            "equivalence_bf16_rel": eq_rel,
            "equivalence_fp32_max_abs_diff": eq32,
            "causality_max_abs_diff": caus_diff}


# ----------------------------------------------------------------- save ----

def trainable_state(student):
    return {k: v.to(torch.bfloat16) for k, v in student.state_dict().items()
            if k.startswith(TRAINABLE_PREFIXES) and v.dtype.is_floating_point}


def save_model(student, tokenizer, out_dir, teacher_name, n_train, n_total):
    model_dir = os.path.join(out_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    torch.save(trainable_state(student),
               os.path.join(model_dir, "student_state.pt"))
    config = {
        "teacher": teacher_name,
        "architecture": "looped student (MoLA-style, Mixture-of-Loops arXiv:2611.34923)",
        "in_blocks": [0, 1],
        "loop_block": 12,
        "out_blocks": [22, 23],
        "loop_embedding": "2-layer MLP, hidden 256, trained from scratch",
        "embed_and_lm_head": "frozen, tied, taken from teacher",
        "mola": {
            "mechanism": "loop block attention at loop r reads [prev-loop K/V (j<=t), "
                         "current-loop K/V (j<=t)] in one shared softmax; no separate gate",
            "mask_form": "prev-loop KV at positions j<=t (MoLT-KV-style prefix memory)",
            "prev_kv": "post-rotary, single previous loop only, reused without re-rotation",
            "gradients": "flow through cross-loop KV (loop forcing, 1:1 loop-loss weights)",
            "added_params": 0,
        },
        "trainable_M": round(n_train / 1e6, 1),
        "total_M": round(n_total / 1e6, 1),
        "teacher_M": 494,
        "load": "build LoopedStudent from modeling_looped.py with the teacher, "
                "then load_state_dict(torch.load('student_state.pt'))",
    }
    with open(os.path.join(model_dir, "student_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    tokenizer.save_pretrained(os.path.join(model_dir, "tokenizer"))
    with open(os.path.join(model_dir, "modeling_looped.py"), "w") as f:
        f.write(MODELING_SOURCE)
    with open(os.path.join(model_dir, "README.md"), "w") as f:
        f.write("# Looped student (MoLA-style conversion test)\n\n"
                "Teacher: Qwen2.5-0.5B (frozen). Student: 2 input blocks + ONE "
                "shared loop block (teacher block 12) + 2 output blocks, all "
                "initialized from teacher weights, plus a loop-count embedding "
                "MLP trained from scratch.\n\n"
                "MoLA change (arXiv:2611.34923 §3.2): the loop block's attention "
                "at loop r reads the previous loop's depth KV as addressable "
                "memory — [prev-loop K/V (j<=t), current K/V (j<=t)] in one "
                "shared softmax, no separate gate.\n\n"
                "Files: student_state.pt (trainable params only, bf16), "
                "student_config.json (architecture spec), tokenizer/ "
                "(Qwen2.5 tokenizer), modeling_looped.py (LoopedStudent class).\n\n"
                "Load:\n```python\n"
                "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
                "from modeling_looped import LoopedStudent\n"
                "teacher = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B')\n"
                "student = LoopedStudent(teacher)\n"
                "student.load_state_dict(torch.load('student_state.pt'), strict=False)\n"
                "logits, _ = student(input_ids, K=2)\n```\n")
    return model_dir


def verify_saved_model(teacher, student, eval_data, device, model_dir):
    """Load the SAVED modeling_looped.py + student_state.pt back and re-eval
    a 16-seq subset with both the saved class and the training class. The
    two must agree: this catches any drift between MODELING_SOURCE and the
    training classes. Non-fatal (the training result stands regardless),
    but reported loudly in results.json."""
    sub = eval_data[:16]
    try:
        spec = importlib.util.spec_from_file_location(
            "mola_modeling_check", os.path.join(model_dir, "modeling_looped.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        student2 = mod.LoopedStudent(teacher).to(device)
        student2.load_state_dict(
            torch.load(os.path.join(model_dir, "student_state.pt")), strict=False)
        ev_orig = evaluate(teacher, student, sub, [1, 2, 3, 4], device)
        ev_saved = evaluate(teacher, student2, sub, [1, 2, 3, 4], device)
        max_kl_diff = max(abs(ev_orig[k]["kl"] - ev_saved[k]["kl"])
                          for k in (1, 2, 3, 4))
        max_ppl_rel = max(abs(ev_orig[k]["ppl"] - ev_saved[k]["ppl"])
                          / ev_orig[k]["ppl"] for k in (1, 2, 3, 4))
        ok = max_kl_diff < 0.01 and max_ppl_rel < 0.01
        print(f"[verify] saved modeling file re-eval: max KL diff={max_kl_diff:.6f}, "
              f"max PPL rel diff={max_ppl_rel:.6f} -> {'OK' if ok else 'MISMATCH'}",
              flush=True)
        return {"ok": ok, "max_kl_diff": max_kl_diff,
                "max_ppl_rel_diff": max_ppl_rel,
                "eval_orig_subset": ev_orig, "eval_saved_subset": ev_saved}
    except Exception as e:  # noqa: BLE001 - report, don't lose the run
        print(f"[verify] saved modeling file load/re-eval FAILED: {e!r}", flush=True)
        return {"ok": False, "error": repr(e)}


# ---------------------------------------------------------------- train ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--k-max", type=int, default=2)
    ap.add_argument("--eval-batches", type=int, default=6)  # x8 = 48 seqs
    ap.add_argument("--train-blocks", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/content/mola_results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # parse_known_args: `colab exec -f` leaks the kernel's own `-f <conn>.json`
    # flag into sys.argv; unknown args are ignored, not fatal.
    args, _ = ap.parse_known_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "checkpoints"), exist_ok=True)
    t0 = time.time()
    print(f"[setup] device={device} teacher={args.teacher} steps={args.steps} "
          f"batch={args.batch_size} seq={args.seq_len} lr={args.lr} k_max={args.k_max}",
          flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    print("[setup] loading teacher...", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, dtype=torch.bfloat16).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    print("[setup] building MoLA student (2 in + 1 shared loop + 2 out)...", flush=True)
    student = LoopedStudent(teacher).to(device)
    for p in student.embed.parameters():
        p.requires_grad_(False)
    for p in student.lm_head.parameters():
        p.requires_grad_(False)
    n_train = sum(p.numel() for p in student.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in student.parameters())
    print(f"[setup] student trainable={n_train/1e6:.1f}M / total={n_total/1e6:.1f}M "
          f"(teacher 494M)", flush=True)

    print("[setup] loading wikitext-103...", flush=True)
    train_data = load_blocks(tokenizer, "train", args.seq_len, args.train_blocks)
    eval_data = load_blocks(tokenizer, "validation", args.seq_len,
                           args.eval_batches * 8)
    print(f"[setup] train blocks={len(train_data)} eval blocks={len(eval_data)}",
          flush=True)

    print("[check] startup checks (equivalence + causality)...", flush=True)
    checks = startup_checks(student, eval_data, device)
    print(f"[check] startup checks passed: {checks}", flush=True)

    K_values = [1, 2, 3, 4]

    print("[eval] baseline (untrained MoLA student)...", flush=True)
    tppl = teacher_ppl(teacher, eval_data, device)
    baseline = evaluate(teacher, student, eval_data, K_values, device)
    baseline["teacher_ppl"] = tppl
    with open(os.path.join(args.out, "baseline.json"), "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"[eval] baseline teacher_ppl={tppl:.2f} " +
          " ".join(f"K{k}: ppl={v['ppl']:.2f} kl={v['kl']:.4f}"
                   for k, v in baseline.items() if isinstance(k, int)), flush=True)
    print(f"[eval] baseline kl_by_loop_k4={[round(x, 4) for x in baseline['kl_by_loop_k4']]}",
          flush=True)

    print("[train] starting...", flush=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    trainable = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps, eta_min=args.lr * 0.1)
    warmup = 20
    losses = []
    ckpts = []
    student.train()
    for step in range(1, args.steps + 1):
        if step <= warmup:
            for g in opt.param_groups:
                g["lr"] = args.lr * step / warmup
        idx = torch.randint(0, len(train_data), (args.batch_size,))
        batch = train_data[idx].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            with torch.no_grad():
                t_logits = teacher(input_ids=batch).logits
            _, per_loop = student(batch, args.k_max, return_per_loop_logits=True)
        shift_t = t_logits[:, :-1]
        loss = 0.0
        n_tok = batch[:, 1:].numel()
        for j, s_logits in enumerate(per_loop):
            shift_s = s_logits[:, :-1]
            loss = loss + kl_sum_chunked(shift_s, shift_t) / n_tok
            # Free the bf16 logits; autograd keeps only the chunk-sized graph.
            per_loop[j] = None
        loss = loss / len(per_loop)
        del shift_t, t_logits
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
        if step % 100 == 0:
            ckpt = os.path.join(args.out, "checkpoints", f"step_{step}.pt")
            torch.save(trainable_state(student), ckpt)
            ckpts.append(ckpt)
            print(f"[train] checkpoint saved {ckpt}", flush=True)
        if step % 20 == 0 or step == 1:
            print(f"[train] step {step}/{args.steps} loss={loss.item():.4f} "
                  f"lr={opt.param_groups[0]['lr']:.2e} elapsed={time.time()-t0:.0f}s",
                  flush=True)

    print("[eval] trained MoLA student...", flush=True)
    trained = evaluate(teacher, student, eval_data, K_values, device)
    trained["teacher_ppl"] = tppl

    print("[probe] greedy generation probe (teacher vs student K=2)...", flush=True)
    probe = generation_probe(teacher, student, tokenizer, device)
    for name, outs in probe.items():
        for o in outs:
            print(f"[probe:{name}] prompt={o['prompt']!r} "
                  f"continuation={o['continuation']!r}", flush=True)

    model_dir = save_model(student, tokenizer, args.out, args.teacher,
                           n_train, n_total)
    print(f"[save] model saved to {model_dir}", flush=True)

    print("[verify] loading saved modeling_looped.py back...", flush=True)
    save_verify = verify_saved_model(teacher, student, eval_data, device, model_dir)

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({
            "teacher": args.teacher,
            "experiment": "MoLA-style conversion test (arXiv:2611.34923 §3.2)",
            "mola": {
                "mechanism": "loop block attention at loop r reads [prev-loop K/V (j<=t), "
                             "current-loop K/V (j<=t)] in one shared softmax; no gate",
                "mask_form": "prev-loop KV at positions j<=t (MoLT-KV-style prefix memory)",
                "prev_kv": "post-rotary, single previous loop only",
                "gradients": "flow through cross-loop KV",
                "loop_forcing": "1:1 loop-loss weights, K_max=2",
                "added_params": 0,
            },
            "student": {"in_blocks": [0, 1], "loop_block": 12,
                        "out_blocks": [22, 23],
                        "loop_emb": "2-layer MLP, hidden 256, from scratch",
                        "trainable_M": round(n_train / 1e6, 1),
                        "total_M": round(n_total / 1e6, 1)},
            "train": {"steps": args.steps, "batch_size": args.batch_size,
                      "seq_len": args.seq_len, "lr": args.lr,
                      "k_max": args.k_max, "loss_history": losses,
                      "final_loss": losses[-1], "checkpoints": ckpts},
            "eval": {"teacher_ppl": tppl, "baseline": baseline, "trained": trained},
            "falsifier_baseline_2026_11_25": FALSIFIER_BASELINE,
            "startup_checks": checks,
            "save_verify": save_verify,
            "generation_probe": probe,
            "warnings_seen": WARNINGS_SEEN,
            "elapsed_s": round(time.time() - t0),
        }, f, indent=2)

    print(f"[eval] trained " +
          " ".join(f"K{k}: ppl={v['ppl']:.2f} kl={v['kl']:.4f}"
                   for k, v in trained.items() if isinstance(k, int)), flush=True)
    print(f"[eval] trained kl_by_loop_k4={[round(x, 4) for x in trained['kl_by_loop_k4']]}",
          flush=True)
    print(f"[eval] falsifier trained ppl "
          f"{[FALSIFIER_BASELINE['trained_ppl'][k] for k in (1, 2, 3, 4)]} kl "
          f"{FALSIFIER_BASELINE['kl_by_loop_k4']}", flush=True)
    kl_loop = trained["kl_by_loop_k4"]
    kl_falls_past_k2 = kl_loop[2] < kl_loop[1] and kl_loop[3] < kl_loop[2]
    kl_monotone = all(kl_loop[i] < kl_loop[i - 1] for i in range(1, 4))
    ppl_past = (trained[3]["ppl"] < trained[2]["ppl"]
                and trained[4]["ppl"] < trained[2]["ppl"])
    if kl_falls_past_k2 and ppl_past:
        verdict = "MoLA RESCUES (KL falls past K=2, PPL improves past K_max)"
    elif kl_falls_past_k2:
        verdict = "MIXED (KL falls past K=2, PPL does not improve past K_max)"
    elif kl_monotone:
        verdict = "MIXED (KL monotone falling, but not past K=2)"
    else:
        verdict = "DOES NOT RESCUE (falsifier verdict stands as scale-specific)"
    print(f"[verdict] {verdict} | kl_falls_past_k2={kl_falls_past_k2} "
          f"kl_monotone={kl_monotone} ppl_improves_past_kmax={ppl_past}", flush=True)
    print("[done] results at " + os.path.join(args.out, "results.json"), flush=True)
    print("RESULTS_JSON=" + json.dumps({
        "teacher_ppl": tppl, "baseline": baseline, "trained": trained,
        "generation_probe": probe, "verdict": verdict,
        "startup_checks": checks, "save_verify_ok": save_verify.get("ok")}), flush=True)


if __name__ == "__main__":
    main()
