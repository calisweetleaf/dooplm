"""Looped student architecture for the Dooop-LM falsifier run.

Student: 2 input blocks + ONE shared loop block + 2 output blocks, all
initialized from Qwen2.5-0.5B teacher weights (loop block = teacher block 12),
plus a small loop-count embedding MLP trained from scratch.
Load with: build with the teacher's layers, then load_state_dict() from
student_state.pt (trainable params only, bf16).
"""
import copy
import torch
import torch.nn as nn


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
        # its blocks as trainable.
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
        for k in range(1, K + 1):
            hidden = hidden + self.loop_emb(k)
            hidden = self._run_block(self.loop_block, hidden, pos_emb, position_ids)
            if return_per_loop_logits:
                h = hidden
                for blk in self.out_blocks:
                    h = self._run_block(blk, h, pos_emb, position_ids)
                per_loop_logits.append(self.lm_head(self.norm(h)))
        h = hidden
        for blk in self.out_blocks:
            h = self._run_block(blk, h, pos_emb, position_ids)
        return self.lm_head(self.norm(h)), per_loop_logits
