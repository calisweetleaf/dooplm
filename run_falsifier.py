#!/usr/bin/env python3
"""
Dooop (Distill to Loop, arXiv:2611.31977) falsifier test, language edition.

Question: does the Dooop conversion recipe transfer from vision DiTs to language?
  - Teacher: Qwen2.5-0.5B (24 transformer blocks), frozen.
  - Student: 2 input blocks + ONE shared loop block + 2 output blocks,
    all initialized from teacher weights (loop block = teacher block 12),
    plus a small loop-count embedding MLP trained from scratch.
  - Loss: per-loop KL of student logits vs teacher logits, k=1..K_max=2,
    w_k=1 (Eq. 9 analog).
  - Eval: held-out wikitext-103, per-loop KL-to-teacher and PPL at K=1..4
    (past K_max=2). Falsifier: if per-loop KL does NOT decrease with k,
    or PPL does NOT improve with K beyond K_max, the recipe is
    vision-DiT-specific.

Falsifier-first: NO benchmark eval, NO big training run. A few hundred steps.

Saves: checkpoints every 100 steps + final weights + student config +
tokenizer + standalone modeling file, all under --out. Also runs a small
greedy generation probe (teacher vs student) and records outputs verbatim,
so any safety-warning text the model emits is captured, not dismissed.
Writes results.json (+ baseline.json) and prints a JSON summary to stdout.
"""
import argparse
import copy
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

# Standalone copy of the student architecture, saved into the model dir so the
# weights are loadable without this training script. Kept identical to the
# classes below; verified by inspection at save time (see save_model()).
MODELING_SOURCE = '''"""Looped student architecture for the Dooop-LM falsifier run.

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
'''


# ---------------------------------------------------------------- model ----

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
        "architecture": "looped student (Dooop recipe, language edition)",
        "in_blocks": [0, 1],
        "loop_block": 12,
        "out_blocks": [22, 23],
        "loop_embedding": "2-layer MLP, hidden 256, trained from scratch",
        "embed_and_lm_head": "frozen, tied, taken from teacher",
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
        f.write("# Looped student (Dooop-LM falsifier)\n\n"
                "Teacher: Qwen2.5-0.5B (frozen). Student: 2 input blocks + ONE "
                "shared loop block (teacher block 12) + 2 output blocks, all "
                "initialized from teacher weights, plus a loop-count embedding "
                "MLP trained from scratch.\n\n"
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
    ap.add_argument("--out", default="/content/dooop_results")
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

    print("[setup] building student (2 in + 1 shared loop + 2 out)...", flush=True)
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

    K_values = [1, 2, 3, 4]

    print("[eval] baseline (untrained student)...", flush=True)
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

    print("[eval] trained student...", flush=True)
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

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({
            "teacher": args.teacher,
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
            "generation_probe": probe,
            "warnings_seen": WARNINGS_SEEN,
            "elapsed_s": round(time.time() - t0),
        }, f, indent=2)

    print(f"[eval] trained " +
          " ".join(f"K{k}: ppl={v['ppl']:.2f} kl={v['kl']:.4f}"
                   for k, v in trained.items() if isinstance(k, int)), flush=True)
    print(f"[eval] trained kl_by_loop_k4={[round(x, 4) for x in trained['kl_by_loop_k4']]}",
          flush=True)
    kl_dec = trained["kl_by_loop_k4"][-1] < trained["kl_by_loop_k4"][0]
    ppl_past = (trained[3]["ppl"] < trained[2]["ppl"]
                and trained[4]["ppl"] < trained[2]["ppl"])
    verdict = ("TRANSFERS (preliminary)" if (kl_dec and ppl_past) else
               "MIXED" if (kl_dec or ppl_past) else
               "DOES NOT TRANSFER (vision-DiT-specific)")
    print(f"[verdict] {verdict} | kl_decreases_with_k={kl_dec} "
          f"ppl_improves_past_kmax={ppl_past}", flush=True)
    print("[done] results at " + os.path.join(args.out, "results.json"), flush=True)
    print("RESULTS_JSON=" + json.dumps({
        "teacher_ppl": tppl, "baseline": baseline, "trained": trained,
        "generation_probe": probe, "verdict": verdict}), flush=True)


if __name__ == "__main__":
    main()
