# dooop-lm-falsifier

A fixed-point-distilled looped language model built to **falsify** the
Distill-to-Loop recipe (arXiv:2611.31977) outside vision diffusion transformers.

**Verdict: the recipe does NOT transfer to language at this scale.**
The student learns to mimic the teacher at K <= K_max, but extra loops past
K_max make it worse — the opposite of the paper's vision-DiT finding
(monotone improvement with K). Per-loop KL to the teacher rises after loop 2
and perplexity is best at K=2. This repo is a negative-result artifact: it
documents the failure mode, with the numbers, the code, and the weights to
reproduce it.

## How it was built

- **Teacher:** Qwen/Qwen2.5-0.5B (24 blocks, frozen, bf16)
- **Student:** 2 input blocks (0, 1) + ONE shared loop block (teacher block 12)
  + 2 output blocks (22, 23), all initialized from teacher weights, plus a
  loop-count embedding (2-layer MLP, hidden 256) trained from scratch.
  74.8M trainable / 210.9M total (teacher 494M). Embed + lm_head frozen (tied).
- **Loss:** per-loop KL of student logits vs teacher logits, k=1..K_max=2,
  w_k=1 (Eq. 9 analog of the paper), 400 steps, batch 8, AdamW lr 2e-5,
  wikitext-103, bf16, on a Colab L4 (1.55 CU).

## Results — held-out wikitext-103 validation (48 seqs, teacher PPL 15.72)

| K | PPL | KL-to-teacher |
|---|-----|---------------|
| 1 | 226.25 | 2.7795 |
| 2 | 222.21 | 2.7555 |
| 3 | 264.01 | 2.9194 |
| 4 | 463.53 | 3.4702 |

Both falsifier checks fail: KL rises with k past loop 2 (the loop drifts
*away* from the teacher) and PPL is best at K=2, worse at K=3 and K=4.

An independent local re-measurement (8-block slice, CPU) reproduced the
pattern: PPL 272.58 / 265.40 / 300.36 / 507.42 and KL 3.1692 / 3.1267 /
3.2389 / 3.7536 at K=1..4. Absolute levels shift with the sample size; the
shape — best at K=2, worse past it — holds. See `RESULTS.md` for the full
writeup, including the generation probe (the K=2 student degenerates into
"the city of the city of the city...").

## Follow-up: MoLA-style conversion test (2026-11-26)

The Mixture-of-Loops Transformer (arXiv:2611.34923) proposes a mechanism
for the drift: the residual stream compresses the previous loop into one
vector, and passing computed KV as addressable memory across loops
(Mixture-of-Loops Attention) should fix it. We tested it with the setup
held exactly constant — same teacher, same student shape, same loss, same
data, same 400 steps — changing only the loop block's attention to also
read the previous loop's depth KV (positions j ≤ t, one shared softmax,
no separate gate, zero added params). See `mola/`.

**Verdict: MoLA does NOT rescue the small-scale case.** KL still rises
past K=2 and PPL is best at K=2 — the same shape as the falsifier, and
slightly worse at every K:

| K | PPL | KL-to-teacher | vs falsifier PPL |
|---|-----|---------------|------------------|
| 1 | 231.53 | 2.7998 | 226.25 (+2.3%) |
| 2 | 225.93 | 2.7722 | 222.21 (+1.7%) |
| 3 | 331.00 | 3.1291 | 264.01 (+25%) |
| 4 | 623.32 | 3.7413 | 463.53 (+34%) |

The extra memory does not prevent the shared block from drifting with each
extra application; at this scale it mildly amplifies the drift past K_max.
The paper's own runs are 100B–400B tokens; this test is 400 steps (~1M
tokens) — the rescue, if it exists, does not show up at this budget. The
falsifier's verdict stands as scale-specific. Full writeup in
`mola/RESULTS.md`, including the verification (manual attention proved
exactly identical to the stock block in fp32, causality exact at 0.0).

## Files

- `run_falsifier.py` — training + eval script (Colab)
- `load_test.py` — independent local load test + re-measurement
- `modeling_looped.py` — the `LoopedStudent` class
- `student_config.json` — architecture spec
- `RESULTS.md` — full writeup with run notes and the safety-warning record
- `results.json`, `baseline.json` — numbers
- `mola/` — the MoLA follow-up: `run_mola.py`, `RESULTS.md`,
  `results.json`, `baseline.json`, `modeling_looped.py`,
  `student_config.json`

## Weights

The trained weights (`student_state.pt`, 149 MB, trainable params only, bf16)
are coming to Hugging Face under **Somnus-Sovereign-Systems** (public) — they are
over GitHub's 100 MB per-file limit, so they don't live in this repo.
Checkpoints (steps 100/200/300/400) are in the experiment archive.

Load the weights (requires `trust_remote_code=True` for
`modeling_looped.py`):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from modeling_looped import LoopedStudent
teacher = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B')
student = LoopedStudent(teacher)
student.load_state_dict(torch.load('student_state.pt'), strict=False)
logits, _ = student(input_ids, K=2)
```

The weights are the trainable params only; `embed` and `lm_head` come from
the teacher (frozen, tied), which is why `load_state_dict` runs with
`strict=False` and reports 2 missing keys.

## Credit

Built end-to-end by **Vanta**, an AI agent, under the direction of
**Christian Trey Levi Rowell** — paper read, falsifier designed, experiment
run, model saved, verified, and published. The human directed; the agent
did the doing.

## Limitations

- One student size (0.5B teacher), one data slice (~1M tokens),
  400 steps. A larger student, more data, or more steps might behave
  differently — this falsifies the recipe *as implemented here*, not
  looping in general.
- Base model only: no instruction tuning, no safety layer, and its free
  generation degenerates into repetition. Do not deploy as a chatbot.
