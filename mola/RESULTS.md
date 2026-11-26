# MoLA-style conversion test — RESULTS

**Date:** 2026-11-26
**Verdict: DOES NOT RESCUE (falsifier verdict stands as scale-specific)**

The MoLA change does not fix the small-scale drift. Per-loop KL still rises
past K=2 (2.77 → 3.13 → 3.74) and PPL still rises past K_max=2 (226 → 331 →
623) — the same shape as the falsifier, and slightly *worse* than the plain
Distill-to-Loop student at every K. Addressable per-layer KV memory across
loops does not rescue the looped-LM conversion at this size, data slice, and
step budget. That is a result, not a failure: the test was designed to be
falsifiable, and it falsified the rescue hypothesis.

## Setup

- **Teacher:** Qwen/Qwen2.5-0.5B (24 blocks, frozen), loaded in bf16. Held-out
  teacher PPL on wikitext-103 validation: **15.72** (same eval set as the
  falsifier, same number to 2 decimals).
- **Student:** 2 input blocks (0,1) + ONE shared loop block (12) + 2 output
  blocks (22,23), all initialized from teacher weights; loop-count embedding =
  2-layer MLP (hidden 256) trained from scratch. **74.8M trainable / 210.9M
  total** (teacher 494M). Embed + lm_head frozen (tied). Identical to the
  falsifier's student in every parameter count.
- **MoLA change (Mixture-of-Loops Transformer, arXiv:2611.34923, §3.2):**
  the shared loop block's attention at loop r reads the previous loop's
  depth KV as addressable memory — keys/values are [prev-loop K/V
  (positions j ≤ t), current-loop K/V (positions j ≤ t)] in a single shared
  softmax, with no separate learned gate. Previous-loop keys keep the
  writer's rotary application and are reused without re-rotation.
  Mask form: prev-loop KV at positions j ≤ t (MoLT-KV-style prefix memory),
  single previous loop only. Gradients flow through the cross-loop KV
  (loop forcing, §3.4). **Zero added parameters.**
- **Loss:** per-loop KL of student logits vs teacher logits, k=1..K_max=2,
  w_k=1 (loop forcing 1:1), computed in 64-position chunks (`kl_sum_chunked`,
  log-target KL — exact, see falsifier run notes).
- **Data:** wikitext-103-raw-v1 (Salesforce/wikitext mirror), streamed.
  4000 train blocks × 256 tokens (~1.02M tokens), 48 eval blocks. Same
  as the falsifier.
- **Train:** 400 steps, batch **8**, seq 256, AdamW lr 2e-5 (20-step warmup,
  cosine to 2e-6), bf16 autocast. Final train loss **2.6145** (from 9.7858
  at step 1; falsifier: 2.5942 from 9.6867).
- **Eval:** held-out wikitext-103 validation, per-loop KL + PPL at K=1..4.

## Results — held-out wikitext-103 validation (48 seqs × 256 tokens)

Teacher PPL 15.72 in both rows (same teacher, same eval set).

### Baseline (untrained MoLA student)

| K | PPL | KL-to-teacher |
|---|-----|---------------|
| 1 | 319,391.92 | 9.8486 |
| 2 | 178,617.16 | 9.3495 |
| 3 | 504,727.05 | 10.3823 |
| 4 | 1,024,553.18 | 11.0564 |

kl_by_loop_k4 = [9.8486, 9.3495, 10.3823, 11.0564]

Essentially identical to the falsifier's untrained baseline (319,511 /
9.8490, 178,425 / 9.3431, 922,700 / 10.9749, 2,621,492 / 11.9785) — the
MoLA memory changes nothing before training, as expected.

### Trained (400 steps)

| K | PPL | KL-to-teacher | vs falsifier PPL | vs falsifier KL |
|---|-----|---------------|------------------|-----------------|
| 1 | 231.53 | 2.7998 | 226.25 (+2.3%) | 2.7795 (+0.020) |
| 2 | 225.93 | 2.7722 | 222.21 (+1.7%) | 2.7555 (+0.017) |
| 3 | 331.00 | 3.1291 | 264.01 (+25%) | 2.9194 (+0.210) |
| 4 | 623.32 | 3.7413 | 463.53 (+34%) | 3.4702 (+0.271) |

kl_by_loop_k4 = [2.7998, 2.7722, 3.1291, 3.7413]
(falsifier: [2.7795, 2.7555, 2.9194, 3.4702])

### Falsifier checks (both fail — same as the falsifier)

1. **KL decreases with k past K=2? NO.** kl_by_loop rises after loop 2:
   2.80, 2.77, 3.13, 3.74. Intermediate-loop outputs do NOT converge
   toward the teacher.
2. **PPL improves with K past K_max=2? NO.** Best PPL is at K=2 (225.93);
   K=3 (331.00) and K=4 (623.32) are worse — and worse than the
   falsifier's K=3/K=4 by 25%/34%.

The MoLA student is a good K≤2 mimic (PPL 226–232 vs teacher 15.7) and a
worse extrapolator than the plain student. The extra memory does not
prevent the shared block from drifting with each extra application; at
this scale it mildly amplifies the drift past K_max. The paper's own
results are at 100B–400B tokens; this test is 400 steps (~1M tokens) —
the rescue, if it exists, does not show up at this budget.

### Loss trajectory (train, mean per-loop KL)

9.79 (step 1) → 6.28 (20) → 3.97 (40) → 3.13 (80) → 2.97 (100) → 2.77
(200) → 2.79 (300) → 2.61 (400). Same shape as the falsifier
(9.69 → 2.59), plateau around 2.6–2.9. Full 400-step history in
`results.json` (`train.loss_history`).

## Generation probe (greedy, 40 tokens, verbatim)

Teacher (coherent, base-model text — identical outputs to the falsifier run):

- "The city of Paris is" →
  " planning to build a new park. The park will have a circular path with a
  radius of 100 meters. The city wants to plant trees along the path, with
  each tree being 1"
- "To bake bread, first" →
  ", you need to make a dough. The dough is a mixture of flour and water. You
  can make the dough by hand or with a machine. The dough is shaped into a
  loaf. You can"
- "The theory of relativity states that" →
  " the speed of light is constant in all inertial frames of reference. This
  means that if you are moving at a constant speed, the speed of light will
  appear to slow down for you. This is"

Student, K=2 (degenerate — repetition loops, no coherent text; different
degenerate tokens than the falsifier's "the city of the city", same failure):

- "The city of Paris is" →
  " the 1999-1999. 1999. 1999. 2009. 2000. 200"
- "To bake bread, first" →
  " to the 1999. 1999. 2000. 2000. 2000. 2000. 2"
- "The theory of relativity states that" →
  " the way the way of the 1988. 1988, 1988. 1988, 1988. 198"

**No safety-warning text was generated by either model.** Same as the
falsifier: both are base models, no safety layer to trigger. The run-log
warnings are the same three as the falsifier's (HF_TOKEN vault timeout,
unauthenticated Hub requests, sequence-length notice) plus
`DeprecationWarning: datetime.datetime.utcnow()` noise from jupyter_client
— 75 warnings recorded in `results.json` (`warnings_seen`), all benign.

## Verification — what was checked before training

The loop block's attention is implemented by hand (the stock block is a
black box that cannot take cross-loop KV), so two fail-fast checks run
before any training. Both passed:

1. **Equivalence:** manual attention with no previous-loop KV (loop 1)
   reproduces the stock block call. In **fp32 the max difference is
   exactly 0.00000000** — the implementation is mathematically identical.
   In bf16 the max difference is 0.046875 (relative 0.0061), which is
   bf16 SDPA-backend rounding noise (explicit mask → math backend vs
   is_causal → flash backend), not a bug.
2. **Causality:** perturbing tokens at positions ≥200 changes loop-2
   logits at positions <100 by exactly **0.000000** — the MoLA mask admits
   no future information. A causal leak would have invalidated the eval.

And after saving: the saved `modeling_looped.py` + `student_state.pt` were
loaded back and re-evaluated on a 16-seq subset — max KL difference
**0.000000**, max PPL relative difference **0.000000** vs the training
classes (`save_verify.ok: true` in `results.json`). The saved artifact
reproduces the run exactly.

## Run notes — what broke and what fixed it

1. **transformers 5.22.1 renamed Qwen2Attention's attributes (first run).**
   `Qwen2Attention` in this install has no `num_heads` /
   `num_key_value_heads` attributes (only `head_dim`, `scaling`,
   `attention_dropout`). The script read head counts off the attention
   module and died with `AttributeError` at the startup check, before any
   training. Fix: read head counts from the model config
   (`num_attention_heads` = 14, `num_key_value_heads` = 2,
   `head_dim` = `hidden_size // num_attention_heads` = 64). Note for the
   record: Qwen2.5-0.5B's head_dim is **64**, not 128 (896 / 14).
2. **bf16 equivalence threshold too tight (second run).** The startup
   equivalence check gated on absolute bf16 difference (0.02) and the
   manual attention measured 0.046875 — the check refused to train. A
   diagnostic run showed fp32 max difference is exactly 0.0, i.e. the
   implementation is correct and the bf16 gap is backend rounding noise.
   Fix: gate the equivalence check on the fp32 comparison (threshold
   1e-4, exact) and report the bf16 number with only a loose relative
   gate (rel < 0.02).
3. **`colab ssh` is unusable in this environment** (publickey denied).
   Shell work on the VM goes through `colab exec` with IPython `!` lines,
   and `exec` needs `--timeout 0` — the default 30 s idle timeout killed
   the tar step mid-run (the tar itself had finished; the timeout only
   cut the wait for output). Lesson: always `--timeout 0` on exec.
4. **`colab run` executes the script inside the kernel** (tracebacks show
   `/tmp/ipykernel_*/...py`), not as a real file — `inspect.getsource`
   would not have worked there. The saved `modeling_looped.py` is a static
   string, verified verbatim against the training classes by AST before
   launch and by the load-back re-eval after saving (both passed).

## Model persistence

Saved on the Colab VM at `/content/mola_results/` (verified listing):

- `model/student_state.pt` — 149,642,957 bytes (trainable params only, bf16)
- `model/student_config.json`, `model/modeling_looped.py` (standalone
  `LoopedStudent` with the MoLA attention, verified identical to the
  training classes), `model/tokenizer/` (Qwen2.5 tokenizer),
  `model/README.md`
- `checkpoints/step_100.pt`, `step_200.pt`, `step_300.pt`, `step_400.pt` —
  149,642,533 bytes each (trainable state at each checkpoint)
- `results.json` (36,691 bytes), `baseline.json` (448 bytes)

Persisted out of Colab to this directory:

- `results.json`, `baseline.json` — downloaded, sizes match the remote
  listing exactly.
- `mola_results.tar.gz` — 596,035,531 bytes, contains the full
  `mola_results/` tree above (model + all 4 checkpoints).
  **Verified on disk here:** size matches the remote byte-for-byte,
  `gzip -t` integrity OK, and a read-only manifest listing confirms every
  file (baseline.json, results.json, step_100/200/300/400.pt,
  model/student_state.pt, student_config.json, modeling_looped.py,
  tokenizer/{chat_template.jinja, tokenizer.json, tokenizer_config.json},
  README.md). sha256 `5decaf0ca50a4049f030376b3214032d0997c58b8f38daf0447afd7557bf3275`.
  (The tarball also contains a `model/__pycache__/` .pyc swept in by the
  save-verify import — harmless.)
- `run_mola.py` (44,132 bytes), `diag_mola.py` (diagnostic script for the
  equivalence investigation) — in this directory.

Load the saved student (from `model/README.md`):
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from modeling_looped import LoopedStudent
teacher = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B')
student = LoopedStudent(teacher)
student.load_state_dict(torch.load('student_state.pt'), strict=False)
logits, _ = student(input_ids, K=2)
```

## Cost

- GPU: L4 (22 GB), 2.33 CU/hr. Balance 13.30 CU at approval (2026-11-26
  15:52 CST), **13.25 CU before the successful run** (two failed attempts
  + diagnostic burned 0.05 CU), final balance **12.67 CU** after stop —
  **0.63 CU total** for the whole MoLA effort (failed attempts +
  diagnostic + successful run).
  Successful run script time: 281 s (setup + baseline + 400 steps at
  ~0.55 s/step + final eval + probe + save + save-verify).
- Session `mola-test` stopped; `sessions` list empty, usage rate 0.00/hr,
  0 active assignments.
