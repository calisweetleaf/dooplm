#!/usr/bin/env python3
"""Local CPU load test for the saved Dooop-LM student (Trey-approved).

Fix vs v1: the student forward must run under CPU autocast(bf16), mirroring
the original run's CUDA autocast. Without it the freshly-built fp32
loop-count embedding promotes the bf16 hidden state to fp32 and the bf16
loop block then fails (mat1 Float vs mat2 BFloat16).
"""
import json, math, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, "/var/tmp/dooop-extract/dooop_results/model")
from modeling_looped import LoopedStudent  # noqa: E402

BASE = "/var/tmp/dooop-extract/dooop_results"
TOK = BASE + "/model/tokenizer"
STATE = BASE + "/model/student_state.pt"

torch.set_num_threads(2)


def kl_sum_chunked(s_logits, t_logits, chunk=64):
    total = 0.0
    for i in range(0, s_logits.shape[1], chunk):
        ls = F.log_softmax(s_logits[:, i:i + chunk].float(), dim=-1)
        lt = F.log_softmax(t_logits[:, i:i + chunk].float(), dim=-1)
        total = total + F.kl_div(ls, lt, log_target=True, reduction="sum")
    return total


def main():
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import pyarrow.parquet as pq

    print("[setup] loading tokenizer from extracted dir...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(TOK)

    print("[setup] reading wikitext-103 validation parquet...", flush=True)
    table = pq.read_table("/var/tmp/dooop-extract/validation.parquet")
    texts = table.column("text").to_pylist()
    parts, n_chars = [], 0
    need_chars = 8 * 256 * 5
    for t in texts:
        if len(t.strip()) <= 50:
            continue
        parts.append(t)
        n_chars += len(t)
        if n_chars >= need_chars:
            break
    text = "\n\n".join(parts)
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    n = (len(ids) // 256) * 256
    data = ids[:n].view(-1, 256)[:8]
    print(f"[setup] eval blocks={len(data)} seq_len=256 chars_used={n_chars}", flush=True)

    print("[setup] loading teacher Qwen/Qwen2.5-0.5B (bf16, cached)...", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    print("[setup] building student...", flush=True)
    student = LoopedStudent(teacher)
    student.eval()
    sd = torch.load(STATE, map_location="cpu", weights_only=True)
    res = student.load_state_dict(sd, strict=False)
    print(f"[load] state_dict keys={len(sd)} missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}", flush=True)
    print(f"[load] missing_keys sample={res.missing_keys[:5]}", flush=True)
    print(f"[load] unexpected_keys={res.unexpected_keys}", flush=True)
    dtypes = {str(v.dtype) for v in sd.values()}
    print(f"[load] state dtypes={sorted(dtypes)}", flush=True)

    ckpts = {}
    for step in (100, 200, 300, 400):
        p = f"{BASE}/checkpoints/step_{step}.pt"
        csd = torch.load(p, map_location="cpu", weights_only=True)
        ckpts[step] = {"keys": len(csd), "same_keys_as_final": set(csd.keys()) == set(sd.keys())}
        print(f"[ckpt] step_{step}: keys={len(csd)} same_keys_as_final={ckpts[step]['same_keys_as_final']}", flush=True)

    def teacher_ppl(data, bs=2):
        tot_nll = tot_tok = 0.0
        for i in range(0, len(data), bs):
            batch = data[i:i + bs]
            with torch.autocast("cpu", dtype=torch.bfloat16):
                logits = teacher(input_ids=batch).logits
            shift = logits[:, :-1].float()
            labels = batch[:, 1:]
            tot_nll += F.cross_entropy(shift.reshape(-1, shift.size(-1)), labels.reshape(-1), reduction="sum").item()
            tot_tok += labels.numel()
            del logits, shift
        return math.exp(tot_nll / tot_tok)

    def evaluate(K_values, bs=2):
        out = {}
        for K in K_values:
            tot_nll = tot_tok = tot_kl = 0.0
            for i in range(0, len(data), bs):
                batch = data[i:i + bs]
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    t_logits = teacher(input_ids=batch).logits
                    _, per_loop = student(batch, K, return_per_loop_logits=True)
                s_logits = per_loop[-1]
                shift_t, shift_s = t_logits[:, :-1], s_logits[:, :-1]
                labels = batch[:, 1:]
                tot_nll += F.cross_entropy(shift_s.float().reshape(-1, shift_s.size(-1)), labels.reshape(-1), reduction="sum").item()
                tot_kl += kl_sum_chunked(shift_s, shift_t).item()
                tot_tok += labels.numel()
                del t_logits, per_loop, s_logits
            out[K] = {"ppl": math.exp(tot_nll / tot_tok), "kl": tot_kl / tot_tok}
            print(f"[eval] K={K} ppl={out[K]['ppl']:.2f} kl={out[K]['kl']:.4f} elapsed={time.time()-t0:.0f}s", flush=True)
        kl_by_loop = []
        for i in range(0, len(data), bs):
            batch = data[i:i + bs]
            with torch.autocast("cpu", dtype=torch.bfloat16):
                t_logits = teacher(input_ids=batch).logits
                _, per_loop = student(batch, 4, return_per_loop_logits=True)
            shift_t = t_logits[:, :-1]
            for s_logits in per_loop:
                kl_by_loop.append(kl_sum_chunked(s_logits[:, :-1], shift_t).item() / batch[:, 1:].numel())
            del t_logits, per_loop
        n_batches = len(range(0, len(data), bs))
        out["kl_by_loop_k4"] = [sum(kl_by_loop[j::4]) / n_batches for j in range(4)]
        return out

    with torch.no_grad():
        tppl = teacher_ppl(data)
        print(f"[eval] teacher_ppl={tppl:.2f}", flush=True)
        res_eval = evaluate([1, 2, 3, 4])
        print(f"[eval] kl_by_loop_k4={[round(x, 4) for x in res_eval['kl_by_loop_k4']]}", flush=True)

    summary = {"teacher_ppl": tppl, "trained_8blocks": res_eval,
               "checkpoints": ckpts,
               "load": {"keys": len(sd), "missing": len(res.missing_keys), "unexpected": len(res.unexpected_keys)},
               "elapsed_s": round(time.time() - t0)}
    with open("/var/tmp/dooop-extract/loadtest_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("LOADTEST_JSON=" + json.dumps(summary), flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
