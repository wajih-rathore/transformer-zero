"""Tasks 3-5 on the real models (GPU): correctness against Hugging Face, then KV-cache,
attention and speculative-decoding benchmarks. Results are written after every part.

Usage:
  python scripts/benchmark.py --target Qwen/Qwen3-4B-Instruct-2507 --draft Qwen/Qwen3-0.6B \
      --adapter runs/qwen3-4b/best --out results/bench_qwen3-4b.json
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tz import DTYPE  # noqa: E402
from tz.attention import reference_attention, tiled_attention  # noqa: E402
from tz.data import load_dataset, make_split, note_sentences, prompt_text, row_messages, split_frame  # noqa: E402
from tz.decoder import Qwen3Decoder, greedy_generate  # noqa: E402
from tz.explain import explanation_messages  # noqa: E402
from tz.speculative import speculative_generate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--target", required=True)
ap.add_argument("--draft", default="Qwen/Qwen3-0.6B")
ap.add_argument("--adapter", default=None)
ap.add_argument("--out", required=True)
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--parts", nargs="+", default=["correctness", "kv", "attention", "spec"])
ap.add_argument("--parquet", default=None)
args = ap.parse_args()

import transformers  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

dev = torch.device("cuda")
tok = AutoTokenizer.from_pretrained(args.target)
hf = AutoModelForCausalLM.from_pretrained(args.target, dtype=DTYPE, device_map={"": 0}).eval()
if args.adapter:
    from peft import PeftModel
    hf = PeftModel.from_pretrained(hf, args.adapter).eval()
dec = Qwen3Decoder(hf)
lora = bool(args.adapter)
eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}

df = load_dataset(args.parquet)
val = split_frame(df, make_split(df), "val")  # benchmarks never touch the test split
pool = tok("\n".join(val.note.unique()), add_special_tokens=False).input_ids
while len(pool) < 4096:
    pool = pool + pool
ids = lambda n: torch.tensor([pool[:n]], device=dev)  # noqa: E731
enc = lambda text: tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)  # noqa: E731

res = {"env": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "transformers": transformers.__version__,
               "dtype": str(DTYPE), "target": args.target, "adapter": args.adapter, "draft": args.draft, "repeats": args.repeats}}
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
save = lambda: out.write_text(json.dumps(res, indent=2))  # noqa: E731


def first_mismatch(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None if len(a) == len(b) else min(len(a), len(b)))


def repeated(fn):
    """Warm-up once, then average the numeric stats over `repeats` runs."""
    fn()
    runs = [fn() for _ in range(args.repeats)]
    toks = runs[-1][0]
    stats = {k: st.mean(r[1][k] for r in runs) for k, v in runs[-1][1].items() if isinstance(v, (int, float))}
    stats["total_s_std"] = st.pstdev(r[1]["total_s"] for r in runs)
    return toks, stats


if "correctness" in args.parts:  # D2: ours cached == ours uncached == HF generate, 3+ length pairs
    rows = []
    for n_in, n_out in [(64, 32), (256, 64), (1024, 128)]:
        x = ids(n_in)
        cached, _ = greedy_generate(dec, x, n_out, use_cache=True, lora=lora)
        uncached, _ = greedy_generate(dec, x, n_out, use_cache=False, lora=lora)
        theirs = hf.generate(input_ids=x, max_new_tokens=n_out, min_new_tokens=n_out, do_sample=False,
                             pad_token_id=tok.pad_token_id, temperature=None, top_p=None, top_k=None)[0, n_in:].tolist()
        with torch.no_grad():
            diff = (dec.forward(x, lora=lora)[0, -1] - hf(input_ids=x).logits[0, -1].float()).abs().max().item()
        rows.append({"input_tokens": n_in, "output_tokens": n_out, "cached_eq_uncached": cached == uncached,
                     "cached_eq_hf": cached == theirs, "first_mismatch_vs_uncached": first_mismatch(cached, uncached),
                     "first_mismatch_vs_hf": first_mismatch(cached, theirs), "last_logit_max_abs_diff_vs_hf": diff})
        print("correctness", rows[-1], flush=True)
    res["correctness"] = rows
    save()

if "kv" in args.parts:  # D3: cached vs uncached, several input/output lengths
    rows = []
    for n_in in (128, 512, 1024):
        for n_out in (32, 128):
            for use_cache in (True, False):
                _, s = repeated(lambda: greedy_generate(dec, ids(n_in), n_out, use_cache=use_cache, lora=lora))
                rows.append({"input_tokens": n_in, "output_tokens": n_out, "cache": use_cache, **s})
                print("kv", rows[-1], flush=True)
    res["kv_cache"] = rows
    save()

if "attention" in args.parts:  # E3: reference vs tiled vs PyTorch SDPA, causal, growing lengths
    H, D = dec.n_heads, dec.head_dim
    rows = []

    def timed(fn, reps=5):
        fn()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        ms = []
        for _ in range(reps):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            fn()
            b.record()
            torch.cuda.synchronize()
            ms.append(a.elapsed_time(b))
        return {"ms_mean": st.mean(ms), "ms_std": st.pstdev(ms), "peak_extra_mb": (torch.cuda.max_memory_allocated() - base) / 2**20}

    for T in (128, 512, 1024, 2048, 4096, 8192):
        torch.manual_seed(0)
        q, k, v = (torch.randn(1, H, T, D, device=dev, dtype=DTYPE) for _ in range(3))
        try:
            gold = reference_attention(q.float(), k.float(), v.float())  # float32 ground truth
        except torch.cuda.OutOfMemoryError:
            gold = None
        variants = {"reference": lambda: reference_attention(q, k, v),
                    "tiled_64": lambda: tiled_attention(q, k, v, block_q=64, block_k=64),
                    "tiled_128": lambda: tiled_attention(q, k, v, block_q=128, block_k=128),
                    "sdpa": lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True)}  # square case: top-left == bottom-right
        for name, fn in variants.items():
            row = {"seq_len": T, "heads": H, "head_dim": D, "variant": name}
            try:
                o = fn()
                if gold is not None:
                    err = (o.float() - gold).abs()
                    row.update(max_abs_err=err.max().item(), mean_abs_err=err.mean().item())
                del o
                row.update(timed(fn))
            except torch.cuda.OutOfMemoryError:
                row["oom"] = True
            torch.cuda.empty_cache()
            rows.append(row)
            print("attention", row, flush=True)
        del q, k, v, gold
        torch.cuda.empty_cache()
    in_model = []
    x = ids(256)
    base_toks = None
    for name in ("reference", "tiled", "sdpa"):
        dec.attention = name
        toks, s = repeated(lambda: greedy_generate(dec, x, 64, lora=lora))
        base_toks = base_toks or toks
        in_model.append({"attention": name, "same_tokens_as_reference": toks == base_toks, **s})
        print("attention-in-model", in_model[-1], flush=True)
    dec.attention = "reference"
    res["attention"] = {"standalone": rows, "in_model": in_model}
    save()

if "spec" in args.parts:  # F3: speculative vs plain greedy on the target, same prompts
    dtok = AutoTokenizer.from_pretrained(args.draft)
    assert dtok.get_vocab() == tok.get_vocab(), "draft and target tokenizers differ"
    draft = Qwen3Decoder(AutoModelForCausalLM.from_pretrained(args.draft, dtype=DTYPE, device_map={"": 0}).eval())
    rows = val[val.expert_sentences.map(len) > 0].head(5)
    tasks = {"explanation": ([enc(prompt_text(tok, explanation_messages(
        r.criterion_type, r.criterion_text, r.expert_eligibility,
        [(i, note_sentences(r.note)[i]) for i in r.expert_sentences]))) for _, r in rows.iterrows()], 96, False)}
    if lora:
        tasks["classification_json"] = ([enc(prompt_text(tok, row_messages(r))) for _, r in rows.iterrows()], 48, True)
    res["speculative"] = []
    for task, (prompts, n_out, use_lora) in tasks.items():
        for j, x in enumerate(prompts):
            g_toks, g = repeated(lambda: greedy_generate(dec, x, n_out, eos_ids=eos, lora=use_lora))
            for k in ((2, 4, 6, 8) if task == "explanation" else (4,)):
                s_toks, s = repeated(lambda: speculative_generate(dec, draft, x, n_out, k=k, eos_ids=eos, target_lora=use_lora))
                row = {"task": task, "prompt": j, "k": k, "same_tokens_as_greedy": s_toks == g_toks,
                       "first_mismatch": first_mismatch(s_toks, g_toks), "new_tokens": len(s_toks),
                       "acceptance_rate": s["acceptance_rate"], "mean_accepted_per_round": s["mean_accepted_per_round"],
                       "greedy_total_s": g["total_s"], "spec_total_s": s["total_s"],
                       "greedy_tok_per_s": g["new_tokens"] / g["total_s"], "spec_tok_per_s": s["tok_per_s"],
                       "speedup": g["total_s"] / s["total_s"], "spec_peak_mem_mb": s["peak_mem_mb"],
                       "greedy_peak_mem_mb": g["peak_mem_mb"]}
                res["speculative"].append(row)
                print("spec", row, flush=True)
                save()
print("saved", out)
