"""Evaluate a base model or base + LoRA adapter on val and/or test, with identical greedy settings.

Usage:
  python scripts/evaluate.py --model Qwen/Qwen3-4B-Instruct-2507 --name qwen3-4b-base --splits val test
  python scripts/evaluate.py --model Qwen/Qwen3-4B-Instruct-2507 --adapter runs/qwen3-4b/best --name qwen3-4b-lora --splits val test
Writes results/<name>/<split>_metrics.json and <split>_predictions.jsonl.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tz.data import load_dataset, make_split, split_frame  # noqa: E402
from tz.evaluate import evaluate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--adapter", default=None)
ap.add_argument("--name", required=True)
ap.add_argument("--splits", nargs="+", default=["val"])
ap.add_argument("--parquet", default=None)
ap.add_argument("--results", default=str(Path(__file__).resolve().parents[1] / "results"))
args = ap.parse_args()

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, device_map={"": 0})
if args.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter)

df = load_dataset(args.parquet)
split = make_split(df)
out = Path(args.results) / args.name
out.mkdir(parents=True, exist_ok=True)
for s in args.splits:
    t0 = time.time()
    m, pred = evaluate(model, tok, split_frame(df, split, s))
    m["settings"] = {"model": args.model, "adapter": args.adapter, "split": s, "decoding": "greedy",
                     "max_new_tokens": 64, "batch_size": 16, "dtype": "float16", "seconds": round(time.time() - t0, 1)}
    (out / f"{s}_metrics.json").write_text(json.dumps(m, indent=2))
    pred.drop(columns=["note"]).to_json(out / f"{s}_predictions.jsonl", orient="records", lines=True)
    L = m["overall"]["labels"]
    print(f"{args.name} {s}: acc={L['accuracy']:.3f} macroF1={L['macro_f1']:.3f} invalid={L['invalid_rate']:.3f} "
          f"evidence microF1={m['overall']['evidence']['micro']['f1']:.3f}", flush=True)
