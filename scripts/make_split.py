"""Build and save the fixed 37/8/8 patient split, plus token-length stats for the prompts.

Usage: python scripts/make_split.py [--parquet PATH]
Writes data/split_manifest.json.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tz.data import load_dataset, make_split, prompt_text, row_messages, split_frame, target_text  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--parquet", default=None)
ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B-Instruct-2507")
args = ap.parse_args()

df = load_dataset(args.parquet)
split = make_split(df)
manifest = {"seed": 42, "method": "sorted unique patient_id, np.random.default_rng(42).shuffle, first 37 / next 8 / last 8",
            "total_rows": len(df), "total_patients": int(df.patient_id.nunique())}
for name in ("train", "val", "test"):
    d = split_frame(df, split, name)
    manifest[name] = {"patients": split[name], "n_patients": len(split[name]), "n_rows": len(d),
                      "labels": d.expert_eligibility.value_counts().to_dict(),
                      "criterion_type": d.criterion_type.value_counts().to_dict()}
assert sum(manifest[s]["n_rows"] for s in ("train", "val", "test")) == len(df)

from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(args.tokenizer)
lens = [len(tok(prompt_text(tok, row_messages(r)) + target_text(r.expert_eligibility, r.expert_sentences),
                add_special_tokens=False).input_ids) for _, r in df.iterrows()]
manifest["token_lengths"] = {"tokenizer": args.tokenizer, "max": max(lens), "mean": round(sum(lens) / len(lens), 1),
                             "note": "prompt + target; no row is truncated at max_len 1024"}
assert max(lens) <= 1024

out = Path(__file__).resolve().parents[1] / "data" / "split_manifest.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(manifest, indent=2))
print(json.dumps({k: (v if k not in ("train", "val", "test") else {kk: vv for kk, vv in v.items() if kk != "patients"})
                  for k, v in manifest.items()}, indent=1))
