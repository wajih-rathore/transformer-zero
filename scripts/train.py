"""LoRA fine-tuning on the 37-patient training split, validating on the 8 val patients after every epoch.

Usage (Colab T4):
  python scripts/train.py --model Qwen/Qwen3-4B-Instruct-2507 --out runs/qwen3-4b
  python scripts/train.py --model Qwen/Qwen3-1.7B --out runs/qwen3-1.7b
  python scripts/train.py --model Qwen/Qwen3-0.6B --out runs/smoke --smoke   # CPU pipeline check, tiny random model
Writes <out>/best (adapter with the best val macro-F1), <out>/train_log.json, <out>/val_*.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tz import DTYPE  # noqa: E402
from tz.data import load_dataset, make_split, prompt_text, row_messages, split_frame, target_text  # noqa: E402
from tz.evaluate import evaluate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=float, default=3)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--max-steps", type=int, default=-1)
ap.add_argument("--parquet", default=None)
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import (AutoModelForCausalLM, AutoTokenizer, Qwen3Config, Qwen3ForCausalLM,  # noqa: E402
                          Trainer, TrainerCallback, TrainingArguments)

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
df = load_dataset(args.parquet)
split = make_split(df)
train_df, val_df = split_frame(df, split, "train"), split_frame(df, split, "val")
tok = AutoTokenizer.from_pretrained(args.model)

if args.smoke:  # pipeline check only: random tiny model, real tokenizer, a few rows
    train_df, val_df = train_df.head(16), val_df.head(8)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                         num_attention_heads=4, num_key_value_heads=2, head_dim=16))
else:
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=DTYPE, device_map={"": 0})
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model = get_peft_model(model, LoraConfig(
    r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()


def encode(r):
    """Prompt tokens get label -100, so the loss only covers the JSON answer and the end-of-turn token."""
    p = tok(prompt_text(tok, row_messages(r)), add_special_tokens=False).input_ids
    t = tok(target_text(r.expert_eligibility, r.expert_sentences) + "<|im_end|>", add_special_tokens=False).input_ids
    return {"input_ids": p + t, "labels": [-100] * len(p) + t}


def collate(batch):
    n = max(len(b["input_ids"]) for b in batch)
    pad = lambda xs, v: xs + [v] * (n - len(xs))  # noqa: E731
    return {"input_ids": torch.tensor([pad(b["input_ids"], tok.pad_token_id) for b in batch]),
            "labels": torch.tensor([pad(b["labels"], -100) for b in batch]),
            "attention_mask": torch.tensor([pad([1] * len(b["input_ids"]), 0) for b in batch])}


class ValidateEachEpoch(TrainerCallback):
    def __init__(self):
        self.best, self.history = -1.0, []

    def on_epoch_end(self, _args, state, control, model=None, **kw):
        t0 = time.time()
        m, pred = evaluate(model, tok, val_df)
        rec = {"epoch": round(state.epoch, 2), "step": state.global_step,
               "val_macro_f1": m["overall"]["labels"]["macro_f1"], "val_accuracy": m["overall"]["labels"]["accuracy"],
               "val_invalid_rate": m["overall"]["labels"]["invalid_rate"],
               "val_evidence_micro_f1": m["overall"]["evidence"]["micro"]["f1"], "eval_s": round(time.time() - t0, 1)}
        self.history.append(rec)
        print("VAL", rec, flush=True)
        if rec["val_macro_f1"] > self.best:
            self.best = rec["val_macro_f1"]
            model.save_pretrained(out / "best")
            (out / "val_metrics.json").write_text(json.dumps(m, indent=2))
            pred.drop(columns=["note"]).to_json(out / "val_predictions.jsonl", orient="records", lines=True)


cb = ValidateEachEpoch()
trainer = Trainer(
    model=model, train_dataset=[encode(r) for _, r in train_df.iterrows()], data_collator=collate, callbacks=[cb],
    args=TrainingArguments(output_dir=str(out / "trainer"), per_device_train_batch_size=args.batch,
                           gradient_accumulation_steps=args.accum, num_train_epochs=args.epochs, max_steps=args.max_steps,
                           learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=0.05, logging_steps=5,
                           fp16=torch.cuda.is_available() and DTYPE == torch.float16, bf16=DTYPE == torch.bfloat16, save_strategy="no", report_to="none", seed=42,
                           remove_unused_columns=False))
t0 = time.time()
trainer.train()
log = {"args": vars(args), "train_rows": len(train_df), "val_rows": len(val_df), "train_s": round(time.time() - t0, 1),
       "best_val_macro_f1": cb.best, "val_history": cb.history, "trainer_log": trainer.state.log_history,
       "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None}
(out / "train_log.json").write_text(json.dumps(log, indent=2))
print("DONE", {k: log[k] for k in ("train_s", "best_val_macro_f1", "peak_mem_mb")})
