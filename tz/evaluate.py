"""Batched greedy prediction with Hugging Face generate (used for all reported metrics)."""
import torch

from .data import note_sentences, parse_output, prompt_text, row_messages
from .metrics import all_metrics


@torch.no_grad()
def predict(model, tok, df, batch_size: int = 16, max_new_tokens: int = 64):
    """Adds raw_output, pred_label, pred_evidence columns to a copy of df."""
    tok.padding_side = "left"  # decoder-only batching: pad on the left so every prompt ends at the same position
    was_training = model.training
    model.eval()
    prompts = [prompt_text(tok, row_messages(r)) for _, r in df.iterrows()]
    raw = []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                             pad_token_id=tok.pad_token_id, temperature=None, top_p=None, top_k=None)
        raw += tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    if was_training:
        model.train()
    out = df.copy()
    out["raw_output"] = raw
    parsed = [parse_output(t, r.criterion_type, len(note_sentences(r.note))) for t, (_, r) in zip(raw, df.iterrows())]
    out["pred_label"] = [p[0] for p in parsed]
    out["pred_evidence"] = [p[1] for p in parsed]
    return out


def evaluate(model, tok, df, **kw):
    pred = predict(model, tok, df, **kw)
    return all_metrics(pred), pred
