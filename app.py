"""Transformer Zero: browser app (Gradio, runs on a Hugging Face ZeroGPU Space or any GPU box).

Env vars (defaults are the deployed setup):
  TZ_TARGET  base model           (Qwen/Qwen3-4B-Instruct-2507)
  TZ_ADAPTER LoRA adapter repo    (wajih-rathore/transformer-zero), TZ_ADAPTER_DIR subfolder (qwen3-4b-lora); "" = none
  TZ_DRAFT   speculative draft    (Qwen/Qwen3-0.6B)
"""
import datetime as dt
import html
import os
import tempfile
import time

import gradio as gr
import pandas as pd
import torch

try:
    import spaces
    gpu = spaces.GPU
    ZEROGPU = True
except ImportError:  # not on a ZeroGPU Space: run functions as they are
    ZEROGPU = False
    def gpu(*a, **k):
        return a[0] if a and callable(a[0]) else (lambda f: f)

from tz.data import LABELS, build_messages, load_dataset, note_sentences, parse_output, prompt_text, split_free_text
from tz.decoder import Qwen3Decoder, greedy_generate
from tz.explain import explanation_messages, template_explanation
from tz.speculative import speculative_generate

TARGET = os.environ.get("TZ_TARGET", "Qwen/Qwen3-4B-Instruct-2507")
ADAPTER = os.environ.get("TZ_ADAPTER", "wajih-rathore/transformer-zero")
ADAPTER_DIR = os.environ.get("TZ_ADAPTER_DIR", "qwen3-4b-lora")
DRAFT = os.environ.get("TZ_DRAFT", "Qwen/Qwen3-0.6B")
MAX_CRITERIA, MAX_NOTE_TOKENS = 30, 3000
NOTICE = "Educational screening prototype only. Not a clinical enrollment decision."

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"  # where inference runs
dtype = torch.bfloat16 if dev == "cuda" else torch.float32
tok = AutoTokenizer.from_pretrained(TARGET)
# Load on CPU; apply LoRA on CPU (torch_device="cpu"). On ZeroGPU the real GPU attaches
# only inside @spaces.GPU and spaces patches cuda.is_available()->True at import, so loading
# weights to cuda now would crash. _ensure_device() moves everything onto the GPU on the
# first decorated call; off ZeroGPU we move immediately.
model = AutoModelForCausalLM.from_pretrained(TARGET, dtype=dtype).eval()
if ADAPTER:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, ADAPTER, subfolder=ADAPTER_DIR, torch_device="cpu").eval()
draft_hf = AutoModelForCausalLM.from_pretrained(DRAFT, dtype=dtype).eval()
if not ZEROGPU:
    model, draft_hf = model.to(dev), draft_hf.to(dev)
dec, draft = Qwen3Decoder(model), Qwen3Decoder(draft_hf)


def _ensure_device():
    """On ZeroGPU the GPU exists only inside @spaces.GPU; move models here on first use."""
    if not torch.cuda.is_available() or dec.device.type == "cuda":
        return
    model.to("cuda")
    draft_hf.to("cuda")
    dec.device = next(model.parameters()).device
    draft.device = next(draft_hf.parameters()).device
EOS = {tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
LORA = bool(ADAPTER)

# example patients and trials from the validation split (never the test split)
_df = load_dataset()
from tz.data import make_split, split_frame  # noqa: E402
_val = split_frame(_df, make_split(_df), "val")
EXAMPLES = {}
for (pid, trial), g in _val.groupby(["patient_id", "trial_id"]):
    EXAMPLES[f"{pid} / {trial}: {g.trial_title.iloc[0][:60]}"] = (
        g.note.iloc[0],
        "\n".join(g[g.criterion_type == "inclusion"].criterion_text),
        "\n".join(g[g.criterion_type == "exclusion"].criterion_text))
    if len(EXAMPLES) >= 8:
        break


def _ids(text):
    return tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)


@torch.no_grad()
def _hf_batch(prompts, max_new, lora):
    tok.padding_side = "left"
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
    kw = dict(max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id, temperature=None, top_p=None, top_k=None)
    if LORA and not lora:
        with model.disable_adapter():
            out = model.generate(**enc, **kw)
    else:
        out = model.generate(**enc, **kw)
    return tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)


def _ours(prompt, max_new, lora, speculative):
    x = _ids(prompt)
    if speculative:
        toks, _ = speculative_generate(dec, draft, x, max_new, k=4, eos_ids=EOS, target_lora=lora and LORA)
    else:
        toks, _ = greedy_generate(dec, x, max_new, eos_ids=EOS, lora=lora and LORA)
    return tok.decode(toks, skip_special_tokens=True)


def verdict(rows):
    """Explicit aggregation rule; the dataset has no patient-level ground truth."""
    labels = [(r["Type"], r["Decision"]) for r in rows]
    if any(l == ("inclusion", "not included") or l == ("exclusion", "excluded") for l in labels):
        return "Likely NOT eligible", "at least one inclusion criterion is not met or one exclusion criterion applies"
    if any(d in ("not enough information", "invalid output") for _, d in labels):
        return "Needs more information", "no criterion rules the patient out, but some cannot be decided from the note"
    return "Likely eligible", "every criterion is met or not applicable based on the note"


@gpu(duration=120)
def _screen_gpu(sentences, criteria, engine):
    _ensure_device()
    prompts = [prompt_text(tok, build_messages(sentences, t, c)) for t, c in criteria]
    t0 = time.perf_counter()
    if engine.startswith("Fast"):
        raw = _hf_batch(prompts, 64, lora=True)
    else:
        raw = [_ours(p, 64, True, engine.endswith("speculative")) for p in prompts]
    rows, need = [], []
    for (t, c), r in zip(criteria, raw):
        label, ev = parse_output(r, t, len(sentences))
        rows.append({"Type": t, "Criterion": c, "Decision": label or "invalid output", "Evidence IDs": ev,
                     "Evidence": " | ".join(f"[{i}] {sentences[i]}" for i in ev)})
        if label and ev:
            need.append(len(rows) - 1)
    exp_prompts = [prompt_text(tok, explanation_messages(rows[i]["Type"], rows[i]["Criterion"], rows[i]["Decision"],
                                                         [(j, sentences[j]) for j in rows[i]["Evidence IDs"]])) for i in need]
    if exp_prompts:
        exps = _hf_batch(exp_prompts, 96, lora=False) if engine.startswith("Fast") else \
            [_ours(p, 96, False, engine.endswith("speculative")) for p in exp_prompts]
    else:
        exps = []
    for i, e in zip(need, exps):
        rows[i]["Explanation"] = e.strip()
    for r in rows:
        r.setdefault("Explanation", template_explanation(r["Decision"]) if r["Decision"] != "invalid output"
                     else "The model's answer could not be read.")
    return rows, time.perf_counter() - t0


def screen(note, inc, exc, engine):
    note, inc, exc = (note or "").strip(), (inc or "").strip(), (exc or "").strip()
    if not note:
        raise gr.Error("Paste a patient note first.")
    sentences = split_free_text(note)
    if len(tok(" ".join(sentences)).input_ids) > MAX_NOTE_TOKENS:
        raise gr.Error(f"The note is too long (over {MAX_NOTE_TOKENS} tokens). Shorten it and try again.")
    criteria = [("inclusion", l.strip()) for l in inc.splitlines() if l.strip()] + \
               [("exclusion", l.strip()) for l in exc.splitlines() if l.strip()]
    if not criteria:
        raise gr.Error("Add at least one inclusion or exclusion criterion (one per line).")
    if len(criteria) > MAX_CRITERIA:
        raise gr.Error(f"Too many criteria ({len(criteria)}). The limit is {MAX_CRITERIA} per run.")
    rows, secs = _screen_gpu(sentences, criteria, engine)
    head, why = verdict(rows)
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences))
    table = pd.DataFrame([{k: (", ".join(map(str, v)) if k == "Evidence IDs" else v) for k, v in r.items()} for r in rows])
    summary = f"## {head}\nRule: {why}. ({len(rows)} criteria in {secs:.1f} s, engine: {engine})\n\n> {NOTICE}"
    return summary, table, numbered, _report(note, numbered, rows, head, why, engine, secs)


def _report(note, numbered, rows, head, why, engine, secs):
    cells = "".join("<tr>" + "".join(f"<td>{html.escape(str(r[k]))}</td>" for k in
                                     ("Type", "Criterion", "Decision", "Evidence IDs", "Evidence", "Explanation")) + "</tr>" for r in rows)
    doc = f"""<!doctype html><meta charset="utf-8"><title>Screening report</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem}}td,th{{border:1px solid #ccc;padding:6px;vertical-align:top}}table{{border-collapse:collapse}}.n{{background:#fff3cd;padding:8px}}</style>
<h1>Trial screening report</h1><p class="n"><b>{NOTICE}</b></p>
<h2>Overall: {html.escape(head)}</h2><p>Rule: {html.escape(why)}.</p>
<table><tr><th>Type</th><th>Criterion</th><th>Decision</th><th>Evidence IDs</th><th>Evidence</th><th>Explanation</th></tr>{cells}</table>
<h3>Patient note (numbered sentences)</h3><pre>{html.escape(numbered)}</pre>
<h3>Provenance</h3><ul><li>Generated: {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}</li>
<li>Model: {TARGET}; LoRA adapter: {ADAPTER or 'none'}/{ADAPTER_DIR if ADAPTER else ''}; draft: {DRAFT}</li>
<li>Engine: {html.escape(engine)}; greedy decoding; label max 64 tokens, explanation max 96 tokens; {secs:.1f} s</li>
<li>Explanations: written by the base model (LoRA off) from the cited sentences only.</li></ul>"""
    path = os.path.join(tempfile.mkdtemp(), "screening_report.html")
    open(path, "w", encoding="utf-8").write(doc)
    return path


@gpu(duration=120)
def lab(prompt, experiment, max_new, k):
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Type a prompt first.")
    _ensure_device()
    x = _ids(prompt_text(tok, [{"role": "user", "content": prompt}]))
    max_new, k = int(max_new), int(k)
    runs = []
    if experiment.startswith("KV"):
        for name, cache in (("with KV cache", True), ("no cache (recompute everything)", False)):
            toks, s = greedy_generate(dec, x, max_new, eos_ids=EOS, use_cache=cache, lora=False)
            runs.append((name, toks, s))
    elif experiment.startswith("Attention"):
        names = {"reference": "reference attention", "tiled": "tiled online-softmax", "sdpa": "PyTorch fused SDPA"}
        for name, label in names.items():
            dec.attention = name
            toks, s = greedy_generate(dec, x, max_new, eos_ids=EOS, lora=False)
            runs.append((label, toks, s))
        dec.attention = "reference"
    else:
        toks, s = greedy_generate(dec, x, max_new, eos_ids=EOS, lora=False)
        runs.append(("plain greedy (4B only)", toks, s))
        toks, s = speculative_generate(dec, draft, x, max_new, k=k, eos_ids=EOS)
        runs.append((f"speculative (0.6B drafts {k}, 4B checks)", toks, s))
    base = runs[0][1]
    table = pd.DataFrame([{
        "Method": n, "Same tokens as first row": t == base, "New tokens": s["new_tokens"],
        "Prefill (s)": round(s.get("prefill_s", float("nan")), 3), "Total (s)": round(s["total_s"], 3),
        "Tokens / s": round(s["new_tokens"] / s["total_s"], 1),
        "Acceptance rate": round(s["acceptance_rate"], 2) if "acceptance_rate" in s else None,
        "Peak GPU memory (MB)": round(s["peak_mem_mb"]) if s.get("peak_mem_mb") else None} for n, t, s in runs])
    return table, tok.decode(base, skip_special_tokens=True)


def load_example(name):
    return EXAMPLES[name] if name in EXAMPLES else ("", "", "")


with gr.Blocks(title="Transformer Zero") as demo:
    gr.Markdown(f"# Transformer Zero\nCheck a patient against a clinical trial's criteria, one criterion at a time, "
                f"with the patient sentences that back each decision.\n\n> **{NOTICE}**")
    with gr.Tab("Screen a patient"):
        with gr.Row():
            with gr.Column():
                ex = gr.Dropdown(list(EXAMPLES), label="Load an example (optional)")
                note = gr.Textbox(label="Patient note", lines=8, placeholder="Paste the patient's description here.")
                inc = gr.Textbox(label="Inclusion criteria (one per line)", lines=4)
                exc = gr.Textbox(label="Exclusion criteria (one per line)", lines=4)
                engine = gr.Radio(["Fast (batched)", "Our decoder (KV cache)", "Our decoder + speculative"],
                                  value="Fast (batched)", label="Engine")
                go = gr.Button("Screen", variant="primary")
            with gr.Column():
                summary = gr.Markdown()
                report = gr.File(label="Download report")
        table = gr.Dataframe(label="Result per criterion", wrap=True)
        with gr.Accordion("Patient note as numbered sentences", open=False):
            numbered = gr.Textbox(show_label=False, lines=8)
        ex.change(load_example, ex, [note, inc, exc])
        go.click(screen, [note, inc, exc, engine], [summary, table, numbered, report])
    with gr.Tab("Inference lab"):
        gr.Markdown("Run the same prompt through different decoding methods and compare speed. "
                    "All methods are our own code; matching tokens prove they compute the same answer.")
        prompt = gr.Textbox(label="Prompt", lines=3, value="Explain in three sentences what a clinical trial exclusion criterion is.")
        experiment = gr.Radio(["KV cache vs no cache", "Attention: reference vs tiled vs PyTorch", "Speculative vs plain greedy"],
                              value="KV cache vs no cache", label="Experiment")
        with gr.Row():
            max_new = gr.Slider(8, 256, value=64, step=8, label="Tokens to generate")
            k = gr.Slider(1, 8, value=4, step=1, label="Draft tokens per round (speculative)")
        run_btn = gr.Button("Run", variant="primary")
        lab_table = gr.Dataframe(label="Results")
        lab_text = gr.Textbox(label="Generated text", lines=4)
        run_btn.click(lab, [prompt, experiment, max_new, k], [lab_table, lab_text])

if __name__ == "__main__":
    demo.launch()
