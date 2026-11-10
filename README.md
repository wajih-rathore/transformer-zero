# Transformer Zero

A fine-tuned, open-weight decoder that reads a patient note against one clinical-trial
eligibility criterion, returns an eligibility label with the **patient sentences that justify it**,
and ships with from-scratch **KV-cached decoding**, **tiled online-softmax attention**, and
**greedy speculative decoding**.

> **Educational screening prototype only — not a clinical enrollment decision tool.**

**[Live app ↗](https://web-wajihs-projects-109cd685.vercel.app)** ·
**[Model Space ↗](https://huggingface.co/spaces/wajih-rathore/transformer-zero-screen)** ·
**[Adapter weights ↗](https://huggingface.co/wajih-rathore/transformer-zero)** ·
**[Dataset ↗](https://huggingface.co/datasets/ncbi/TrialGPT-Criterion-Annotations)** ·
[Report (PDF)](report/main.pdf)

---

## What it does

Base model **Qwen3-4B-Instruct-2507** (Apache-2.0), fine-tuned with LoRA on the
TrialGPT Criterion Annotations dataset. The hand-written Qwen3 forward pass in `tz/decoder.py`
loads the Hugging Face weights directly and powers Tasks 3–5, verified token-for-token against
the reference implementation.

### Results — held-out test set (206 criteria, greedy decoding)

| Model | Accuracy (base → tuned) | Macro-F1 | Evidence micro-F1 |
|---|---|---|---|
| **Qwen3-4B** | 0.495 → **0.762** | 0.554 → **0.675** | 0.444 → **0.617** |
| Qwen3-1.7B | 0.218 → 0.714 | 0.259 → 0.452 | 0.262 → 0.542 |

For context, the dataset's own GPT-4 labels agree with the expert consensus at only 0.903 —
and that is inflated, since the gold labels were made by correcting GPT-4. No fine-tuned run
emits malformed JSON.

---

## Quickstart

```bash
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate
pip install -r requirements.txt     # or: make setup

python app.py                       # or: make app  →  http://127.0.0.1:7860
```

The app needs a GPU for the 4B model. Without one, the hosted
[Space](https://huggingface.co/spaces/wajih-rathore/transformer-zero-screen) and
[Vercel app](https://web-wajihs-projects-109cd685.vercel.app) run it for you.

## Common tasks (`make`)

| Target | What it does | Needs |
|---|---|---|
| `make app` | Open the hosted screening app (model runs on the remote ZeroGPU) | no GPU |
| `make lab` | Open the model Space (screening + the decoding-benchmark lab) | no GPU |
| `make test` | 27 equivalence tests: cached = uncached = HF, tiled = reference, speculative = greedy | CPU, ~1.5 min |
| `make split` | Rebuild the 37/8/8 patient split manifest | CPU |
| `make figures` | Regenerate the report figures from `results/` | CPU |
| `make report` | Compile the IEEE PDF | LaTeX |
| `make app-local` | Run the app on this machine instead | local GPU |
| `make train` | Fine-tune Qwen3-4B with LoRA | GPU, ~8 min |
| `make eval` | Evaluate base vs fine-tuned on val + test | GPU |
| `make bench` | Benchmark KV cache / attention / speculative | GPU |

`make` with no target lists them. No `make`? Each target is a single command — copy it from the
[`Makefile`](Makefile) and run it directly.

---

## Where each task lives

| Task | Implementation |
|---|---|
| 1–2 · Fine-tuning, eligibility + evidence | `tz/data.py`, `scripts/train.py`, `scripts/evaluate.py`, `tz/metrics.py`, `tz/explain.py` |
| 3 · KV-cached decoding | `tz/decoder.py` — `Qwen3Decoder`, `KVCache`, `greedy_generate` |
| 4 · Tiled online-softmax attention | `tz/attention.py` — `reference_attention`, `tiled_attention` |
| 5 · Greedy speculative decoding | `tz/speculative.py` — `speculative_generate` |
| 6 · Browser app | `app.py` (Gradio) · `web/index.html` (custom Vercel UI) |
| Correctness suite | `tests/test_inference.py` |
| Report | `report/` (LaTeX source + figures + PDF) |

## Repository layout

```
tz/            model + decoding library (importable package)
scripts/       make_split, train, evaluate, benchmark, make_figures
tests/         CPU equivalence tests (pytest)
data/          split_manifest.json  — the fixed 37/8/8 patient split (seed 42)
results/       metrics, predictions, training logs, bench_qwen3-4b.json
report/        IEEE LaTeX paper (main.tex, refs.bib, figures/, main.pdf)
web/           static Vercel front-end that calls the hosted model
app.py         Gradio application (Screen tab + Inference lab tab)
```

## Compute

Trained and benchmarked on one NVIDIA RTX PRO 4000 Blackwell (24 GB, bf16), PyTorch 2.11 /
CUDA 12.8. The 4B model fits in 24 GB with LoRA, gradient checkpointing, and batch 1 /
accumulation 16. Adapters and metrics are published to the
[model repo](https://huggingface.co/wajih-rathore/transformer-zero).

## AI assistance

Built with an AI coding assistant (Claude) for scaffolding, debugging, orchestration, figures,
and report drafting. All design decisions, the experimental setup, and the final claims were
directed and verified by the author; every reported number was produced by the committed code
on the stated hardware. See the report's AI-use appendix.
