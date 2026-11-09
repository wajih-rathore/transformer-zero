# Transformer Zero: Design

Date: 2026-11-09. Deadline: 2026-11-10 23:59 local time (Google Classroom, 2-hour grace, then zero).
Source of requirements: `GenAI_Assignment#2.pdf` (Generative AI, Assignment 2). Rubric codes (A1–H3) below refer to its rubric table.

Items marked **(agreed)** were decided with the owner. Items marked **(my call)** are defaults the designer chose; the owner has not confirmed them.

## 1. Intent

A fine-tuned open-weight decoder that reads a patient note and one clinical-trial criterion and returns an eligibility label, the patient sentences that support it, and an explanation. Around it sit three hand-written inference mechanisms (KV-cached decoding, tiled online-softmax attention, greedy speculative decoding) and a browser app that exposes both the screening and the inference comparisons.

The point is to show understanding of the Transformer, not just to call libraries: the owner must be able to explain, run, and modify any component in a live defense.

## 2. Deliverables (from the assignment)

- GitHub repo: code, environment spec, data prep, training/eval scripts, attention and decoding code, app source, reproduction steps, results.
- Trained LoRA adapters hosted externally, with a working download link.
- Split manifest (patient IDs and counts) and benchmark logs.
- IEEE-format LaTeX paper (PDF).
- Browser app usable without running code.
- 5–7 minute recorded app demo (video link).
- Brief AI-use appendix.
- Live demo and defense (no marks if missed).

## 3. Goals, non-goals, anti-goals

**Goals**
- Every rubric row A1–H3 has visible evidence in the app, code, outputs, or paper.
- One hand-written Qwen3 forward pass powers Tasks 3, 4, and 5, verified against Hugging Face's implementation.
- Honest results, including slowdowns and failures.

**Non-goals**
- Clinical accuracy beyond what the dataset supports; no claims of clinical use.
- Custom CUDA/Triton kernels.
- Fetching live trial data from ClinicalTrials.gov (allowed by the brief, skipped for time) **(my call)**.

**Anti-goals (failure even if it "works")**
- Any `gpt4_*` column reaching model input, training targets, or evaluation (B2).
- Any patient appearing in more than one split, or any tuning on the test split (B1).
- Using `use_cache=True` or an assisted-generation API as the implementation of Tasks 3/5 (allowed only as a reference check).
- Collapsing the label set to yes/no.

## 4. Constraints

- Free compute only. **(agreed)** No Kaggle: the owner will not give Kaggle a phone number, and Kaggle requires phone verification for every GPU and for notebook internet (Kaggle staff, https://www.kaggle.com/discussions/product-feedback/751699). Training, evaluation, and benchmarks run on **free Colab, one T4** (15 GB, no bf16, so fp16). Sessions drop when idle and GPU availability is not guaranteed, so jobs checkpoint to Google Drive or the Hugging Face Hub and can resume.
- Owner's PC: E: (project drive) has ~2.4 GB free, C: ~18 GB. No model weights on the PC. Any local Python env with torch goes on C:, not E:. MiKTeX (pdflatex, latexmk, bibtex) is installed for the report.
- App hosting: Hugging Face ZeroGPU Space (owner's account is eligible: created 2024). Per https://huggingface.co/docs/hub/spaces-zerogpu (read 2026-11-09): Gradio only; torch 2.8.0–2.13.0; Python 3.10.13 or 3.12.12; default GPU is half an RTX Pro 6000 Blackwell (48 GB); GPU work only inside `@spaces.GPU` functions, 60 s default per call (raisable with `duration=`); models placed on `cuda` at module level; no `torch.compile`. Daily GPU quota is charged to the **visitor**: 2 min unauthenticated, 5 min for a free logged-in account. The owner's own use spends the owner's 5 minutes.
- Fixed split: sort unique `patient_id`s, shuffle with `np.random.default_rng(42)`, first 37 train / next 8 val / last 8 test. Ignore the dataset's own `training` column.

## 5. Decisions

### 5.1 Data **(agreed)**

- Dataset: `ncbi/TrialGPT-Criterion-Annotations`, 1,015 rows, 53 patients, 103 trials.
- Inputs used: `note` (sentences already numbered `0. …`, `1. …`), `criterion_type`, `criterion_text`. Optionally `trial_title`.
- Targets: `expert_eligibility` (six labels, kept as-is) and `expert_sentences` (sentence IDs).
- Label sets per type: inclusion → `included`, `not included`, `not enough information`, `not applicable`; exclusion → `excluded`, `not excluded`, `not enough information`, `not applicable`. Totals: not excluded 484, not enough information 293, included 150, not applicable 55, not included 23, excluded 10. Rare classes will drag macro-F1; report it, do not hide it.
- Split manifest saved to `data/split_manifest.json` (IDs and row counts per split) with an automated no-leak check.
- Truncation: measure token lengths first. Record how many rows get truncated and whether any gold evidence sentence is lost (the brief requires this). Prefer a max length that truncates nothing if memory allows.

### 5.2 Models **(agreed)**

| Role | Model | Notes |
|---|---|---|
| Main target | `Qwen/Qwen3-4B-Instruct-2507` | 36 layers, 32 Q / 8 KV heads, head_dim 128, Apache 2.0, non-thinking |
| Second candidate | `Qwen/Qwen3-1.7B` | 28 layers, 16/8 heads; fine-tuned the same way for the comparison |
| Draft | `Qwen/Qwen3-0.6B` | not fine-tuned; identical `tokenizer.json` to both targets (sha256 verified) |

Rejected, for the paper's alternatives section (A3): Qwen3.5 (linear-attention layers with recurrent state, so a KV cache exists in only 1 of 4 layers and speculative rollback is messy), Gemma 4 and MedGemma (sliding-window/global mix, bundled vision tower, MedGemma's HAI-DEF license), Llama 3.2 3B/1B (gated access, older).

Qwen3-1.7B and 0.6B are hybrid thinking models: always format with thinking disabled.

### 5.3 Fine-tuning

- **(agreed)** LoRA on both targets. With one Colab T4 they train one after the other, 4B first.
- **(agreed)** Model output is compact JSON: `{"label": "<one of the labels>", "evidence": [<sentence ids>]}`.
- **(agreed)** Loss only on the answer tokens, not the prompt.
- **(my call)** Prompt order: note first, then criterion type and text, so the note's tokens form a shared prefix across criteria.
- **(my call)** Starting hyperparameters: rank 16, alpha 32, dropout 0.05, all attention and MLP projections, lr 2e-4, ~3 epochs, effective batch ~16, fp16 with gradient checkpointing. Choose the checkpoint by validation macro-F1.
- First run is a ~50-step timing and stability check. If fp16 is unstable or the 4B does not fit one T4, fall back to QLoRA (4-bit base).
- Output: adapters pushed from Colab straight to Hugging Face Hub; training curves and logs committed to the repo.

### 5.4 Prediction pipeline (Task 2) **(agreed)**

Per criterion:
1. Fine-tuned model (LoRA on) generates the JSON label and sentence IDs.
2. Evidence text is looked up from the IDs by code.
3. Explanation: the same model with LoRA **off** gets only the criterion, the predicted label, and the cited sentences, and writes 1–2 sentences justifying that label. If no sentence is cited, a fixed template is used instead ("No sentence in the note addresses this criterion.").
- **(my call)** Unparseable JSON counts as a wrong prediction and the invalid-output rate is reported.

### 5.5 Evaluation **(agreed)**

- Base vs fine-tuned, same val/test examples, greedy decoding, settings recorded (B4).
- Labels: accuracy, macro-F1, per-class P/R/F1, confusion matrices; overall and split by inclusion/exclusion (C1, C2).
- Evidence: micro and macro sentence P/R/F1 (C3). **(my call)** For macro, a record with empty gold and empty prediction scores 1; one empty and the other not scores 0. Also report the non-empty-gold subset.
- At least eight analysed errors, including unsupported and missing-information cases (C4).
- Test split run once, at the end.
- **(my call)** Metric runs use Hugging Face `generate` with the PEFT adapter, batched, greedy. The hand-written decoder is proven equivalent on greedy tokens separately (5.6), so metrics do not depend on which one ran them.
- **(my call)** If training falls back to QLoRA, inference loads the base in fp16/bf16 and applies the adapter on top; the small train/inference mismatch is documented.

### 5.6 Hand-written decoder (Tasks 3–5) **(agreed)**

One plain-PyTorch Qwen3 forward pass that loads the real Hugging Face weights: embeddings, RMSNorm, Q/K/V projections, per-head QK-Norm, RoPE at explicit positions, grouped-query attention, SwiGLU MLP, tied LM head. All dimensions come from the model's `config.json`. The same code runs the 4B, 1.7B, and 0.6B.

- **Correctness anchor:** its logits match Hugging Face's model on the same input (tolerance documented).
- **LoRA flag:** applies the LoRA deltas when on and skips them when off (used by step 3 of 5.4).
- **Task 3, KV cache:** preallocated per-layer K/V tensors plus a current-length pointer. Prefill writes the prompt; each decode step writes one position with the right RoPE offset and attends over the cache. The reference path recomputes the full sequence each step. Tests: identical greedy tokens across at least 3 input/output length pairs; mismatches investigated (fp16 can cause them). Benchmarks: prefill time, decode time, tokens/s, peak memory, with warm-up, `cuda.synchronize`, and repeats (D1–D3).
- **Task 4, attention:** a hand-written reference (scaled scores, causal mask, stable softmax) and a tiled version that loops over key blocks keeping a running max and running sum. Selectable as the decoder's attention. Tested on several shapes with causal masks, max and mean error reported, plus speed and memory against PyTorch SDPA, with an explanation of why the Python version is slower (E1–E3).
- **Task 5, speculative:** the draft proposes k tokens (default 4, swept 2–8), the target verifies them in one forward pass, the longest matching prefix is kept plus one corrected token, and both caches roll back by moving their pointers. Must produce exactly the target's greedy tokens. Reports acceptance rate, latency, tokens/s, and speedup or slowdown (F1–F3). **(my call)** Benchmarked mainly on the explanation step (longer output, LoRA off, closer to the plain draft); also reported on the short JSON step.

### 5.7 App (Task 6)

**(agreed)** The owner demos the app to the examiner in person. It must be simple enough that anyone understands it at first look: plain words, one obvious main action, technical detail tucked behind expanders.

**(my call)** Gradio on a ZeroGPU Space, with two tabs. A separate frontend (e.g. Vercel) is a non-goal: the GPU code still has to run in a Space, so it adds a second deployment and the same quota.

- **Screen:** paste a patient note (auto-split into numbered sentences and shown), then choose a trial from the dataset's trials or paste inclusion and exclusion criteria (one per line). Run produces one row per criterion: type, criterion, label, cited IDs, evidence text, explanation. An overall preliminary outcome from an explicit rule, labelled as a rule and not ground truth:
  - any inclusion `not included` or exclusion `excluded` → likely ineligible;
  - else any `not enough information` → needs more information;
  - else → likely eligible.

  Because of the 5-minute daily quota, screening defaults to batched Hugging Face generation (all criteria in one batch, then all explanations in one batch), with a dropdown to rerun through the hand-written decoder (KV cache, optional speculative). Criteria per run are capped (about 30), and output tokens are capped.

  Fixed notice: educational screening prototype, not a clinical enrollment decision. Export a downloadable report (HTML or Markdown) with provenance: model IDs, adapter version, decoding settings, timestamp. Input validation with clear messages for an empty note, no criteria, or an over-length note (G1, G2, G4).
- **Inference lab:** run live, on a chosen input: cached vs uncached, reference vs tiled attention, speculative vs plain greedy. Shows timings, tokens/s, memory, acceptance rate, and whether tokens matched (G3: live comparison, not static charts).

Demo-day backup: the same app run in a Colab notebook with Gradio's temporary public share link (no quota; lives only while the notebook runs).

## 6. Left to the builder

- Repo layout, module names, test framework, and how the Colab notebooks are packaged.
- Exact prompt wording and system message.
- Whether to reuse the note's KV prefix across criteria in the app (a speed win; optional).
- Batch sizes, max length after measuring, epoch count after the first run.
- Where tests run: tiny random-weight Qwen3 configs on CPU are fine for decoder, cache, attention, and speculative equivalence tests; real-model checks run on Colab.
- Report figure choices and the ≥5 literature sources (likely candidates: TrialGPT, LoRA, QLoRA, FlashAttention, online-softmax, speculative decoding, Qwen3 technical report).

## 7. Order of work

1. Data prep, split manifest, prompt format, token-length check.
2. Training script; start the Colab run (4B, then 1.7B) as early as possible (longest pole).
3. While training runs: hand-written decoder, KV cache, tiled attention, speculative decoding, each with equivalence tests; base-model (zero-shot) evaluation on val.
4. Fine-tuned evaluation on val, then the single test run.
5. Benchmarks on the Colab T4; logs committed.
6. App on ZeroGPU, then the demo video.
7. Paper, AI-use appendix, submission with buffer before the deadline.

**(my call)** If time runs out, cut in this order: the 1.7B fine-tune (keep its base-model numbers), the k sweep (keep k=4), app polish. Never cut: split integrity, Tasks 3–5 equivalence tests, the app's core screening flow, the paper.

## 8. Risks

- fp16 on T4 with Qwen3 may overflow or diverge; caught by the 50-step run.
- Colab free tier can disconnect or deny a GPU; training must checkpoint and resume.
- ZeroGPU quotas (2 min anonymous, 5 min logged in, per day) may run out during grading or the defense. The Colab share-link backup has no quota but only lives while the notebook runs.
- The owner still needs a Hugging Face write token (for adapter upload and the Space).
- Unknown note lengths may force truncation.
- Rare labels (10 `excluded`, 23 `not included`) make macro-F1 noisy on an 8-patient test set.
