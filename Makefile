# Transformer Zero — common tasks.  Run `make` (or `make help`) to list them.
# Activate your Python environment first, or run `make setup`.
# Override the interpreter with:  make test PYTHON=python3

PYTHON  ?= python
BASE_4B := Qwen/Qwen3-4B-Instruct-2507
ADAPTER := runs/qwen3-4b/best

.DEFAULT_GOAL := help
.PHONY: help setup split test figures app lab report app-local train eval bench

help:  ## List the available targets
	@grep -E '^[a-z][a-zA-Z_-]*:.*## ' $(MAKEFILE_LIST) | sort | \
		sed -E 's/:.*## /\t/' | awk -F'\t' '{printf "  %-9s %s\n", $$1, $$2}'

## --- quick, CPU-only (safe to run live in the demo) ---

setup:    ## Install dependencies into the active environment
	$(PYTHON) -m pip install -r requirements.txt

split:    ## Rebuild the 37/8/8 patient split manifest -> data/split_manifest.json
	$(PYTHON) scripts/make_split.py

test:     ## Run the 27 equivalence tests (KV cache / attention / speculative vs HF)
	$(PYTHON) -m pytest tests/ -q

figures:  ## Regenerate the report figures from results/ -> report/figures/
	$(PYTHON) scripts/make_figures.py

app:      ## Open the hosted screening app in your browser (runs on the remote ZeroGPU)
	$(PYTHON) -m webbrowser -t "https://web-wajihs-projects-109cd685.vercel.app"

lab:      ## Open the model Space: screening + the KV-cache/attention/speculative lab
	$(PYTHON) -m webbrowser -t "https://huggingface.co/spaces/wajih-rathore/transformer-zero-screen"

report:   ## Compile the IEEE report PDF (needs LaTeX) -> report/main.pdf
	cd report && latexmk -pdf -interaction=nonstopmode main.tex

## --- heavy, needs a GPU (minutes each) ---

app-local: ## Run the app on THIS machine instead (needs a local GPU) -> http://127.0.0.1:7860
	$(PYTHON) app.py

train:    ## Fine-tune Qwen3-4B with LoRA -> runs/qwen3-4b/
	$(PYTHON) scripts/train.py --model $(BASE_4B) --out runs/qwen3-4b

eval:     ## Evaluate base vs fine-tuned on val + test -> results/
	$(PYTHON) scripts/evaluate.py --model $(BASE_4B) --adapter $(ADAPTER) --name qwen3-4b-lora --splits val test

bench:    ## Benchmark KV cache / attention / speculative -> results/bench_qwen3-4b.json
	$(PYTHON) scripts/benchmark.py --target $(BASE_4B) --adapter $(ADAPTER) --out results/bench_qwen3-4b.json
