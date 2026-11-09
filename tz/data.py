"""Dataset loading, the fixed patient split, prompt formatting and output parsing."""
import json
import re

import numpy as np
import pandas as pd

DATASET_REPO = "ncbi/TrialGPT-Criterion-Annotations"
DATASET_FILE = "data/train-00000-of-00001.parquet"

# Only these columns are ever used. gpt4_*, explanation_correctness and the dataset's
# own `training` flag are dropped on load so they cannot leak into inputs or evaluation.
KEEP = ["annotation_id", "patient_id", "note", "trial_id", "trial_title",
        "criterion_type", "criterion_text", "expert_sentences", "expert_eligibility"]

LABELS = {
    "inclusion": ["included", "not included", "not enough information", "not applicable"],
    "exclusion": ["excluded", "not excluded", "not enough information", "not applicable"],
}
ALL_LABELS = ["included", "not included", "excluded", "not excluded", "not enough information", "not applicable"]

SYSTEM = ("You screen patients for clinical trials. Decide how one eligibility criterion applies to the "
          "patient, using only the numbered patient note. Answer with JSON only.")


def load_dataset(path: str | None = None) -> pd.DataFrame:
    if path is None:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(DATASET_REPO, DATASET_FILE, repo_type="dataset")
    df = pd.read_parquet(path)[KEEP].copy()
    df["expert_sentences"] = df["expert_sentences"].map(lambda s: sorted(set(json.loads(s))))
    df["criterion_text"] = df["criterion_text"].fillna("")  # one row has no criterion text; kept, not dropped
    return df


def make_split(df: pd.DataFrame) -> dict:
    """The assignment's split: sort unique patient IDs, shuffle with default_rng(42), 37/8/8."""
    pids = np.array(sorted(df["patient_id"].unique()))
    np.random.default_rng(42).shuffle(pids)
    split = {"train": pids[:37].tolist(), "val": pids[37:45].tolist(), "test": pids[45:].tolist()}
    assert not (set(split["train"]) & set(split["val"]) or set(split["train"]) & set(split["test"])
                or set(split["val"]) & set(split["test"])), "patient leak between splits"
    return split


def split_frame(df: pd.DataFrame, split: dict, name: str) -> pd.DataFrame:
    return df[df["patient_id"].isin(split[name])].reset_index(drop=True)


def note_sentences(note: str) -> list[str]:
    """Dataset notes are already numbered '0. text' per line; return the sentence texts in order."""
    out = []
    for line in note.splitlines():
        m = re.match(r"\s*(\d+)\.\s+(.*)", line)
        if m:
            assert int(m.group(1)) == len(out), "note sentences are not numbered 0..n-1"
            out.append(m.group(2).strip())
    return out


def split_free_text(text: str) -> list[str]:
    """Turn a pasted note into sentences (app input). Keeps already-numbered notes as they are."""
    numbered = note_sentences(text)
    if numbered:
        return numbered
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", " ".join(text.split()))
    return [p.strip() for p in parts if p.strip()]


def build_messages(sentences: list[str], criterion_type: str, criterion_text: str) -> list[dict]:
    """Note first, criterion last, so all criteria for one patient share the note as a prefix."""
    note = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences))
    labels = ", ".join(f'"{l}"' for l in LABELS[criterion_type])
    user = (f"Patient note:\n{note}\n\n"
            f"Criterion type: {criterion_type}\n"
            f"Criterion: {criterion_text.strip()}\n\n"
            f"Allowed labels: {labels}\n"
            'Reply with JSON: {"label": <one allowed label>, "evidence": [<ids of the note sentences that support the label>]}')
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def prompt_text(tokenizer, messages: list[dict]) -> str:
    # enable_thinking=False matters for the hybrid Qwen3-1.7B/0.6B; the 4B-Instruct-2507 template ignores it
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def target_text(label: str, evidence: list[int]) -> str:
    return json.dumps({"label": label, "evidence": evidence})


def row_messages(row) -> list[dict]:
    return build_messages(note_sentences(row["note"]), row["criterion_type"], row["criterion_text"])


def parse_output(text: str, criterion_type: str, n_sentences: int):
    """Returns (label or None if invalid, evidence ids). Invalid output counts as a wrong prediction."""
    m = re.search(r"\{.*\}", text, re.S)
    try:
        obj = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        obj = {}
    label = obj.get("label") if obj.get("label") in LABELS[criterion_type] else None
    evidence = []
    for i in obj.get("evidence", []) if isinstance(obj.get("evidence"), list) else []:
        if isinstance(i, int) and 0 <= i < n_sentences and i not in evidence:
            evidence.append(i)
    return label, sorted(evidence)
