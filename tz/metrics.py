"""Label and evidence metrics, computed the same way for every model and split."""
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from .data import ALL_LABELS

INVALID = "invalid output"


def label_metrics(gold: list[str], pred: list[str | None]) -> dict:
    """Accuracy, macro-F1 over the gold label set, per-class P/R/F1, confusion matrix.
    Unparseable predictions become their own 'invalid output' class, so they always count as wrong."""
    pred = [p if p is not None else INVALID for p in pred]
    labels = [l for l in ALL_LABELS if l in gold]
    p, r, f, n = precision_recall_fscore_support(gold, pred, labels=labels, zero_division=0)
    cm_labels = labels + ([INVALID] if INVALID in pred else [])
    return {
        "n": len(gold),
        "accuracy": accuracy_score(gold, pred),
        "macro_f1": f1_score(gold, pred, labels=labels, average="macro", zero_division=0),
        "invalid_rate": pred.count(INVALID) / len(pred),
        "per_class": {l: {"precision": p[i], "recall": r[i], "f1": f[i], "support": int(n[i])} for i, l in enumerate(labels)},
        "confusion": {"labels": cm_labels, "matrix": confusion_matrix(gold, pred, labels=cm_labels).tolist()},
    }


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def evidence_metrics(gold: list[list[int]], pred: list[list[int]]) -> dict:
    """Sentence-level evidence scores.
    micro: pool TP/FP/FN over all records.
    macro: average per-record P/R/F1; a record with empty gold AND empty prediction scores 1,
           one empty and the other not scores 0."""
    tp = fp = fn = 0
    per = []
    for g, p in zip(gold, pred):
        g, p = set(g), set(p)
        tp, fp, fn = tp + len(g & p), fp + len(p - g), fn + len(g - p)
        per.append((1.0, 1.0, 1.0) if not g and not p else _prf(len(g & p), len(p - g), len(g - p)))
    mp, mr, mf = _prf(tp, fp, fn)
    k = len(per)
    return {
        "micro": {"precision": mp, "recall": mr, "f1": mf},
        "macro": {"precision": sum(x[0] for x in per) / k, "recall": sum(x[1] for x in per) / k,
                  "f1": sum(x[2] for x in per) / k},
        "empty_gold": sum(1 for g in gold if not g),
        "empty_pred": sum(1 for p in pred if not p),
    }


def all_metrics(df) -> dict:
    """df needs: criterion_type, expert_eligibility, pred_label, expert_sentences, pred_evidence."""
    out = {"overall": {"labels": label_metrics(df.expert_eligibility.tolist(), df.pred_label.tolist()),
                       "evidence": evidence_metrics(df.expert_sentences.tolist(), df.pred_evidence.tolist())}}
    for t in ("inclusion", "exclusion"):
        d = df[df.criterion_type == t]
        out[t] = {"labels": label_metrics(d.expert_eligibility.tolist(), d.pred_label.tolist()),
                  "evidence": evidence_metrics(d.expert_sentences.tolist(), d.pred_evidence.tolist())}
    return out
