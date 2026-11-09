"""Step 2 of a prediction: a short explanation written from the cited sentences only.

The fine-tuned model (LoRA on) picks the label and sentence IDs; the same model with
LoRA off writes the explanation, seeing nothing but the criterion, the label and the
cited sentences. No cited sentences means nothing to explain from, so a template is used.
"""
SYSTEM = "You explain clinical trial screening decisions in one or two plain sentences, using only the evidence given."


def explanation_messages(criterion_type: str, criterion_text: str, label: str, cited: list[tuple[int, str]]):
    evidence = "\n".join(f"{i}. {s}" for i, s in cited)
    user = (f"Criterion ({criterion_type}): {criterion_text.strip()}\n"
            f"Decision: {label}\n"
            f"Patient sentences cited as evidence:\n{evidence}\n\n"
            "Explain in one or two sentences why these sentences lead to the decision. "
            "Refer to sentences by their number and do not add facts that are not in them.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def template_explanation(label: str) -> str:
    return f"No sentence in the patient note addresses this criterion, so it is marked '{label}'."
