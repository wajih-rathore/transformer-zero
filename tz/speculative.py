"""Greedy speculative decoding on top of the hand-written decoder.

Invariant at the top of every round: both caches hold the prompt plus every
committed token except the newest one (`last`), which neither model has seen yet.
"""
import time

import torch

from .decoder import Qwen3Decoder, _sync


@torch.no_grad()
def speculative_generate(target: Qwen3Decoder, draft: Qwen3Decoder, input_ids, max_new_tokens: int,
                         k: int = 4, eos_ids=(), target_lora: bool = False):
    """Returns the same tokens as greedy decoding with `target`, plus acceptance and timing stats."""
    assert input_ids.shape[0] == 1
    device = input_ids.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    t0 = time.perf_counter()

    max_len = input_ids.shape[1] + max_new_tokens + k + 1
    tc, dc = target.new_cache(max_len), draft.new_cache(max_len)
    last = target.forward(input_ids, tc, target_lora)[:, -1].argmax(-1, keepdim=True)
    draft.forward(input_ids, dc, lora=False)
    out = [last.item()]
    drafted = accepted = rounds = 0

    while len(out) < max_new_tokens and out[-1] not in eos_ids:
        base = tc.length

        # 1. draft guesses k tokens, one at a time (cheap model)
        tok, guesses = last, []
        for _ in range(k):
            tok = draft.forward(tok, dc, lora=False)[:, -1].argmax(-1, keepdim=True)
            guesses.append(tok)
        guess = torch.cat(guesses, dim=1)  # [1, k]

        # 2. target scores `last` + all k guesses in ONE forward pass
        preds = target.forward(torch.cat([last, guess], dim=1), tc, target_lora, all_logits=True)[0].argmax(-1)
        # preds[i] = target's own next token after seeing last, g1..gi

        # 3. accept the longest prefix where the draft matched the target, plus the target's next token
        n = int((guess[0] == preds[:k]).int().cumprod(0).sum())
        new = guess[0, :n].tolist() + [preds[n].item()]
        drafted, accepted, rounds = drafted + k, accepted + n, rounds + 1

        # 4. roll both caches back to: old state + last + accepted guesses
        tc.crop(base + n + 1)
        if n == k:  # the draft never ran on its own final guess; add it so both caches line up
            draft.forward(guess[:, -1:], dc, lora=False)
        dc.crop(base + n + 1)

        # 5. commit, stopping at EOS or the token budget
        for t in new:
            out.append(t)
            if t in eos_ids or len(out) >= max_new_tokens:
                break
        last = torch.tensor([[out[-1]]], device=device)

    _sync(device)
    total = time.perf_counter() - t0
    return out, {
        "prompt_tokens": input_ids.shape[1],
        "new_tokens": len(out),
        "k": k,
        "rounds": rounds,
        "acceptance_rate": accepted / drafted if drafted else 0.0,
        "mean_accepted_per_round": accepted / rounds if rounds else 0.0,
        "total_s": total,
        "tok_per_s": len(out) / total,
        "peak_mem_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
    }
