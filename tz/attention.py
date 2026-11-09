"""Reference and tiled (online-softmax) scaled dot-product attention.

Shapes: q [B, H, Tq, D], k and v [B, H, Tk, D], with Tq <= Tk.

Causal alignment: the Tq queries are the *last* Tq positions of the Tk-long
sequence, so query i sits at absolute position (Tk - Tq + i) and may attend to
keys 0 .. Tk - Tq + i. That one rule is correct both for prefill (Tq == Tk)
and for decoding against a KV cache (Tq = new tokens, Tk = cached + new).
Note: torch's SDPA `is_causal=True` aligns the mask top-left instead, which is
wrong when Tq < Tk, so comparisons against SDPA pass an explicit mask.
"""
import math

import torch


def causal_mask(tq: int, tk: int, device=None) -> torch.Tensor:
    """Bool [tq, tk] mask, True where query may attend to key (bottom-right aligned)."""
    q_pos = torch.arange(tk - tq, tk, device=device)[:, None]
    k_pos = torch.arange(tk, device=device)[None, :]
    return k_pos <= q_pos


def reference_attention(q, k, v, causal: bool = True) -> torch.Tensor:
    """softmax(q k^T / sqrt(d) + mask) v, materialising the full [Tq, Tk] score matrix.

    Computed in float32 for a numerically stable baseline, returned in q's dtype.
    """
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = (q.float() @ k.float().transpose(-2, -1)) * scale
    if causal:
        scores = scores.masked_fill(~causal_mask(q.shape[-2], k.shape[-2], q.device), float("-inf"))
    scores = scores - scores.amax(dim=-1, keepdim=True)  # subtract row max: exp never overflows
    p = scores.exp()
    p = p / p.sum(dim=-1, keepdim=True)
    return (p @ v.float()).to(q.dtype)


def tiled_attention(q, k, v, causal: bool = True, block_q: int = 64, block_k: int = 64) -> torch.Tensor:
    """Same result as reference_attention, computed block by block with an online softmax.

    For each query block we stream over key blocks and keep three running values per query row:
      m   - the largest score seen so far
      l   - the sum of exp(score - m) seen so far
      acc - the sum of exp(score - m) * v seen so far
    When a new block raises the max from m to m_new, the old l and acc are rescaled by
    exp(m - m_new), so the result is exact. Only a [block_q, block_k] score tile is ever
    held, instead of the full [Tq, Tk] matrix.
    """
    B, H, Tq, D = q.shape
    Tk = k.shape[-2]
    scale = 1.0 / math.sqrt(D)
    out = torch.empty(B, H, Tq, D, dtype=q.dtype, device=q.device)
    offset = Tk - Tq  # absolute position of query 0

    for qs in range(0, Tq, block_q):
        qe = min(qs + block_q, Tq)
        qb = q[:, :, qs:qe].float() * scale
        q_pos = torch.arange(qs + offset, qe + offset, device=q.device)[:, None]
        m = torch.full((B, H, qe - qs, 1), float("-inf"), device=q.device)
        l = torch.zeros((B, H, qe - qs, 1), device=q.device)
        acc = torch.zeros((B, H, qe - qs, D), device=q.device)

        # keys after the last query in this block are masked for every row: skip them
        k_end = (qe + offset) if causal else Tk
        for ks in range(0, k_end, block_k):
            ke = min(ks + block_k, k_end)
            s = qb @ k[:, :, ks:ke].float().transpose(-2, -1)
            if causal:
                allowed = torch.arange(ks, ke, device=q.device)[None, :] <= q_pos
                s = s.masked_fill(~allowed, float("-inf"))
            m_new = torch.maximum(m, s.amax(dim=-1, keepdim=True))
            # a row with no visible key yet has m_new = -inf; use 0 so exp() gives 0, not nan
            m_safe = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)
            alpha = torch.exp(m - m_safe)
            p = torch.exp(s - m_safe)
            l = l * alpha + p.sum(dim=-1, keepdim=True)
            acc = acc * alpha + p @ v[:, :, ks:ke].float()
            m = m_new

        out[:, :, qs:qe] = (acc / l).to(q.dtype)
    return out
