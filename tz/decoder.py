"""Hand-written Qwen3 forward pass with an explicit KV cache.

It reads the weights straight out of a loaded Hugging Face Qwen3 model (optionally
wrapped by PEFT), but every computation below is ours: RMSNorm, QK-Norm, RoPE at
explicit positions, grouped-query attention, SwiGLU MLP, and the cache reads/writes.
The same code runs Qwen3-0.6B / 1.7B / 4B, since only config sizes differ.
"""
import time

import torch
import torch.nn.functional as F

from .attention import causal_mask, reference_attention, tiled_attention


def _rms_norm(x, norm):
    """RMSNorm as Qwen3 does it: normalise in float32, scale by the weight in the input dtype."""
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
    return norm.weight * x.to(dtype)


def _linear(mod, x, lora: bool):
    """nn.Linear, or a PEFT LoRA layer: W x (+ B A x * scaling when lora is on)."""
    base = getattr(mod, "base_layer", mod)
    y = F.linear(x, base.weight, base.bias)
    if lora and hasattr(mod, "lora_A"):
        for name in mod.active_adapters:
            if name in mod.lora_A:
                a, b = mod.lora_A[name], mod.lora_B[name]
                y = y + (b(a(x.to(a.weight.dtype))) * mod.scaling[name]).to(y.dtype)
    return y


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class KVCache:
    """Preallocated K/V buffers for all layers plus a length pointer.

    Layout: [layers, batch, kv_heads, max_len, head_dim]. Positions 0..length-1 are valid.
    Rolling back (speculative decoding) only lowers `length`; stale entries are overwritten later.
    """

    def __init__(self, n_layers, n_kv_heads, head_dim, max_len, dtype, device, batch=1):
        shape = (n_layers, batch, n_kv_heads, max_len, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.length = 0
        self.max_len = max_len

    def crop(self, length: int):
        assert 0 <= length <= self.length
        self.length = length


class Qwen3Decoder:
    def __init__(self, hf_model, attention: str = "reference"):
        m = hf_model.get_base_model() if hasattr(hf_model, "get_base_model") else hf_model
        cfg = m.config
        assert not getattr(cfg, "use_sliding_window", False), "sliding-window attention not implemented"
        rope = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None) or {}
        assert rope.get("rope_type", "default") == "default", "only plain RoPE is implemented"
        theta = rope.get("rope_theta", getattr(cfg, "rope_theta", None))

        self.model, self.lm_head, self.config = m.model, m.lm_head, cfg
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.dtype = m.model.embed_tokens.weight.dtype
        self.device = m.model.embed_tokens.weight.device
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        self.attention = attention  # "reference" | "tiled" | "sdpa"

    def new_cache(self, max_len: int, batch: int = 1) -> KVCache:
        return KVCache(len(self.model.layers), self.n_kv, self.head_dim, max_len, self.dtype, self.device, batch)

    def _rope(self, positions):
        freqs = positions[:, None].float() * self.inv_freq.to(positions.device)[None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(self.dtype), emb.sin().to(self.dtype)

    def _attend(self, q, k, v):
        if self.attention == "tiled":
            return tiled_attention(q, k, v)
        if self.attention == "sdpa":  # PyTorch's fused kernel, with our bottom-right causal mask
            return F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask(q.shape[-2], k.shape[-2], q.device))
        return reference_attention(q, k, v)

    @torch.no_grad()
    def forward(self, input_ids, cache: KVCache | None = None, lora: bool = True, all_logits: bool = False):
        """Run new tokens through the model.

        With a cache, the new tokens get positions cache.length.. and their K/V are appended;
        attention then covers cached + new keys. Without a cache, positions start at 0.
        Returns float32 logits for the last position, or for every new position if all_logits.
        """
        B, T = input_ids.shape
        start = cache.length if cache is not None else 0
        assert cache is None or start + T <= cache.max_len, "KV cache full"
        cos, sin = self._rope(torch.arange(start, start + T, device=input_ids.device))
        rep = self.n_heads // self.n_kv  # grouped-query attention: one KV head serves `rep` query heads

        h = self.model.embed_tokens(input_ids)
        for i, layer in enumerate(self.model.layers):
            a = layer.self_attn
            x = _rms_norm(h, layer.input_layernorm)
            q = _linear(a.q_proj, x, lora).view(B, T, self.n_heads, self.head_dim)
            k = _linear(a.k_proj, x, lora).view(B, T, self.n_kv, self.head_dim)
            v = _linear(a.v_proj, x, lora).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
            q = _rms_norm(q, a.q_norm).transpose(1, 2)  # QK-Norm: per-head RMSNorm, before RoPE
            k = _rms_norm(k, a.k_norm).transpose(1, 2)
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin
            if cache is not None:
                cache.k[i, :, :, start:start + T] = k
                cache.v[i, :, :, start:start + T] = v
                k, v = cache.k[i, :, :, :start + T], cache.v[i, :, :, :start + T]
            o = self._attend(q, k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1))
            h = h + _linear(a.o_proj, o.transpose(1, 2).reshape(B, T, -1), lora)

            x = _rms_norm(h, layer.post_attention_layernorm)
            mlp = layer.mlp
            h = h + _linear(mlp.down_proj, F.silu(_linear(mlp.gate_proj, x, lora)) * _linear(mlp.up_proj, x, lora), lora)

        if cache is not None:
            cache.length = start + T
        h = _rms_norm(h if all_logits else h[:, -1:], self.model.norm)
        return _linear(self.lm_head, h, lora).float()


@torch.no_grad()
def greedy_generate(dec: Qwen3Decoder, input_ids, max_new_tokens: int, eos_ids=(), use_cache=True, lora=True):
    """Greedy decoding, batch size 1.

    use_cache=True : prefill the prompt once, then feed one token per step against the KV cache.
    use_cache=False: reference path, re-running the whole growing sequence every step.
    Returns (new token ids, timing/memory stats).
    """
    assert input_ids.shape[0] == 1
    device = input_ids.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    t0 = time.perf_counter()

    cache = dec.new_cache(input_ids.shape[1] + max_new_tokens) if use_cache else None
    nxt = dec.forward(input_ids, cache, lora)[:, -1].argmax(-1, keepdim=True)
    _sync(device)
    t1 = time.perf_counter()

    out, seq = [nxt.item()], input_ids
    while len(out) < max_new_tokens and out[-1] not in eos_ids:
        if use_cache:
            logits = dec.forward(nxt, cache, lora)
        else:
            seq = torch.cat([seq, nxt], dim=1)
            logits = dec.forward(seq, None, lora)
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        out.append(nxt.item())
    _sync(device)
    t2 = time.perf_counter()

    return out, {
        "prompt_tokens": input_ids.shape[1],
        "new_tokens": len(out),
        "prefill_s": t1 - t0,
        "decode_s": t2 - t1,
        "decode_tok_per_s": (len(out) - 1) / (t2 - t1) if len(out) > 1 else 0.0,
        "total_s": t2 - t0,
        "peak_mem_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
    }
