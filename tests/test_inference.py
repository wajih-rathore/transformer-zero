"""Equivalence tests for the hand-written attention, decoder, KV cache, LoRA flag and speculative decoding.

They run on CPU in float32 against tiny randomly initialised Qwen3 models, so no download is needed.
The same checks are repeated on the real models on GPU by scripts/benchmark.py.
"""
import pytest
import torch
import torch.nn.functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from tz.attention import causal_mask, reference_attention, tiled_attention
from tz.decoder import Qwen3Decoder, greedy_generate
from tz.speculative import speculative_generate


def tiny_model(seed):
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512,
                      tie_word_embeddings=True, attn_implementation="eager")
    return Qwen3ForCausalLM(cfg).eval()


def prompt(n, seed=0):
    return torch.randint(0, 256, (1, n), generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("tq,tk", [(1, 1), (7, 7), (100, 100), (1, 37), (5, 130), (64, 64)])
@pytest.mark.parametrize("causal", [True, False])
def test_tiled_matches_reference_and_sdpa(tq, tk, causal):
    torch.manual_seed(0)
    q, k, v = torch.randn(2, 3, tq, 16), torch.randn(2, 3, tk, 16), torch.randn(2, 3, tk, 16)
    ref = reference_attention(q, k, v, causal)
    tiled = tiled_attention(q, k, v, causal, block_q=16, block_k=24)  # blocks that do not divide the lengths
    mask = causal_mask(tq, tk) if causal else None
    sdpa = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    assert (tiled - ref).abs().max() < 1e-5
    assert (sdpa - ref).abs().max() < 1e-5


def test_decoder_logits_match_huggingface():
    hf = tiny_model(0)
    ids = prompt(20)
    ours = Qwen3Decoder(hf).forward(ids, all_logits=True)
    theirs = hf(ids).logits
    assert (ours - theirs).abs().max() < 1e-4


@pytest.mark.parametrize("n_prompt,n_new", [(5, 10), (30, 25), (80, 40)])
@pytest.mark.parametrize("attention", ["reference", "tiled", "sdpa"])
def test_cached_equals_uncached_equals_huggingface(n_prompt, n_new, attention):
    hf = tiny_model(0)
    dec = Qwen3Decoder(hf, attention=attention)
    ids = prompt(n_prompt)
    cached, _ = greedy_generate(dec, ids, n_new, use_cache=True)
    uncached, _ = greedy_generate(dec, ids, n_new, use_cache=False)
    theirs = hf.generate(ids, max_new_tokens=n_new, min_new_tokens=n_new, do_sample=False, pad_token_id=0)[0, n_prompt:]
    assert cached == uncached == theirs.tolist()


@pytest.mark.parametrize("k", [1, 3, 5])
def test_speculative_equals_target_greedy(k):
    target, draft = Qwen3Decoder(tiny_model(0)), Qwen3Decoder(tiny_model(1))
    ids = prompt(30)
    greedy, _ = greedy_generate(target, ids, 40, lora=False)
    spec, stats = speculative_generate(target, draft, ids, 40, k=k)
    assert spec == greedy
    assert 0.0 <= stats["acceptance_rate"] <= 1.0


def test_speculative_with_identical_draft_accepts_everything():
    target = Qwen3Decoder(tiny_model(0))
    spec, stats = speculative_generate(target, target, prompt(30), 40, k=4)
    assert stats["acceptance_rate"] == 1.0
    assert spec == greedy_generate(target, prompt(30), 40, lora=False)[0]


def test_lora_flag_matches_peft():
    peft = pytest.importorskip("peft")
    model = peft.get_peft_model(tiny_model(0), peft.LoraConfig(
        r=4, lora_alpha=8, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        init_lora_weights=False)).eval()  # random A and B so the adapter actually changes outputs
    ids = prompt(20)
    dec = Qwen3Decoder(model)
    on = dec.forward(ids, lora=True, all_logits=True)
    off = dec.forward(ids, lora=False, all_logits=True)
    assert (on - model(ids).logits).abs().max() < 1e-4
    with model.disable_adapter():
        assert (off - model(ids).logits).abs().max() < 1e-4
    assert (on - off).abs().max() > 1e-3
