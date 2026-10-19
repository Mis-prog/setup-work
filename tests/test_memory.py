from __future__ import annotations

from setup_work.config import Attention
from setup_work.memory import (
    GiB,
    kv_cache_bytes,
    kv_heads_per_shard,
    max_context_for_budget,
    weight_bytes_per_shard,
)

QWEN3_8B = Attention(num_layers=36, num_attention_heads=32, num_kv_heads=8, head_dim=128)
GEMMA3_12B = Attention(
    num_layers=48,
    num_attention_heads=16,
    num_kv_heads=8,
    head_dim=256,
    sliding_window=1024,
    sliding_window_layers=40,
)


def test_kv_cache_dense_gqa():
    # 2 (K,V) × 36 слоёв × 8 голов × 128 × 2 байта = 147456 байт/токен
    assert kv_cache_bytes(QWEN3_8B, 1) == 147456
    assert kv_cache_bytes(QWEN3_8B, 32768) == 147456 * 32768  # 4.5 GiB
    assert abs(kv_cache_bytes(QWEN3_8B, 32768) / GiB - 4.5) < 1e-9


def test_kv_cache_scales_with_concurrency_and_dtype():
    one = kv_cache_bytes(QWEN3_8B, 8192)
    assert kv_cache_bytes(QWEN3_8B, 8192, concurrency=4) == 4 * one
    assert kv_cache_bytes(QWEN3_8B, 8192, cache_dtype="fp8") == one // 2


def test_sliding_window_layers_saturate():
    short = kv_cache_bytes(GEMMA3_12B, 1024)
    long = kv_cache_bytes(GEMMA3_12B, 32768)
    per_tok_layer = 2 * 8 * 256 * 2
    assert long == per_tok_layer * (8 * 32768 + 40 * 1024)
    # Без скользящего окна при 32k кэш был бы в 6 раз больше
    full = Attention(num_layers=48, num_attention_heads=16, num_kv_heads=8, head_dim=256)
    assert kv_cache_bytes(full, 32768) / long > 4
    assert short == per_tok_layer * 48 * 1024


def test_tensor_parallel_kv_sharding_and_replication():
    assert kv_heads_per_shard(8, 2) == 4
    assert kv_heads_per_shard(4, 8) == 1  # репликация: у каждой доли остаётся одна голова
    assert kv_cache_bytes(QWEN3_8B, 1000, tp=2) == kv_cache_bytes(QWEN3_8B, 1000) // 2
    a = Attention(num_layers=1, num_attention_heads=8, num_kv_heads=2, head_dim=64)
    assert kv_cache_bytes(a, 100, tp=8) == kv_cache_bytes(a, 100, tp=2)


def test_weight_sharding_has_replication_overhead():
    w = 10 * GiB
    assert weight_bytes_per_shard(w, 1) == w
    assert weight_bytes_per_shard(w, 2) > w / 2


def test_max_context_inverts_kv_cache():
    budget = 2 * GiB
    ctx = max_context_for_budget(QWEN3_8B, budget, 1, "fp16")
    assert kv_cache_bytes(QWEN3_8B, ctx) <= budget < kv_cache_bytes(QWEN3_8B, ctx + 1)
    assert max_context_for_budget(QWEN3_8B, 0, 1, "fp16") == 0
