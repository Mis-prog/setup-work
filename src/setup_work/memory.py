"""Оценка памяти в байтах.

Учитываются веса с распределением по GPU, KV-кэш и накладные расходы.
KV-кэш: 2 × слои × KV-головы × размер головы × байты элемента × токены.
Для скользящего окна число токенов ограничивается размером окна.
При TP больше числа KV-голов головы реплицируются."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import Attention

GiB = 1024**3

KV_BYTES_PER_ELEMENT = {"fp32": 4.0, "fp16": 2.0, "bf16": 2.0, "fp8": 1.0, "q8_0": 34 / 32}

# Оценка сохранения качества для ранжирования форматов, не результат бенчмарка.
QUANT_QUALITY = {
    "none": 1.00,
    "f16": 1.00,
    "mxfp4": 1.00,  # gpt-oss изначально выпущен в MXFP4
    "fp8": 0.99,
    "q8_0": 0.99,
    "q6_k": 0.97,
    "q5_k_m": 0.95,
    "awq": 0.94,
    "q4_k_m": 0.92,
}

# Репликация эмбеддингов и нормализаций при распределении весов по GPU.
TP_REPLICATION_OVERHEAD = 0.02


def gib(n: float | int | None) -> str:
    return "n/a" if n is None else f"{n / GiB:.1f} GiB"


def kv_heads_per_shard(num_kv_heads: int, tp: int) -> int:
    """vLLM распределяет KV-головы по TP и реплицирует их, если TP больше числа голов."""
    return max(1, math.ceil(num_kv_heads / tp))


def kv_cache_bytes(
    att: Attention,
    context: int,
    concurrency: int = 1,
    cache_dtype: str = "fp16",
    tp: int = 1,
) -> int:
    """KV-кэш для concurrency последовательностей по context токенов на одну долю TP."""
    if context <= 0 or concurrency <= 0:
        return 0
    per_layer_token = (
        2
        * kv_heads_per_shard(att.num_kv_heads, tp)
        * att.head_dim
        * KV_BYTES_PER_ELEMENT[cache_dtype]
    )
    full_layers = att.num_layers - att.sliding_window_layers
    swa_tokens = min(context, att.sliding_window) if att.sliding_window else 0
    tokens = full_layers * context + att.sliding_window_layers * swa_tokens
    return int(per_layer_token * tokens * concurrency)


def max_context_for_budget(
    att: Attention, kv_budget: float, concurrency: int, cache_dtype: str, tp: int = 1
) -> int:
    """Максимальный контекст одной последовательности при заданном бюджете KV-кэша."""
    if kv_budget <= 0:
        return 0
    lo, hi = 0, 1 << 22
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if kv_cache_bytes(att, mid, concurrency, cache_dtype, tp) <= kv_budget:
            lo = mid
        else:
            hi = mid - 1
    return lo


def weight_bytes_per_shard(total_weight_bytes: int, tp: int) -> int:
    if tp <= 1:
        return total_weight_bytes
    return int(total_weight_bytes / tp * (1 + TP_REPLICATION_OVERHEAD))


@dataclass
class Estimate:
    weights_bytes: int
    kv_bytes: int
    overhead_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.weights_bytes + self.kv_bytes + self.overhead_bytes
