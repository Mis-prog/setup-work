"""Memory estimation primitives.

All results are estimates in bytes. The model is deliberately explicit so every
number in a recommendation can be explained:

* weights   = exact artifact size from the catalog (what is loaded), divided across
              tensor-parallel shards plus a small replication overhead;
* KV cache  = 2 (K and V) x layers x kv_heads x head_dim x bytes/element x tokens,
              where sliding-window layers only cache ``min(tokens, window)`` tokens and
              KV heads are replicated when the TP degree exceeds the KV-head count;
* overhead  = fixed per-device runtime cost (CUDA context, graphs, activations).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import Attention

GiB = 1024**3

KV_BYTES_PER_ELEMENT = {"fp32": 4.0, "fp16": 2.0, "bf16": 2.0, "fp8": 1.0, "q8_0": 34 / 32}

# Relative quality retained by each weight format (ordering only, not a benchmark).
QUANT_QUALITY = {
    "none": 1.00,
    "f16": 1.00,
    "mxfp4": 1.00,  # gpt-oss is released natively in MXFP4
    "fp8": 0.99,
    "q8_0": 0.99,
    "q6_k": 0.97,
    "q5_k_m": 0.95,
    "awq": 0.94,
    "q4_k_m": 0.92,
}

# Replicated tensors (embeddings, norms) when sharding weights across GPUs.
TP_REPLICATION_OVERHEAD = 0.02


def gib(n: float | int | None) -> str:
    return "n/a" if n is None else f"{n / GiB:.1f} GiB"


def kv_heads_per_shard(num_kv_heads: int, tp: int) -> int:
    """vLLM shards KV heads across TP ranks and replicates them when tp > heads."""
    return max(1, math.ceil(num_kv_heads / tp))


def kv_cache_bytes(
    att: Attention,
    context: int,
    concurrency: int = 1,
    cache_dtype: str = "fp16",
    tp: int = 1,
) -> int:
    """KV cache for `concurrency` sequences of `context` tokens, per TP shard."""
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
    """Largest per-sequence context whose KV cache fits in `kv_budget` bytes."""
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
