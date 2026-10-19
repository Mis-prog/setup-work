from __future__ import annotations

import pytest

from setup_work.errors import NotFoundError, UsageError
from setup_work.models import RecommendRequest, evaluate_variant, recommend, resolve


def rec(cfg, hw, **kw):
    kw.setdefault("max_results", 50)
    return recommend(cfg, hw, RecommendRequest(**kw))


def test_cpu_only_never_recommends_vllm(cfg, hw_cpu):
    res = rec(cfg, hw_cpu, task="coding", context=8192)
    assert res.candidates
    assert {c.engine for c in res.candidates} == {"llama.cpp"}
    assert all(c.layout == "cpu" for c in res.candidates)
    vllm_rej = [r for r in res.rejections if r.engine == "vllm"]
    assert vllm_rej and any("NVIDIA" in why for r in vllm_rej for why in r.reasons)


def test_macos_uses_metal_llama_cpp_only(cfg, hw_mac):
    res = rec(cfg, hw_mac, task="general", context=8192)
    assert {c.engine for c in res.candidates} == {"llama.cpp"}
    assert all(c.layout == "full-gpu" for c in res.candidates)


def test_v100_rejects_awq_and_fp8_and_falls_back_to_fp16(cfg, hw_v100):
    spec, awq = resolve(cfg, "qwen3-8b:awq")
    cand, why = evaluate_variant(
        cfg, "qwen3-8b", spec, awq, "vllm", hw_v100, RecommendRequest(context=8192)
    )
    assert cand is None and any("7.5" in w for w in why)
    _, fp8 = resolve(cfg, "qwen3-8b:fp8")
    cand, why = evaluate_variant(
        cfg, "qwen3-8b", spec, fp8, "vllm", hw_v100, RecommendRequest(context=8192)
    )
    assert cand is None and any("FP8" in w for w in why)
    _, bf16 = resolve(cfg, "qwen3-8b:bf16")
    cand, _ = evaluate_variant(
        cfg, "qwen3-8b", spec, bf16, "vllm", hw_v100, RecommendRequest(context=8192)
    )
    assert cand is not None and cand.dtype == "half" and cand.confidence == "medium"


def test_fp16_unsafe_model_rejected_on_v100(cfg, hw_v100):
    spec, bf16 = resolve(cfg, "gemma-3-12b:bf16")
    cand, why = evaluate_variant(
        cfg, "gemma-3-12b", spec, bf16, "vllm", hw_v100, RecommendRequest(context=8192)
    )
    assert cand is None and any("fp16" in w for w in why)


def test_fp8_native_on_ada_weight_only_on_ampere(cfg, hw_4090, hw_a100x4):
    spec, fp8 = resolve(cfg, "qwen3-8b:fp8")
    c1, _ = evaluate_variant(
        cfg, "qwen3-8b", spec, fp8, "vllm", hw_4090, RecommendRequest(context=8192)
    )
    c2, _ = evaluate_variant(
        cfg, "qwen3-8b", spec, fp8, "vllm", hw_a100x4, RecommendRequest(context=8192)
    )
    assert c1.confidence == "high"
    assert c2.confidence == "medium"


def test_multi_gpu_tensor_parallel_needed_for_large_bf16(cfg, hw_a100x4):
    spec, bf16 = resolve(cfg, "qwen3-32b:bf16")  # 65,5 GB весов
    single, _ = evaluate_variant(
        cfg, "qwen3-32b", spec, bf16, "vllm", hw_a100x4, RecommendRequest(context=32768)
    )
    assert single is not None and single.tensor_parallel == 1  # 61 + 8 + 2 GiB < 72 GiB
    cand, _ = evaluate_variant(
        cfg,
        "qwen3-32b",
        spec,
        bf16,
        "vllm",
        hw_a100x4,
        RecommendRequest(context=32768, concurrency=4),
    )
    assert cand is not None
    assert cand.tensor_parallel == 2
    assert cand.layout == "multi-gpu"
    assert cand.weights_bytes < bf16.bytes  # веса на одну долю


def test_weights_that_fit_are_not_enough(cfg, hw_4090):
    """Веса Qwen3-14B q8_0 помещаются в 24 GB, но KV-кэш 32k и четыре запроса уже не помещаются."""
    spec, q8 = resolve(cfg, "qwen3-14b:q8_0")
    ok, _ = evaluate_variant(
        cfg, "qwen3-14b", spec, q8, "llama.cpp", hw_4090, RecommendRequest(context=4096)
    )
    assert ok is not None and ok.layout == "full-gpu"
    big, why = evaluate_variant(
        cfg,
        "qwen3-14b",
        spec,
        q8,
        "llama.cpp",
        hw_4090,
        RecommendRequest(context=32768, concurrency=4),
    )
    assert big is None or big.layout == "partial-offload"


def test_moe_memory_uses_total_parameters(cfg, hw_4090):
    spec, q4 = resolve(cfg, "qwen3-30b-a3b:q4_k_m")
    cand, _ = evaluate_variant(
        cfg, "qwen3-30b-a3b", spec, q4, "llama.cpp", hw_4090, RecommendRequest(context=8192)
    )
    assert cand.weights_bytes == q4.bytes  # загружены все эксперты
    assert any("все эксперты" in r for r in cand.rationale)


def test_context_beyond_native_rejected(cfg, hw_a100x4):
    res = rec(cfg, hw_a100x4, task="coding", context=200_000)
    ids = {c.model_id for c in res.candidates}
    assert ids <= {"qwen3-coder-30b-a3b"}
    assert any("поддерживаемый контекст" in why for r in res.rejections for why in r.reasons)


def test_embedding_task_only_returns_embedding_models(cfg, hw_cpu):
    res = rec(cfg, hw_cpu, task="embedding", context=2048)
    assert res.candidates and all(c.model_id == "qwen3-embedding-0.6b" for c in res.candidates)
    gen = rec(cfg, hw_cpu, task="general", context=2048)
    assert all(c.model_id != "qwen3-embedding-0.6b" for c in gen.candidates)


def test_engine_filter(cfg, hw_a100x4):
    res = rec(cfg, hw_a100x4, task="coding", engine="vllm")
    assert res.candidates and all(c.engine == "vllm" for c in res.candidates)
    with pytest.raises(UsageError):
        rec(cfg, hw_a100x4, engine="tensorrt")


def test_deterministic_and_sorted(cfg, hw_mixed):
    a = rec(cfg, hw_mixed, task="agents", context=16384)
    b = rec(cfg, hw_mixed, task="agents", context=16384)
    assert [c.ref + c.engine for c in a.candidates] == [c.ref + c.engine for c in b.candidates]
    scores = [c.score for c in a.candidates]
    assert scores == sorted(scores, reverse=True)


def test_heterogeneous_layer_split_for_llama_cpp(cfg, hw_mixed):
    spec, q8 = resolve(cfg, "qwen2.5-coder-32b:q8_0")
    cand, _ = evaluate_variant(
        cfg, "qwen2.5-coder-32b", spec, q8, "llama.cpp", hw_mixed, RecommendRequest(context=32768)
    )
    assert cand.layout == "multi-gpu"
    assert cand.tensor_split and abs(sum(cand.tensor_split) - 1) < 0.01
    assert cand.tensor_split[0] > cand.tensor_split[1]  # A100 получает большую долю


def test_explicit_tp_must_divide_heads(cfg, hw_a100x4):
    spec, bf16 = resolve(cfg, "qwen2.5-coder-7b:bf16")  # 28 голов внимания
    cand, why = evaluate_variant(
        cfg,
        "qwen2.5-coder-7b",
        spec,
        bf16,
        "vllm",
        hw_a100x4,
        RecommendRequest(context=8192, tensor_parallel=8),
    )
    assert cand is None
    assert any("TP=8" in w for w in why)


def test_unknown_task_and_model(cfg, hw_cpu):
    with pytest.raises(UsageError):
        rec(cfg, hw_cpu, task="poetry")
    with pytest.raises(NotFoundError):
        resolve(cfg, "nope-1b")
    with pytest.raises(NotFoundError, match="не имеет варианта"):
        resolve(cfg, "qwen3-8b:q2_k")


def test_candidates_carry_explanations(cfg, hw_4090):
    res = rec(cfg, hw_4090, task="coding", context=16384, max_results=3)
    top = res.candidates[0]
    assert top.rationale and top.serve_hint().startswith("make serve MODEL=")
    d = res.to_dict()
    assert {"recommendations", "rejected", "hardware", "request"} <= set(d)
    assert d["recommendations"][0]["confidence"] in ("high", "medium", "low")
