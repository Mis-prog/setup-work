"""Каталог моделей и подбор под оборудование.

Для каждой пары модель/движок проверяются задача, контекст, совместимость,
память и размещение по устройствам. Подходящие варианты ранжируются по
качеству, квантованию, уверенности оценки и способу размещения.
Причины отказа сохраняются для recommend --explain."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .config import Config, ModelSpec, Variant
from .engines import ENGINES, Compat, FitPlan, FitRequest, get_engine
from .errors import NotFoundError, UsageError
from .hardware import HardwareInfo
from .memory import QUANT_QUALITY

CONFIDENCE_FACTOR = {"high": 1.0, "medium": 0.85, "low": 0.6}
LEVEL_FACTOR = {"supported": 1.0, "experimental": 0.8, "unsupported": 0.0}


def layout_factor(plan: FitPlan, model: ModelSpec, hw: HardwareInfo) -> float:
    if plan.layout in ("full-gpu", "multi-gpu"):
        return 1.0
    if plan.layout == "partial-offload":
        return 0.55
    # CPU: скорость зависит от активных параметров, читаемых для каждого токена.
    active = model.effective_active_b
    if active <= 4:
        return 0.7
    if active <= 10:
        return 0.45
    return 0.25


@dataclass
class ModelRef:
    model_id: str
    variant_id: str | None = None

    @classmethod
    def parse(cls, ref: str) -> ModelRef:
        if ":" in ref:
            m, v = ref.split(":", 1)
            return cls(m, v or None)
        return cls(ref)

    def __str__(self) -> str:
        return f"{self.model_id}:{self.variant_id}" if self.variant_id else self.model_id


def resolve(cfg: Config, ref: str) -> tuple[ModelSpec, Variant | None]:
    r = ModelRef.parse(ref)
    models = cfg.models.models
    if r.model_id not in models:
        close = [m for m in models if r.model_id.lower() in m]
        raise NotFoundError(
            f"model {r.model_id!r} отсутствует в каталоге config/models.yaml",
            hint=(f"возможно, вы имели в виду: {', '.join(close[:5])}? " if close else "")
            + "выполните make models для просмотра идентификаторов",
        )
    spec = models[r.model_id]
    if r.variant_id is None:
        return spec, None
    v = spec.variant(r.variant_id)
    if v is None:
        raise NotFoundError(
            f"model {r.model_id} не имеет варианта {r.variant_id!r}",
            hint=f"доступные варианты: {', '.join(x.id for x in spec.variants)}",
        )
    return spec, v


@dataclass
class Candidate:
    model_id: str
    display_name: str
    engine: str
    variant: str
    quantization: str
    repository: str
    files: list[str] | None
    architecture: str
    parameters_b: float
    active_parameters_b: float | None
    context: int
    max_context: int | None
    concurrency: int
    layout: str
    tensor_parallel: int
    gpu_layers: int | None
    weights_bytes: int
    kv_cache_bytes: int
    overhead_bytes: int
    total_bytes: int
    per_device_bytes: list[int]
    budget_bytes: int | None
    compatibility: str
    confidence: str
    quality: float
    score: float
    rationale: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    gpu_memory_utilization: float | None = None
    tensor_split: list[float] | None = None
    dtype: str | None = None

    @property
    def ref(self) -> str:
        return f"{self.model_id}:{self.variant}"

    def serve_hint(self) -> str:
        parts = [f"make serve MODEL={self.ref}", f"ENGINE={self.engine}", f"CONTEXT={self.context}"]
        if self.tensor_parallel > 1:
            parts.append(f"TP={self.tensor_parallel}")
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ref"] = self.ref
        d["serve"] = self.serve_hint()
        return d


@dataclass
class Rejection:
    model_id: str
    engine: str
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RecommendRequest:
    task: str = "general"
    context: int = 8192
    concurrency: int = 1
    engine: str | None = None
    kv_cache_dtype: str = "fp16"
    tensor_parallel: int | None = None
    gpu_memory_utilization: float = 0.90
    max_results: int = 5


@dataclass
class RecommendResult:
    request: RecommendRequest
    candidates: list[Candidate]
    rejections: list[Rejection]
    hardware_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": asdict(self.request),
            "hardware": self.hardware_summary,
            "recommendations": [c.to_dict() for c in self.candidates],
            "rejected": [r.to_dict() for r in self.rejections],
        }


def evaluate_variant(
    cfg: Config,
    model_id: str,
    model: ModelSpec,
    variant: Variant,
    engine_name: str,
    hw: HardwareInfo,
    req: RecommendRequest,
) -> tuple[Candidate | None, list[str]]:
    engine = get_engine(engine_name)
    compat: Compat = engine.compatibility(model, variant, hw)
    if compat.level == "unsupported":
        return None, [f"{variant.id}: {r}" for r in compat.reasons[-1:]]
    fit_req = FitRequest(
        context=req.context,
        concurrency=req.concurrency,
        kv_cache_dtype=req.kv_cache_dtype,
        tensor_parallel=req.tensor_parallel,
        gpu_memory_utilization=req.gpu_memory_utilization,
    )
    plan = engine.fit(model, variant, hw, fit_req, cfg)
    if not plan.fits or plan.estimate is None:
        return None, [f"{variant.id}: {r}" for r in plan.reasons]
    quality = model.quality.get(req.task, 0.0)
    score = (
        quality
        * QUANT_QUALITY.get(variant.quantization, 0.9)
        * CONFIDENCE_FACTOR[compat.confidence]
        * LEVEL_FACTOR[compat.level]
        * layout_factor(plan, model, hw)
    )
    if req.concurrency > 1:
        score *= 1.05 if engine_name == "vllm" else 0.95
    est = plan.estimate
    rationale = [
        f"{model.architecture}"
        + (
            f", {model.parameters_b:g}B всего / {model.active_parameters_b:g}B active parameters "
            "(в памяти хранятся все эксперты)"
            if model.architecture == "moe"
            else f", {model.parameters_b:g}B parameters"
        ),
        f"{variant.id} artifact {variant.size_gb:g} GB from {variant.repository}",
        *plan.reasons,
        *compat.reasons,
    ]
    cand = Candidate(
        model_id=model_id,
        display_name=model.display_name,
        engine=engine_name,
        variant=variant.id,
        quantization=variant.quantization,
        repository=variant.repository,
        files=variant.files,
        architecture=model.architecture,
        parameters_b=model.parameters_b,
        active_parameters_b=model.active_parameters_b,
        context=req.context,
        max_context=plan.max_context,
        concurrency=plan.effective_concurrency or req.concurrency,
        layout=plan.layout,
        tensor_parallel=plan.tensor_parallel,
        gpu_layers=plan.gpu_layers,
        weights_bytes=est.weights_bytes,
        kv_cache_bytes=est.kv_bytes,
        overhead_bytes=est.overhead_bytes,
        total_bytes=est.total_bytes,
        per_device_bytes=plan.per_device_bytes,
        budget_bytes=plan.budget_bytes,
        compatibility=compat.level,
        confidence=compat.confidence,
        quality=quality,
        score=round(score, 4),
        rationale=rationale,
        warnings=plan.warnings,
        gpu_memory_utilization=plan.gpu_memory_utilization,
        tensor_split=plan.tensor_split,
        dtype=compat.dtype,
    )
    return cand, []


def recommend(cfg: Config, hw: HardwareInfo, req: RecommendRequest) -> RecommendResult:
    if req.task not in ("general", "coding", "reasoning", "agents", "embedding"):
        raise UsageError(
            f"неизвестная задача {req.task!r}",
            hint="выберите general, coding, reasoning, agents или embedding",
        )
    engines = [get_engine(req.engine).name] if req.engine else list(ENGINES)
    candidates: list[Candidate] = []
    rejections: list[Rejection] = []
    for model_id, model in sorted(cfg.models.models.items()):
        if req.task == "embedding":
            if "embedding" not in model.workloads:
                continue
        elif "embedding" in model.workloads or (
            req.task not in model.workloads and model.quality.get(req.task, 0) <= 0
        ):
            continue
        for engine_name in engines:
            variants = [v for v in model.variants if engine_name in v.engines]
            if not variants:
                rejections.append(
                    Rejection(model_id, engine_name, ["нет артефакта для этого движка"])
                )
                continue
            if req.context > model.context_length:
                rejections.append(
                    Rejection(
                        model_id,
                        engine_name,
                        [
                            f"запрошенный контекст {req.context} превышает поддерживаемый контекст "
                            f"{model.context_length} (масштабирование RoPE не настроено)"
                        ],
                    )
                )
                continue
            best: Candidate | None = None
            why: list[str] = []
            for v in variants:
                cand, reasons = evaluate_variant(cfg, model_id, model, v, engine_name, hw, req)
                why.extend(reasons)
                if cand and (
                    best is None
                    or (cand.score, -cand.total_bytes) > (best.score, -best.total_bytes)
                ):
                    best = cand
            if best:
                if req.task not in model.workloads:
                    best.warnings.append(f"нет оценки для '{req.task}'; порядок по качеству")
                candidates.append(best)
            else:
                rejections.append(Rejection(model_id, engine_name, why))
    candidates.sort(key=lambda c: (-c.score, c.total_bytes, c.model_id, c.engine))
    summary = {
        "source": hw.source,
        "os": f"{hw.os.system} {hw.os.arch}",
        "accelerator": hw.accelerator,
        "gpus": [
            {"name": g.name, "memory_total_bytes": g.memory_total_bytes, "cc": g.compute_capability}
            for g in hw.gpus
        ],
        "ram_bytes": hw.memory.effective_total,
    }
    return RecommendResult(req, candidates[: req.max_results], rejections, summary)
