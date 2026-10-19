"""Адаптеры движков инференса.

Адаптер проверяет совместимость формата и железа, оценивает память, подбирает
параллелизм и выгрузку на CPU, формирует команду и окружение сервера.
Запуск, остановка, проверка готовности и логи находятся в inference.py."""

from __future__ import annotations

import ipaddress
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from .config import Config, ModelSpec, Variant
from .hardware import GiB, HardwareInfo
from .memory import (
    Estimate,
    gib,
    kv_cache_bytes,
    max_context_for_budget,
    weight_bytes_per_shard,
)
from .runner import which

Level = Literal["supported", "experimental", "unsupported"]
Confidence = Literal["high", "medium", "low"]
_CONF_ORDER = {"high": 2, "medium": 1, "low": 0}


@dataclass
class Compat:
    level: Level = "supported"
    confidence: Confidence = "high"
    reasons: list[str] = field(default_factory=list)
    dtype: str | None = None  # вычислительный тип для vLLM --dtype

    def downgrade(self, confidence: Confidence, reason: str, level: Level | None = None) -> None:
        if _CONF_ORDER[confidence] < _CONF_ORDER[self.confidence]:
            self.confidence = confidence
        if level == "experimental" and self.level == "supported":
            self.level = level
        self.reasons.append(reason)

    def reject(self, reason: str) -> Compat:
        self.level = "unsupported"
        self.reasons.append(reason)
        return self


@dataclass
class FitRequest:
    context: int
    concurrency: int = 1
    kv_cache_dtype: str = "fp16"
    tensor_parallel: int | None = None
    gpu_memory_utilization: float = 0.90


@dataclass
class FitPlan:
    fits: bool
    layout: str  # full-gpu | multi-gpu | partial-offload | cpu | none
    estimate: Estimate | None = None
    tensor_parallel: int = 1
    gpu_layers: int | None = None
    tensor_split: list[float] | None = None
    per_device_bytes: list[int] = field(default_factory=list)
    budget_bytes: int | None = None
    max_context: int | None = None
    effective_concurrency: int | None = None
    gpu_memory_utilization: float | None = None
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class LaunchConfig:
    """Параметры запуска сервера, подготовленные inference.plan_launch."""

    name: str
    engine: str
    model_ref: str  # ссылка каталога или идентификатор пользователя
    model_path: str  # локальный GGUF или каталог модели vLLM
    served_model_name: str
    host: str = "127.0.0.1"
    port: int = 8000
    context: int | None = None
    concurrency: int = 1
    dtype: str | None = None
    quantization: str | None = None
    tensor_parallel: int = 1
    gpu_memory_utilization: float = 0.90
    gpu_layers: int | None = None
    tensor_split: list[float] | None = None
    threads: int | None = None
    kv_cache_dtype: str | None = None
    embedding: bool = False
    api_key_file: str | None = None
    binary: str | None = None
    extra_args: list[str] = field(default_factory=list)

    @property
    def base_url(self) -> str:
        host = self.host
        if host in ("0.0.0.0", "::"):  # noqa: S104 - обращение к общему адресу выполняется через loopback
            host = "127.0.0.1"
        if ":" in host:
            host = f"[{host}]"
        return f"http://{host}:{self.port}"


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class InferenceEngine(Protocol):
    name: str
    binary_name: str
    health_path: str

    def find_binary(self, cfg: Config) -> str | None: ...
    def compatibility(self, model: ModelSpec, variant: Variant, hw: HardwareInfo) -> Compat: ...
    def fit(
        self, model: ModelSpec, variant: Variant, hw: HardwareInfo, req: FitRequest, cfg: Config
    ) -> FitPlan: ...
    def build_command(self, launch: LaunchConfig, binary: str) -> list[str]: ...
    def build_env(self, launch: LaunchConfig) -> dict[str, str]: ...


# =========================================================================== vLLM


VLLM_LAUNCHER = (
    "import sys; from vllm.entrypoints.cli.main import main; sys.argv[0] = 'vllm'; sys.exit(main())"
)


class VllmEngine:
    name = "vllm"
    binary_name = "vllm"
    health_path = "/health"

    def find_binary(self, cfg: Config) -> str | None:
        """Искать Python сначала в корпоративном venv, затем в управляемом venv и PATH."""
        corp = cfg.engines_dir / "vllm" / "current" / "bin" / "python"
        if corp.exists():
            return str(corp)
        return which("vllm", [cfg.engines_dir / "vllm" / ".venv" / "bin"])

    def compatibility(self, model: ModelSpec, variant: Variant, hw: HardwareInfo) -> Compat:
        c = Compat()
        if variant.format != "safetensors" or "vllm" not in variant.engines:
            return c.reject(f"вариант {variant.id} ({variant.format}) не подготовлен для vLLM")
        if hw.os.system != "Linux":
            return c.reject(
                "для управляемого запуска vLLM на GPU нужна Linux; на этой ОС используйте llama.cpp"
            )
        if hw.accelerator != "nvidia" or not hw.gpus:
            return c.reject(
                "для управляемого запуска vLLM нужен NVIDIA GPU; CPU/ROCm здесь не обслуживаются"
            )
        cc = hw.min_compute_capability
        if cc is None:
            c.downgrade(
                "low",
                "compute capability GPU неизвестна; поддержка вычислительных ядер не проверена",
            )
            cc = 7.0
        if cc < 7.0:
            return c.reject(f"vLLM требует compute capability >= 7.0; обнаружено {cc})")
        q = variant.quantization
        if q == "none":
            want = variant.dtype or "bf16"
            if want == "bf16" and cc < 8.0:
                if not model.fp16_safe:
                    return c.reject(
                        f"bf16 требует compute capability >= 8.0; обнаружено {cc}) and "
                        f"{model.display_name} не обеспечивает численную устойчивость в fp16"
                    )
                c.dtype = "half"
                c.downgrade(
                    "medium", f"bf16 недоступен при compute capability {cc}: используется fp16"
                )
            else:
                c.dtype = "bfloat16" if want == "bf16" else "half"
        elif q == "fp8":
            if cc >= 8.9:
                c.reasons.append("аппаратные тензорные ядра FP8")
            elif cc >= 8.0:
                c.downgrade(
                    "medium",
                    "при compute capability ниже 8.9 FP8 работает через Marlin только для весов",
                )
            else:
                return c.reject(
                    f"для весов FP8 требуется compute capability >= 8.0; обнаружено {cc})"
                )
        elif q == "awq":
            if cc >= 8.0:
                c.reasons.append("AWQ через вычислительные ядра Marlin")
            elif cc >= 7.5:
                c.downgrade("medium", "AWQ без Marlin при compute capability 7.5")
            else:
                return c.reject(
                    f"для AWQ в vLLM требуется compute capability >= 7.5; обнаружено {cc})"
                )
            if not model.fp16_safe:
                c.downgrade("low", "активации AWQ используют fp16, но модель чувствительна к fp16")
        elif q == "mxfp4":
            if cc >= 9.0:
                c.reasons.append("вычислительные ядра MXFP4 для Hopper/Blackwell")
            elif cc >= 8.0:
                c.downgrade(
                    "medium",
                    "MXFP4 на Ampere/Ada использует резервные ядра; проверьте вашу версию vLLM",
                    level="experimental",
                )
            else:
                return c.reject(f"для MXFP4 требуется compute capability >= 8.0; обнаружено {cc})")
        else:
            return c.reject(f"квантование {q} не поддерживается адаптером vLLM")
        if "embedding" in model.workloads:
            c.downgrade("medium", "используется режим эмбеддингов через /v1/embeddings")
        return c

    def _candidate_tp(self, model: ModelSpec, hw: HardwareInfo, req: FitRequest) -> list[int]:
        n = hw.gpu_count
        if req.tensor_parallel:
            return [req.tensor_parallel]
        return [t for t in (1, 2, 4, 8) if t <= n and model.attention.num_attention_heads % t == 0]

    def fit(
        self, model: ModelSpec, variant: Variant, hw: HardwareInfo, req: FitRequest, cfg: Config
    ) -> FitPlan:
        inf = cfg.defaults.inference
        overhead = int(inf.runtime_overhead_gb.get("vllm", 2.0) * GiB)
        margin = int(inf.vram_safety_margin_gb * GiB)
        att = model.attention
        attempts: list[str] = []
        partial: FitPlan | None = None
        for tp in self._candidate_tp(model, hw, req):
            if tp > hw.gpu_count:
                return FitPlan(
                    False, "none", reasons=[f"TP={tp} при доступных {hw.gpu_count} GPU(s)"]
                )
            if att.num_attention_heads % tp:
                return FitPlan(
                    False,
                    "none",
                    reasons=[f"{att.num_attention_heads} голов внимания не делятся на TP={tp}"],
                )
            devs = sorted(hw.gpus, key=lambda g: g.index)[:tp]
            totals = [g.memory_total_bytes or 0 for g in devs]
            frees = [
                g.memory_free_bytes if g.memory_free_bytes is not None else t
                for g, t in zip(devs, totals, strict=True)
            ]
            util = req.gpu_memory_utilization
            budgets = [min(t * util, f - margin) for t, f in zip(totals, frees, strict=True)]
            budget = int(min(budgets))
            warnings: list[str] = []
            if any(f - margin < t * util for t, f in zip(totals, frees, strict=True)):
                safe_util = max(
                    0.1, min((f - margin) / t for t, f in zip(totals, frees, strict=True) if t)
                )
                warnings.append(
                    f"видеопамять уже занята: gpu_memory_utilization уменьшается до {safe_util:.2f}"
                )
                util = round(safe_util - 0.005, 2)
            w = weight_bytes_per_shard(variant.bytes, tp)
            kv_need = kv_cache_bytes(att, req.context, req.concurrency, req.kv_cache_dtype, tp)
            kv_avail = budget - w - overhead
            est = Estimate(w, kv_need, overhead)
            if kv_avail <= 0:
                attempts.append(
                    f"TP={tp}: веса {gib(w)} + накладные расходы {gib(overhead)} превышают "
                    f"бюджет одного GPU {gib(budget)}"
                )
                continue
            one_seq = kv_cache_bytes(att, req.context, 1, req.kv_cache_dtype, tp)
            if kv_avail < one_seq:
                ctx_max = max_context_for_budget(att, kv_avail, 1, req.kv_cache_dtype, tp)
                attempts.append(
                    f"TP={tp}: KV-кэш {req.context} токенов требует {gib(one_seq)}, "
                    f"only {gib(kv_avail)} доступно (макс. контекст примерно{ctx_max})"
                )
                continue
            eff_conc = req.concurrency
            if kv_avail < kv_need:
                eff_conc = max(1, int(kv_avail // max(one_seq, 1)))
                warnings.append(
                    f"KV-кэш: примерно {eff_conc} запросов по {req.context} токенов "
                    f"(запрошено {req.concurrency})"
                )
            reasons = [
                f"TP={tp}: веса на GPU {gib(w)} + KV {gib(min(kv_need, kv_avail))} + "
                f"накладные расходы {gib(overhead)} в пределах бюджета {gib(budget)} "
                f"(использование {util:.2f}, запас {gib(margin)})"
            ]
            if tp > 1:
                if not hw.homogeneous_gpus:
                    warnings.append("разные GPU: каждая доля ограничена наименьшим устройством")
                if att.num_kv_heads < tp:
                    warnings.append(
                        f"{att.num_kv_heads} KV-голов < TP={tp}: KV-головы реплицируются"
                    )
                warnings.append("скорость TP на нескольких GPU зависит от соединения NVLink/PCIe")
            plan = FitPlan(
                True,
                "multi-gpu" if tp > 1 else "full-gpu",
                estimate=est,
                tensor_parallel=tp,
                per_device_bytes=[w + overhead + min(kv_need, kv_avail)] * tp,
                budget_bytes=budget,
                max_context=min(
                    model.context_length,
                    max_context_for_budget(att, kv_avail, 1, req.kv_cache_dtype, tp),
                ),
                effective_concurrency=eff_conc,
                gpu_memory_utilization=util,
                reasons=attempts + reasons,
                warnings=warnings,
            )
            if eff_conc >= req.concurrency:
                return plan
            # Предпочесть больший TP, обеспечивающий запрошенное число одновременных запросов.
            partial = partial or plan
            attempts.append(f"TP={tp}: помещается {eff_conc}/{req.concurrency} запросов")
        if partial is not None:
            return partial
        if not attempts:
            attempts.append(
                f"нет подходящей степени тензорного параллелизма для {hw.gpu_count} GPU и "
                f"{att.num_attention_heads} голов внимания"
            )
        return FitPlan(False, "none", reasons=attempts)

    def build_command(self, launch: LaunchConfig, binary: str) -> list[str]:
        # В распакованном venv строки запуска могут указывать на старые пути: запускать CLI
        # через интерпретатор, а не bin/vllm.
        head = [binary, "-c", VLLM_LAUNCHER] if Path(binary).name.startswith("python") else [binary]
        argv = [*head, "serve", launch.model_path]
        argv += ["--host", launch.host, "--port", str(launch.port)]
        argv += ["--served-model-name", launch.served_model_name]
        if launch.context:
            argv += ["--max-model-len", str(launch.context)]
        if launch.dtype:
            argv += ["--dtype", launch.dtype]
        if launch.quantization and launch.quantization not in ("none", "awq", "fp8", "mxfp4"):
            argv += ["--quantization", launch.quantization]
        if launch.tensor_parallel > 1:
            argv += ["--tensor-parallel-size", str(launch.tensor_parallel)]
        argv += ["--gpu-memory-utilization", f"{launch.gpu_memory_utilization:.2f}"]
        if launch.concurrency > 1:
            argv += ["--max-num-seqs", str(launch.concurrency)]
        if launch.kv_cache_dtype == "fp8":
            argv += ["--kv-cache-dtype", "fp8"]
        if launch.embedding:
            argv += ["--runner", "pooling"]
        argv += launch.extra_args
        return argv

    def build_env(self, launch: LaunchConfig) -> dict[str, str]:
        env: dict[str, str] = {}
        if launch.binary and Path(launch.binary).name.startswith("python"):
            venv = Path(launch.binary).parent.parent
            env["VIRTUAL_ENV"] = str(venv)
            env["PATH"] = os.pathsep.join([str(venv / "bin"), os.environ.get("PATH", "")])
        if launch.api_key_file:
            # vLLM читает VLLM_API_KEY; ключ не попадает в argv и вывод ps.
            env["VLLM_API_KEY"] = Path(launch.api_key_file).read_text(encoding="utf-8").strip()
        return env


# ====================================================================== llama.cpp


class LlamaCppEngine:
    name = "llama.cpp"
    binary_name = "llama-server"
    health_path = "/health"

    def find_binary(self, cfg: Config) -> str | None:
        current = cfg.engines_dir / "llama.cpp" / "current"
        return which("llama-server", [current])

    def compatibility(self, model: ModelSpec, variant: Variant, hw: HardwareInfo) -> Compat:
        c = Compat()
        if variant.format != "gguf" or "llama.cpp" not in variant.engines:
            return c.reject(f"вариант {variant.id} ({variant.format}) не является GGUF-артефактом")
        if hw.os.system not in ("Linux", "Darwin"):
            return c.reject(f"{hw.os.system} не поддерживается setup-work")
        if hw.accelerator == "nvidia":
            cc = hw.min_compute_capability
            if cc is None:
                c.downgrade(
                    "medium",
                    "compute capability неизвестна; совместимость сборки CUDA предполагается",
                )
            elif cc < 6.0:
                c.downgrade(
                    "medium",
                    f"compute capability {cc} недостаточна для сборок CUDA: доступен только CPU",
                )
            if hw.os.arch != "x86_64":
                c.downgrade(
                    "medium",
                    "готовые CUDA-бинарники рассчитаны на x86_64; для ARM нужна собственная сборка",
                )
        elif hw.accelerator == "apple":
            c.reasons.append("Metal на Apple Silicon")
        else:
            c.reasons.append("вычисления на CPU")
        if variant.provenance == "community":
            c.downgrade("medium", "сторонняя конвертация в GGUF")
        if variant.quantization == "mxfp4" and hw.accelerator == "nvidia":
            cc = hw.min_compute_capability or 0
            if cc < 7.5:
                c.downgrade("low", "MXFP4 GGUF не проверен на GPU старше Turing")
        return c

    def fit(
        self, model: ModelSpec, variant: Variant, hw: HardwareInfo, req: FitRequest, cfg: Config
    ) -> FitPlan:
        inf = cfg.defaults.inference
        overhead = int(inf.runtime_overhead_gb.get("llama.cpp", 0.6) * GiB)
        margin = int(inf.vram_safety_margin_gb * GiB)
        ram_margin = int(cfg.defaults.recommend.ram_safety_margin_gb * GiB)
        att = model.attention
        w = variant.bytes
        # llama.cpp делит -c между слотами --parallel; общий кэш = контекст × слоты.
        kv = kv_cache_bytes(att, req.context, req.concurrency, req.kv_cache_dtype)
        need = w + kv + overhead
        est = Estimate(w, kv, overhead)
        ram_total = hw.memory.effective_total or 0
        ram_avail = hw.memory.available_bytes or ram_total
        reasons: list[str] = []

        usable_gpus = []
        if hw.accelerator == "nvidia":
            cc = hw.min_compute_capability
            if cc is None or cc >= 6.0:
                usable_gpus = hw.gpus
        elif hw.accelerator == "apple":
            usable_gpus = hw.gpus

        if hw.accelerator == "apple" and usable_gpus:
            ws = hw.metal_working_set_bytes or 0
            budget = min(ws, ram_total - ram_margin) - margin
            if need <= budget:
                return FitPlan(
                    True,
                    "full-gpu",
                    estimate=est,
                    gpu_layers=-1,
                    per_device_bytes=[need],
                    budget_bytes=budget,
                    max_context=self._max_ctx(model, budget - w - overhead, req),
                    effective_concurrency=req.concurrency,
                    reasons=[
                        f"веса {gib(w)} + KV {gib(kv)} + запас {gib(overhead)} = {gib(need)} "
                        f"within Metal working set budget {gib(budget)}"
                    ],
                    warnings=["общая память: другие приложения используют ту же RAM"]
                    if need > 0.7 * budget
                    else [],
                )
            return FitPlan(
                False,
                "none",
                estimate=est,
                budget_bytes=budget,
                reasons=[f"требуется {gib(need)}, бюджет рабочего объёма Metal {gib(budget)}"],
            )

        if usable_gpus:
            budgets = []
            for g in usable_gpus:
                total = g.memory_total_bytes or 0
                free = g.memory_free_bytes if g.memory_free_bytes is not None else total
                budgets.append(max(0, free - margin - overhead))
            gpu_budget = sum(budgets)
            if w + kv <= gpu_budget:
                split = None
                layout = "full-gpu"
                if len(usable_gpus) > 1 and w + kv > budgets[0]:
                    tot = sum(budgets)
                    split = [round(b / tot, 3) for b in budgets]
                    layout = "multi-gpu"
                    reasons.append(
                        "распределение слоёв между GPU без предположения об общей памяти P2P"
                    )
                reasons.insert(
                    0,
                    f"веса {gib(w)} + KV {gib(kv)} = {gib(w + kv)} в пределах бюджета GPU "
                    f"{gib(gpu_budget if split else budgets[0])} (после {gib(overhead)} накладных "
                    f"расходов и {gib(margin)} запаса на каждый GPU)",
                )
                return FitPlan(
                    True,
                    layout,
                    estimate=est,
                    gpu_layers=-1,
                    tensor_split=split,
                    per_device_bytes=[int(need * s) for s in split] if split else [need],
                    budget_bytes=gpu_budget,
                    max_context=self._max_ctx(model, gpu_budget - w, req),
                    effective_concurrency=req.concurrency,
                    reasons=reasons,
                )
            # При частичной выгрузке каждый слой переносится со своими весами и KV-кэшем.
            per_layer = (w + kv) / att.num_layers
            layers = int(gpu_budget // per_layer) if per_layer else 0
            layers = min(layers, att.num_layers - 1)
            host_need = need - int(layers * per_layer)
            if layers >= 1 and host_need <= ram_avail - ram_margin:
                return FitPlan(
                    True,
                    "partial-offload",
                    estimate=est,
                    gpu_layers=layers,
                    per_device_bytes=[int(layers * per_layer)],
                    budget_bytes=gpu_budget,
                    max_context=None,
                    effective_concurrency=req.concurrency,
                    reasons=[
                        f"{gib(need)} не помещается в бюджет GPU {gib(gpu_budget)}: выгрузка "
                        f"{layers}/{att.num_layers} слоёв; в RAM остаётся {gib(host_need)}"
                    ],
                    warnings=["частичная выгрузка на CPU существенно снижает скорость"],
                )
            reasons.append(
                f"требуется {gib(need)}; бюджет GPU {gib(gpu_budget)} и RAM "
                f"{gib(ram_avail - ram_margin)} недостаточен для частичной выгрузки"
            )
            return FitPlan(False, "none", estimate=est, budget_bytes=gpu_budget, reasons=reasons)

        budget = ram_avail - ram_margin
        if need <= budget:
            return FitPlan(
                True,
                "cpu",
                estimate=est,
                gpu_layers=0,
                per_device_bytes=[need],
                budget_bytes=budget,
                max_context=self._max_ctx(model, budget - w - overhead, req),
                effective_concurrency=req.concurrency,
                reasons=[f"инференс на CPU: {gib(need)} в пределах доступной RAM {gib(budget)}"],
                warnings=["скорость инференса на CPU ограничена пропускной способностью памяти"],
            )
        return FitPlan(
            False,
            "none",
            estimate=est,
            budget_bytes=budget,
            reasons=[f"требуется {gib(need)}, доступный бюджет RAM {gib(budget)}"],
        )

    @staticmethod
    def _max_ctx(model: ModelSpec, kv_budget: float, req: FitRequest) -> int:
        return min(
            model.context_length,
            max_context_for_budget(model.attention, kv_budget, req.concurrency, req.kv_cache_dtype),
        )

    def build_command(self, launch: LaunchConfig, binary: str) -> list[str]:
        argv = [binary, "-m", launch.model_path, "--host", launch.host, "--port", str(launch.port)]
        argv += ["-a", launch.served_model_name]
        if launch.context:
            # -c задаёт общий объём кэша для всех слотов.
            argv += ["-c", str(launch.context * max(1, launch.concurrency))]
        argv += ["-np", str(max(1, launch.concurrency))]
        if launch.gpu_layers is not None:
            argv += ["-ngl", "all" if launch.gpu_layers < 0 else str(launch.gpu_layers)]
        if launch.tensor_split:
            argv += ["-sm", "layer", "-ts", ",".join(f"{x:g}" for x in launch.tensor_split)]
        threads = launch.threads or _default_threads()
        if threads:
            argv += ["-t", str(threads)]
        if launch.kv_cache_dtype and launch.kv_cache_dtype in ("q8_0",):
            argv += ["-ctk", "q8_0", "-ctv", "q8_0", "-fa", "on"]
        if launch.embedding:
            argv += ["--embedding"]
        if launch.api_key_file:
            argv += ["--api-key-file", launch.api_key_file]
        argv += launch.extra_args
        return argv

    def build_env(self, launch: LaunchConfig) -> dict[str, str]:
        return {}

    @staticmethod
    def runtime_env(binary: str) -> dict[str, str]:
        """Добавить в окружение библиотеки, лежащие рядом с llama-server."""
        bdir = Path(binary).resolve().parent
        if not any(bdir.glob("*.so*")):
            return {}
        dirs = [str(bdir)]
        current = os.environ.get("LD_LIBRARY_PATH")
        return {"LD_LIBRARY_PATH": os.pathsep.join(dirs + ([current] if current else []))}


def _default_threads() -> int | None:
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    if not n:
        return None
    # Генерация ограничена памятью; свободные ядра сохраняют отзывчивость системы.
    return max(1, math.ceil(n * 0.75)) if n > 4 else n


ENGINES: dict[str, InferenceEngine] = {"vllm": VllmEngine(), "llama.cpp": LlamaCppEngine()}


def get_engine(name: str) -> InferenceEngine:
    from .errors import UsageError

    aliases = {"llamacpp": "llama.cpp", "llama-cpp": "llama.cpp", "llama_cpp": "llama.cpp"}
    key = aliases.get(name.lower(), name.lower())
    if key not in ENGINES:
        raise UsageError(
            f"неизвестный движок {name!r}",
            hint=f"выберите один из: {', '.join(ENGINES)} (SGLang пока не реализован)",
        )
    return ENGINES[key]
