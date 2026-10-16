"""Inference engine adapters.

An engine adapter owns everything engine-specific:

* ``compatibility``: can this weight variant run on this hardware at all, and how
  sure are we? (OS, accelerator, compute capability, quantization kernels)
* ``fit``: does it fit in memory for the requested context/concurrency, and with
  which parallelism / offloading layout?
* ``build_command`` / ``build_env``: the exact argv and environment to launch it.

Process lifecycle (start, stop, readiness, logs, state files) is identical for
all OpenAI-compatible servers and lives in :mod:`setup_work.inference`. Adding
SGLang means writing one more adapter class and registering it in ``ENGINES``.
"""

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
    dtype: str | None = None  # compute dtype to request (vLLM --dtype)

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
    """Everything needed to start a server. Built by `inference.plan_launch`."""

    name: str
    engine: str
    model_ref: str  # catalog ref or user-supplied id
    model_path: str  # local file (gguf) or local model dir (vllm)
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
        if host in ("0.0.0.0", "::"):  # noqa: S104 - wildcard binds are reached via loopback
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
        """Corporate venv interpreter first, then the managed venv, then PATH."""
        corp = cfg.engines_dir / "vllm" / "current" / "bin" / "python"
        if corp.exists():
            return str(corp)
        return which("vllm", [cfg.engines_dir / "vllm" / ".venv" / "bin"])

    def compatibility(self, model: ModelSpec, variant: Variant, hw: HardwareInfo) -> Compat:
        c = Compat()
        if variant.format != "safetensors" or "vllm" not in variant.engines:
            return c.reject(f"variant {variant.id} ({variant.format}) is not packaged for vLLM")
        if hw.os.system != "Linux":
            return c.reject("vLLM GPU inference requires Linux; use llama.cpp on this OS")
        if hw.accelerator != "nvidia" or not hw.gpus:
            return c.reject("vLLM needs an NVIDIA GPU (CPU/ROCm backends are not managed here)")
        cc = hw.min_compute_capability
        if cc is None:
            c.downgrade("low", "GPU compute capability unknown; kernel support unverified")
            cc = 7.0
        if cc < 7.0:
            return c.reject(f"vLLM requires compute capability >= 7.0 (found {cc})")
        q = variant.quantization
        if q == "none":
            want = variant.dtype or "bf16"
            if want == "bf16" and cc < 8.0:
                if not model.fp16_safe:
                    return c.reject(
                        f"bf16 needs compute capability >= 8.0 (found {cc}) and "
                        f"{model.display_name} is not numerically safe in fp16"
                    )
                c.dtype = "half"
                c.downgrade("medium", f"no bf16 on compute capability {cc}: running in fp16")
            else:
                c.dtype = "bfloat16" if want == "bf16" else "half"
        elif q == "fp8":
            if cc >= 8.9:
                c.reasons.append("native FP8 tensor cores")
            elif cc >= 8.0:
                c.downgrade(
                    "medium", "FP8 runs as weight-only (Marlin) below compute capability 8.9"
                )
            else:
                return c.reject(f"FP8 checkpoints need compute capability >= 8.0 (found {cc})")
        elif q == "awq":
            if cc >= 8.0:
                c.reasons.append("AWQ via Marlin kernels")
            elif cc >= 7.5:
                c.downgrade("medium", "AWQ without Marlin kernels on compute capability 7.5")
            else:
                return c.reject(f"vLLM AWQ kernels need compute capability >= 7.5 (found {cc})")
            if not model.fp16_safe:
                c.downgrade("low", "AWQ runs activations in fp16; model is fp16-sensitive")
        elif q == "mxfp4":
            if cc >= 9.0:
                c.reasons.append("MXFP4 kernels for Hopper/Blackwell")
            elif cc >= 8.0:
                c.downgrade(
                    "medium",
                    "MXFP4 on Ampere/Ada uses fallback kernels; verify with your vLLM version",
                    level="experimental",
                )
            else:
                return c.reject(f"MXFP4 needs compute capability >= 8.0 (found {cc})")
        else:
            return c.reject(f"quantization {q} is not handled by the vLLM adapter")
        if "embedding" in model.workloads:
            c.downgrade("medium", "served with the pooling runner (/v1/embeddings)")
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
                return FitPlan(False, "none", reasons=[f"TP={tp} but only {hw.gpu_count} GPU(s)"])
            if att.num_attention_heads % tp:
                return FitPlan(
                    False,
                    "none",
                    reasons=[f"{att.num_attention_heads} attention heads not divisible by TP={tp}"],
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
                    f"GPU memory already in use: lowering gpu_memory_utilization to {safe_util:.2f}"
                )
                util = round(safe_util - 0.005, 2)
            w = weight_bytes_per_shard(variant.bytes, tp)
            kv_need = kv_cache_bytes(att, req.context, req.concurrency, req.kv_cache_dtype, tp)
            kv_avail = budget - w - overhead
            est = Estimate(w, kv_need, overhead)
            if kv_avail <= 0:
                attempts.append(
                    f"TP={tp}: weights {gib(w)} + overhead {gib(overhead)} exceed "
                    f"per-GPU budget {gib(budget)}"
                )
                continue
            one_seq = kv_cache_bytes(att, req.context, 1, req.kv_cache_dtype, tp)
            if kv_avail < one_seq:
                ctx_max = max_context_for_budget(att, kv_avail, 1, req.kv_cache_dtype, tp)
                attempts.append(
                    f"TP={tp}: KV cache for one {req.context}-token sequence needs {gib(one_seq)}, "
                    f"only {gib(kv_avail)} left (max context ~{ctx_max})"
                )
                continue
            eff_conc = req.concurrency
            if kv_avail < kv_need:
                eff_conc = max(1, int(kv_avail // max(one_seq, 1)))
                warnings.append(
                    f"KV cache holds ~{eff_conc} concurrent {req.context}-token sequences "
                    f"(requested {req.concurrency})"
                )
            reasons = [
                f"TP={tp}: per-GPU weights {gib(w)} + KV {gib(min(kv_need, kv_avail))} + "
                f"overhead {gib(overhead)} within budget {gib(budget)} "
                f"(utilization {util:.2f}, margin {gib(margin)})"
            ]
            if tp > 1:
                if not hw.homogeneous_gpus:
                    warnings.append("heterogeneous GPUs: every rank is limited by the smallest one")
                if att.num_kv_heads < tp:
                    warnings.append(f"{att.num_kv_heads} KV heads < TP={tp}: KV heads replicated")
                warnings.append("multi-GPU TP performance depends on interconnect (NVLink vs PCIe)")
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
            # Prefer a larger TP degree that serves the requested concurrency in full.
            partial = partial or plan
            attempts.append(f"TP={tp}: fits only {eff_conc}/{req.concurrency} concurrent sequences")
        if partial is not None:
            return partial
        if not attempts:
            attempts.append(
                f"no valid tensor-parallel degree for {hw.gpu_count} GPU(s) and "
                f"{att.num_attention_heads} attention heads"
            )
        return FitPlan(False, "none", reasons=attempts)

    def build_command(self, launch: LaunchConfig, binary: str) -> list[str]:
        # A venv unpacked from an archive may have stale script shebangs: run the CLI
        # through its interpreter instead of bin/vllm.
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
            # vLLM reads VLLM_API_KEY; keeps the secret out of argv / `ps` output.
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
            return c.reject(f"variant {variant.id} ({variant.format}) is not a GGUF artifact")
        if hw.os.system not in ("Linux", "Darwin"):
            return c.reject(f"{hw.os.system} is not supported by setup-work")
        if hw.accelerator == "nvidia":
            cc = hw.min_compute_capability
            if cc is None:
                c.downgrade(
                    "medium", "GPU compute capability unknown; CUDA build compatibility assumed"
                )
            elif cc < 6.0:
                c.downgrade("medium", f"compute capability {cc} too old for CUDA builds: CPU only")
            if hw.os.arch != "x86_64":
                c.downgrade("medium", "prebuilt CUDA binaries are x86_64; ARM needs a source build")
        elif hw.accelerator == "apple":
            c.reasons.append("Metal backend on Apple Silicon")
        else:
            c.reasons.append("CPU backend")
        if variant.provenance == "community":
            c.downgrade("medium", "third-party GGUF conversion")
        if variant.quantization == "mxfp4" and hw.accelerator == "nvidia":
            cc = hw.min_compute_capability or 0
            if cc < 7.5:
                c.downgrade("low", "MXFP4 GGUF on pre-Turing GPUs is untested")
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
        # llama.cpp splits -c across --parallel slots; total cache = context x slots.
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
                        f"weights {gib(w)} + KV {gib(kv)} + overhead {gib(overhead)} = {gib(need)} "
                        f"within Metal working set budget {gib(budget)}"
                    ],
                    warnings=["unified memory: other applications compete for the same RAM"]
                    if need > 0.7 * budget
                    else [],
                )
            return FitPlan(
                False,
                "none",
                estimate=est,
                budget_bytes=budget,
                reasons=[f"needs {gib(need)}, Metal working set budget is {gib(budget)}"],
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
                        "layer split across GPUs (no peer-to-peer memory sharing assumed)"
                    )
                reasons.insert(
                    0,
                    f"weights {gib(w)} + KV {gib(kv)} = {gib(w + kv)} within GPU budget "
                    f"{gib(gpu_budget if split else budgets[0])} (after {gib(overhead)} runtime "
                    f"overhead and {gib(margin)} margin per GPU)",
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
            # Partial offload: each offloaded layer brings its weights and KV slice.
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
                        f"{gib(need)} does not fit in GPU budget {gib(gpu_budget)}: offloading "
                        f"{layers}/{att.num_layers} layers, {gib(host_need)} stays in system RAM"
                    ],
                    warnings=["partial CPU offload: expect much lower throughput"],
                )
            reasons.append(
                f"needs {gib(need)}; GPU budget {gib(gpu_budget)} and RAM "
                f"{gib(ram_avail - ram_margin)} insufficient for partial offload"
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
                reasons=[f"CPU inference: {gib(need)} within available RAM budget {gib(budget)}"],
                warnings=["CPU-only inference: throughput is bounded by memory bandwidth"],
            )
        return FitPlan(
            False,
            "none",
            estimate=est,
            budget_bytes=budget,
            reasons=[f"needs {gib(need)}, available RAM budget is {gib(budget)}"],
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
            # -c is the total cache shared by all slots.
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
        """Expose shared libraries shipped next to llama-server in the corporate build."""
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
    # Generation is memory-bound; leaving cores free keeps the machine responsive.
    return max(1, math.ceil(n * 0.75)) if n > 4 else n


ENGINES: dict[str, InferenceEngine] = {"vllm": VllmEngine(), "llama.cpp": LlamaCppEngine()}


def get_engine(name: str) -> InferenceEngine:
    from .errors import UsageError

    aliases = {"llamacpp": "llama.cpp", "llama-cpp": "llama.cpp", "llama_cpp": "llama.cpp"}
    key = aliases.get(name.lower(), name.lower())
    if key not in ENGINES:
        raise UsageError(
            f"unknown engine {name!r}",
            hint=f"choose one of: {', '.join(ENGINES)} (SGLang is planned)",
        )
    return ENGINES[key]
