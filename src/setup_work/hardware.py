"""Определение оборудования с обработкой недоступных источников.

Отсутствующий nvidia-smi, недоступный /proc или неожиданный формат ответа
оставляют неизвестные поля и пояснение вместо аварийного завершения.
SETUP_WORK_HARDWARE_FILE позволяет использовать сохранённый профиль машины
вместо определения текущего оборудования."""

from __future__ import annotations

import contextlib
import json
import os
import platform
import re
import shutil
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .runner import Result, probe

GiB = 1024**3
MiB = 1024**2

ProbeFn = Callable[[Sequence[str]], Result]

# Резервное определение compute_cap для драйверов старше R510.
_CC_BY_NAME: list[tuple[str, str]] = [
    (r"\bB(100|200)\b|GB200", "10.0"),
    (r"RTX PRO 6000|RTX 50\d0", "12.0"),
    (r"\bH(100|200|800|20)\b|GH200", "9.0"),
    (r"\bL4\b|\bL40S?\b|RTX 40\d0|RTX \d000 Ada", "8.9"),
    (r"\bA(10G?|16|2|40|30)\b|RTX 30\d0|RTX A\d000", "8.6"),
    (r"\bA100\b|\bA800\b|\bA30\b", "8.0"),
    (r"\bT4\b|RTX 20\d0|Quadro RTX|TITAN RTX", "7.5"),
    (r"\bV100\b|TITAN V", "7.0"),
    (r"\bP100\b", "6.0"),
    (r"\bP40\b|\bP4\b|GTX 10\d0", "6.1"),
]


@dataclass
class OSInfo:
    system: str  # Linux | Darwin
    release: str
    distro: str | None
    arch: str  # x86_64 | arm64
    container: bool = False
    wsl: bool = False


@dataclass
class CPUInfo:
    model: str | None
    logical_cores: int | None
    physical_cores: int | None
    usable_cores: int | None


@dataclass
class MemoryInfo:
    total_bytes: int | None
    available_bytes: int | None
    cgroup_limit_bytes: int | None = None

    @property
    def effective_total(self) -> int | None:
        vals = [v for v in (self.total_bytes, self.cgroup_limit_bytes) if v]
        return min(vals) if vals else None


@dataclass
class GPU:
    index: int
    vendor: str  # nvidia | apple | amd
    name: str
    memory_total_bytes: int | None
    memory_free_bytes: int | None
    compute_capability: str | None = None
    compute_capability_source: str | None = None
    unified_memory: bool = False
    uuid: str | None = None
    pci_bus_id: str | None = None

    @property
    def cc(self) -> float | None:
        try:
            return float(self.compute_capability) if self.compute_capability else None
        except ValueError:
            return None


@dataclass
class StorageInfo:
    path: str
    total_bytes: int
    free_bytes: int


@dataclass
class HardwareInfo:
    os: OSInfo
    cpu: CPUInfo
    memory: MemoryInfo
    gpus: list[GPU] = field(default_factory=list)
    accelerator: str = "cpu"  # nvidia | apple | amd | cpu
    nvidia_driver: str | None = None
    cuda_driver_version: str | None = None
    cuda_toolkit_version: str | None = None
    metal_working_set_bytes: int | None = None
    metal_working_set_source: str | None = None
    precisions: list[str] = field(default_factory=list)
    storage: list[StorageInfo] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    source: str = "live"

    # ------------------------------------------------------------ вычисляемые свойства
    @property
    def gpu_count(self) -> int:
        return len(self.gpus)

    @property
    def total_vram_bytes(self) -> int | None:
        """Сумма видеопамяти отдельных устройств, не единый доступный пул."""
        if not self.gpus or self.accelerator == "apple":
            return None
        vals = [g.memory_total_bytes for g in self.gpus]
        return sum(vals) if all(v is not None for v in vals) else None

    @property
    def homogeneous_gpus(self) -> bool:
        return len({(g.name, g.memory_total_bytes) for g in self.gpus}) <= 1

    @property
    def min_compute_capability(self) -> float | None:
        ccs = [g.cc for g in self.gpus if g.vendor == "nvidia"]
        if not ccs or any(c is None for c in ccs):
            return None
        return min(c for c in ccs if c is not None)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["summary"] = {
            "gpu_count": self.gpu_count,
            "total_vram_bytes": self.total_vram_bytes,
            "total_vram_note": "сумма памяти отдельных устройств, не общий пул",
            "homogeneous_gpus": self.homogeneous_gpus,
            "min_compute_capability": self.min_compute_capability,
            "effective_ram_bytes": self.memory.effective_total,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HardwareInfo:
        return cls(
            os=OSInfo(**d["os"]),
            cpu=CPUInfo(**d["cpu"]),
            memory=MemoryInfo(**d["memory"]),
            gpus=[GPU(**g) for g in d.get("gpus", [])],
            accelerator=d.get("accelerator", "cpu"),
            nvidia_driver=d.get("nvidia_driver"),
            cuda_driver_version=d.get("cuda_driver_version"),
            cuda_toolkit_version=d.get("cuda_toolkit_version"),
            metal_working_set_bytes=d.get("metal_working_set_bytes"),
            metal_working_set_source=d.get("metal_working_set_source"),
            precisions=list(d.get("precisions", [])),
            storage=[StorageInfo(**s) for s in d.get("storage", [])],
            notes=list(d.get("notes", [])),
            source=d.get("source", "file"),
        )


# ------------------------------------------------------------------ вспомогательные функции


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def nvidia_precisions(cc: float | None) -> list[str]:
    if cc is None:
        return ["fp32", "fp16"]
    out = ["fp32"]
    if cc >= 6.0:
        out.append("fp16")
    if cc >= 7.5:
        out.append("int8")
    if cc >= 8.0:
        out.append("bf16")
    if cc >= 8.9:
        out.append("fp8")
    if cc >= 10.0:
        out.append("fp4")
    return out


def cc_from_name(name: str) -> str | None:
    for pattern, cc in _CC_BY_NAME:
        if re.search(pattern, name):
            return cc
    return None


def parse_nvidia_smi_csv(text: str, with_cc: bool) -> list[GPU]:
    gpus: list[GPU] = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        expected = 8 if with_cc else 7
        if len(parts) < expected:
            continue

        def num(s: str) -> int | None:
            try:
                return int(float(s)) * MiB
            except ValueError:
                return None

        idx, name, uuid, total, free, bus = (
            parts[0],
            parts[1],
            parts[2],
            parts[3],
            parts[4],
            parts[5],
        )
        cc = parts[7] if with_cc else None
        cc_src = "nvidia-smi" if cc and re.fullmatch(r"\d+\.\d+", cc) else None
        if not cc_src:
            cc = cc_from_name(name)
            cc_src = "name-lookup" if cc else None
        gpus.append(
            GPU(
                index=int(idx) if idx.isdigit() else len(gpus),
                vendor="nvidia",
                name=name,
                uuid=uuid or None,
                memory_total_bytes=num(total),
                memory_free_bytes=num(free),
                pci_bus_id=bus or None,
                compute_capability=cc,
                compute_capability_source=cc_src,
            )
        )
    return gpus


def detect_nvidia(run: ProbeFn) -> tuple[list[GPU], str | None, str | None, list[str]]:
    notes: list[str] = []
    if not shutil.which("nvidia-smi") and run is probe:
        return [], None, None, notes
    base = "index,name,uuid,memory.total,memory.free,pci.bus_id,driver_version"
    res = run(["nvidia-smi", f"--query-gpu={base},compute_cap", "--format=csv,noheader,nounits"])
    with_cc = True
    if not res.ok:
        res = run(["nvidia-smi", f"--query-gpu={base}", "--format=csv,noheader,nounits"])
        with_cc = False
        if res.ok:
            notes.append("драйвер не сообщает compute capability; значение определено по имени GPU")
    if not res.ok:
        if res.returncode != 127:
            notes.append(f"ошибка nvidia-smi: {(res.stderr or res.stdout).strip()[:200]}")
        return [], None, None, notes
    gpus = parse_nvidia_smi_csv(res.stdout, with_cc)
    driver = None
    first = res.stdout.strip().splitlines()[0].split(",") if res.stdout.strip() else []
    if len(first) >= 7:
        driver = first[6].strip() or None
    cuda = None
    head = run(["nvidia-smi"])
    if head.ok:
        m = re.search(r"CUDA Version:\s*([\d.]+)", head.stdout)
        cuda = m.group(1) if m else None
    return gpus, driver, cuda, notes


def detect_cuda_toolkit(run: ProbeFn) -> str | None:
    res = run(["nvcc", "--version"])
    if res.ok:
        m = re.search(r"release\s+([\d.]+)", res.stdout)
        return m.group(1) if m else None
    return None


def _sysctl(run: ProbeFn, key: str) -> str | None:
    res = run(["sysctl", "-n", key])
    return res.stdout.strip() if res.ok and res.stdout.strip() else None


def parse_vm_stat(text: str) -> int | None:
    m = re.search(r"page size of (\d+) bytes", text)
    if not m:
        return None
    page = int(m.group(1))
    pages = 0
    for key in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
        mm = re.search(rf"{key}:\s+(\d+)", text)
        if mm:
            pages += int(mm.group(1))
    return pages * page


def parse_meminfo(text: str) -> tuple[int | None, int | None]:
    vals: dict[str, int] = {}
    for line in text.splitlines():
        m = re.match(r"(\w+):\s+(\d+)\s*kB", line)
        if m:
            vals[m.group(1)] = int(m.group(2)) * 1024
    return vals.get("MemTotal"), vals.get("MemAvailable", vals.get("MemFree"))


def cgroup_memory_limit() -> int | None:
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        raw = _read(path)
        if raw is None:
            continue
        raw = raw.strip()
        if raw.isdigit() and int(raw) < (1 << 60):
            return int(raw)
    return None


def metal_working_set(total_ram: int | None, wired_limit_mb: str | None) -> tuple[int | None, str]:
    """Оценить рабочий объём Metal с учётом RAM и iogpu.wired_limit_mb."""
    if wired_limit_mb and wired_limit_mb.isdigit() and int(wired_limit_mb) > 0:
        return int(wired_limit_mb) * MiB, "iogpu.wired_limit_mb"
    if not total_ram:
        return None, "unknown"
    frac = 0.75 if total_ram > 36 * GiB else 0.67
    return int(total_ram * frac), f"оценка: {int(frac * 100)}% RAM"


def detect_os() -> OSInfo:
    system = platform.system()
    arch = platform.machine().lower()
    arch = {"aarch64": "arm64", "amd64": "x86_64"}.get(arch, arch)
    distro = None
    container = False
    wsl = False
    if system == "Linux":
        osr = _read("/etc/os-release") or ""
        m = re.search(r'^PRETTY_NAME="?([^"\n]+)"?', osr, re.M)
        distro = m.group(1) if m else None
        cg = _read("/proc/1/cgroup") or ""
        container = (
            Path("/.dockerenv").exists()
            or Path("/run/.containerenv").exists()
            or any(k in cg for k in ("docker", "kubepods", "containerd", "lxc"))
            or "KUBERNETES_SERVICE_HOST" in os.environ
        )
        wsl = "microsoft" in platform.release().lower()
    elif system == "Darwin":
        distro = f"macOS {platform.mac_ver()[0]}".strip()
    return OSInfo(system, platform.release(), distro, arch, container, wsl)


def detect_cpu(run: ProbeFn, system: str) -> CPUInfo:
    model = None
    physical = None
    logical = os.cpu_count()
    usable = logical
    if hasattr(os, "sched_getaffinity"):
        with contextlib.suppress(OSError):
            usable = len(os.sched_getaffinity(0))
    if system == "Darwin":
        model = _sysctl(run, "machdep.cpu.brand_string")
        p = _sysctl(run, "hw.physicalcpu")
        physical = int(p) if p and p.isdigit() else None
    else:
        info = _read("/proc/cpuinfo") or ""
        m = re.search(r"^model name\s*:\s*(.+)$", info, re.M)
        model = m.group(1).strip() if m else None
        cores = {
            (a, b)
            for a, b in re.findall(r"physical id\s*:\s*(\d+).*?core id\s*:\s*(\d+)", info, re.S)
        }
        physical = len(cores) or None
    return CPUInfo(model, logical, physical, usable)


def detect_memory(run: ProbeFn, system: str) -> MemoryInfo:
    if system == "Darwin":
        total = _sysctl(run, "hw.memsize")
        vm = run(["vm_stat"])
        return MemoryInfo(
            int(total) if total and total.isdigit() else None,
            parse_vm_stat(vm.stdout) if vm.ok else None,
        )
    total, avail = parse_meminfo(_read("/proc/meminfo") or "")
    return MemoryInfo(total, avail, cgroup_memory_limit())


def detect_storage(paths: Sequence[Path]) -> list[StorageInfo]:
    out: list[StorageInfo] = []
    seen: set[int] = set()
    for p in paths:
        probe_path = Path(p)
        while not probe_path.exists() and probe_path != probe_path.parent:
            probe_path = probe_path.parent
        try:
            dev = probe_path.stat().st_dev
            if dev in seen:
                continue
            seen.add(dev)
            u = shutil.disk_usage(probe_path)
            out.append(StorageInfo(str(p), u.total, u.free))
        except OSError:
            continue
    return out


def detect(storage_paths: Sequence[Path] = (), run: ProbeFn = probe) -> HardwareInfo:
    override = os.environ.get("SETUP_WORK_HARDWARE_FILE")
    if override:
        data = json.loads(Path(override).expanduser().read_text(encoding="utf-8"))
        hw = HardwareInfo.from_dict(data)
        hw.source = f"file:{override}"
        return hw

    osinfo = detect_os()
    notes: list[str] = []
    cpu = detect_cpu(run, osinfo.system)
    mem = detect_memory(run, osinfo.system)
    if mem.cgroup_limit_bytes and mem.total_bytes and mem.cgroup_limit_bytes < mem.total_bytes:
        notes.append("лимит памяти контейнера меньше RAM хоста; используется лимит")
    hw = HardwareInfo(os=osinfo, cpu=cpu, memory=mem)

    if osinfo.system == "Darwin" and osinfo.arch == "arm64":
        chip = cpu.model or "Apple Silicon"
        wired = _sysctl(run, "iogpu.wired_limit_mb")
        ws, src = metal_working_set(mem.total_bytes, wired)
        hw.gpus = [
            GPU(
                index=0,
                vendor="apple",
                name=f"{chip} GPU (Metal)",
                memory_total_bytes=ws,
                memory_free_bytes=None,
                unified_memory=True,
            )
        ]
        hw.accelerator = "apple"
        hw.metal_working_set_bytes = ws
        hw.metal_working_set_source = src
        hw.precisions = ["fp32", "fp16"] + ([] if "M1" in chip else ["bf16"])
        notes.append("GPU и CPU используют общую RAM; бюджет GPU ограничен рабочим объёмом Metal")
    else:
        gpus, driver, cuda, nv_notes = detect_nvidia(run)
        notes.extend(nv_notes)
        if gpus:
            hw.gpus = gpus
            hw.accelerator = "nvidia"
            hw.nvidia_driver = driver
            hw.cuda_driver_version = cuda
            hw.cuda_toolkit_version = detect_cuda_toolkit(run)
            hw.precisions = nvidia_precisions(hw.min_compute_capability)
            if not hw.homogeneous_gpus:
                notes.append(
                    "GPU различаются; тензорный параллелизм "
                    "ограничен памятью наименьшего устройства"
                )
        elif shutil.which("rocm-smi"):
            hw.accelerator = "amd"
            notes.append(
                "обнаружен AMD ROCm; setup-work пока использует для такой машины режим CPU"
            )
            hw.precisions = ["fp32"]
        else:
            hw.precisions = ["fp32", "fp16"]
            notes.append("поддерживаемый GPU не найден; доступен инференс на CPU")
    hw.storage = detect_storage(storage_paths or [Path.home()])
    hw.notes = notes
    return hw
