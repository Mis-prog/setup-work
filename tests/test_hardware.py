from __future__ import annotations

import json

import pytest

from setup_work import hardware as hwmod
from setup_work.hardware import (
    GiB,
    HardwareInfo,
    cc_from_name,
    metal_working_set,
    nvidia_precisions,
    parse_meminfo,
    parse_nvidia_smi_csv,
    parse_vm_stat,
)
from setup_work.runner import Result

NVSMI_CC = """\
0, NVIDIA A100-SXM4-80GB, GPU-aaa, 81920, 80000, 00000000:07:00.0, 550.54.15, 8.0
1, NVIDIA A100-SXM4-80GB, GPU-bbb, 81920, 1024, 00000000:0F:00.0, 550.54.15, 8.0
"""
NVSMI_OLD = "0, Tesla V100-SXM2-32GB, GPU-ccc, 32768, 32000, 00000000:00:1E.0, 470.82.01\n"
NVSMI_HEADER = "| NVIDIA-SMI 550.54.15   Driver Version: 550.54.15   CUDA Version: 12.4     |"


def test_parse_nvidia_smi_with_compute_cap():
    gpus = parse_nvidia_smi_csv(NVSMI_CC, with_cc=True)
    assert len(gpus) == 2
    assert gpus[0].memory_total_bytes == 81920 * 1024**2
    assert gpus[1].memory_free_bytes == 1024 * 1024**2  # busy GPU: per-device free tracked
    assert gpus[0].compute_capability == "8.0"
    assert gpus[0].compute_capability_source == "nvidia-smi"


def test_parse_nvidia_smi_old_driver_falls_back_to_name():
    gpus = parse_nvidia_smi_csv(NVSMI_OLD, with_cc=False)
    assert gpus[0].compute_capability == "7.0"
    assert gpus[0].compute_capability_source == "name-lookup"


@pytest.mark.parametrize(
    "name,cc",
    [
        ("NVIDIA H100 80GB HBM3", "9.0"),
        ("NVIDIA GeForce RTX 4090", "8.9"),
        ("NVIDIA A10G", "8.6"),
        ("Tesla T4", "7.5"),
        ("NVIDIA L4", "8.9"),
        ("Mystery GPU", None),
    ],
)
def test_cc_from_name(name, cc):
    assert cc_from_name(name) == cc


def test_precisions_by_compute_capability():
    assert "bf16" not in nvidia_precisions(7.0)
    assert "bf16" in nvidia_precisions(8.0) and "fp8" not in nvidia_precisions(8.0)
    assert "fp8" in nvidia_precisions(8.9)
    assert "fp4" in nvidia_precisions(10.0)


def fake_probe(table):
    """Return canned output for the longest matching argv prefix."""

    def run(argv):
        for k, out in sorted(table.items(), key=lambda kv: -len(kv[0])):
            if list(k) == list(argv[: len(k)]):
                if out is None:
                    return Result(list(argv), 1, "", "error")
                return Result(list(argv), 0, out, "")
        return Result(list(argv), 127, "", "not found")

    return run


def test_detect_nvidia_with_mock(monkeypatch):
    run = fake_probe(
        {
            (
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,memory.free,pci.bus_id,"
                "driver_version,compute_cap",
            ): NVSMI_CC,
            ("nvidia-smi",): NVSMI_HEADER,
        }
    )
    gpus, driver, cuda, notes = hwmod.detect_nvidia(run)
    assert len(gpus) == 2 and driver == "550.54.15" and cuda == "12.4"
    assert notes == []


def test_detect_nvidia_retries_without_compute_cap():
    calls = []

    def run(argv):
        calls.append(argv)
        if argv[0] == "nvidia-smi" and "compute_cap" in " ".join(argv):
            return Result(argv, 2, "", 'Field "compute_cap" is not a valid field')
        if argv[0] == "nvidia-smi" and len(argv) > 1:
            return Result(argv, 0, NVSMI_OLD, "")
        return Result(argv, 0, "CUDA Version: 11.4", "")

    gpus, driver, cuda, notes = hwmod.detect_nvidia(run)
    assert gpus[0].compute_capability == "7.0"
    assert any("inferred" in n for n in notes)


def test_detect_linux_cpu_only(monkeypatch):
    monkeypatch.setattr(hwmod.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hwmod.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(hwmod.shutil, "which", lambda name: None)
    files = {
        "/proc/meminfo": "MemTotal: 32000000 kB\nMemAvailable: 16000000 kB\n",
        "/proc/cpuinfo": "model name : Test CPU\nphysical id : 0\ncore id : 0\n\n"
        "model name : Test CPU\nphysical id : 0\ncore id : 1\n",
        "/etc/os-release": 'PRETTY_NAME="Ubuntu 24.04 LTS"\n',
    }
    monkeypatch.setattr(hwmod, "_read", lambda p: files.get(p))
    hw = hwmod.detect(run=lambda argv: Result(list(argv), 127, "", ""))
    assert hw.accelerator == "cpu" and hw.gpus == []
    assert hw.memory.total_bytes == 32000000 * 1024
    assert hw.cpu.physical_cores == 2
    assert hw.os.distro == "Ubuntu 24.04 LTS"
    assert any("CPU-only" in n for n in hw.notes)


def test_cgroup_limit_used(monkeypatch):
    monkeypatch.setattr(
        hwmod, "_read", lambda p: "8589934592\n" if p.endswith("memory.max") else None
    )
    assert hwmod.cgroup_memory_limit() == 8 * GiB
    monkeypatch.setattr(hwmod, "_read", lambda p: "max\n")
    assert hwmod.cgroup_memory_limit() is None


def test_parse_meminfo_and_vm_stat():
    total, avail = parse_meminfo("MemTotal: 1024 kB\nMemAvailable: 512 kB\n")
    assert (total, avail) == (1024 * 1024, 512 * 1024)
    vm = (
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 100.\n"
        "Pages inactive: 50.\nPages speculative: 10.\n"
    )
    assert parse_vm_stat(vm) == 160 * 16384


def test_metal_working_set():
    ws, src = metal_working_set(24 * GiB, "0")
    assert ws == int(24 * GiB * 0.67) and "estimated" in src
    ws, _ = metal_working_set(64 * GiB, None)
    assert ws == int(64 * GiB * 0.75)
    ws, src = metal_working_set(64 * GiB, "57344")
    assert ws == 57344 * 1024**2 and src == "iogpu.wired_limit_mb"


def test_aggregate_vram_is_not_a_pool(hw_v100):
    assert hw_v100.total_vram_bytes == 64 * GiB
    d = hw_v100.to_dict()
    assert "not a shared memory pool" in d["summary"]["total_vram_note"]
    assert d["summary"]["min_compute_capability"] == 7.0


def test_roundtrip_and_override_file(tmp_path, monkeypatch, hw_4090):
    p = tmp_path / "hw.json"
    p.write_text(json.dumps(hw_4090.to_dict()))
    monkeypatch.setenv("SETUP_WORK_HARDWARE_FILE", str(p))
    hw = hwmod.detect()
    assert isinstance(hw, HardwareInfo)
    assert hw.gpus[0].name == "NVIDIA GeForce RTX 4090"
    assert hw.source.startswith("file:")


def test_heterogeneous_gpus(hw_mixed):
    assert not hw_mixed.homogeneous_gpus
    assert hw_mixed.min_compute_capability == 7.5
