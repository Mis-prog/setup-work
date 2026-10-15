from __future__ import annotations

import json
from pathlib import Path

import pytest

from setup_work.config import Config
from setup_work.hardware import HardwareInfo

ROOT = Path(__file__).resolve().parents[1]
HW_DIR = ROOT / "examples" / "hardware"


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Every test gets private data/state/bin/config dirs; nothing touches $HOME."""
    for var in ("SETUP_WORK_HARDWARE_FILE", "SETUP_WORK_API_KEY", "SETUP_WORK_MODEL_CACHE"):
        monkeypatch.delenv(var, raising=False)
    for k in list(__import__("os").environ):
        if k.startswith("SETUP_WORK_") and k not in ("SETUP_WORK_CONFIG_DIR",):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("SETUP_WORK_BIN_DIR", str(tmp_path / "bin"))
    monkeypatch.setenv("SETUP_WORK_USER_CONFIG_DIR", str(tmp_path / "userconf"))
    monkeypatch.setenv("SETUP_WORK_CONFIG_DIR", str(ROOT / "config"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf"))
    return tmp_path


@pytest.fixture
def cfg() -> Config:
    return Config()


def load_hw(name: str) -> HardwareInfo:
    return HardwareInfo.from_dict(json.loads((HW_DIR / f"{name}.json").read_text()))


@pytest.fixture
def hw_v100() -> HardwareInfo:
    return load_hw("2x-v100-32gb")


@pytest.fixture
def hw_a100x4() -> HardwareInfo:
    return load_hw("4x-a100-80gb")


@pytest.fixture
def hw_4090() -> HardwareInfo:
    return load_hw("1x-rtx4090")


@pytest.fixture
def hw_cpu() -> HardwareInfo:
    return load_hw("cpu-only-32gb")


@pytest.fixture
def hw_mac() -> HardwareInfo:
    return load_hw("m3-max-64gb")


@pytest.fixture
def hw_mixed() -> HardwareInfo:
    return load_hw("mixed-a100-t4")
