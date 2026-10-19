from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

from setup_work.config import Config
from setup_work.runner import Runner

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("setting", ["SETUP_WORK_PYPI_INDEX", "SETUP_WORK_NPM_REGISTRY"])
def test_registry_placeholder_explains_what_to_fill(tmp_path, setting):
    config = tmp_path / "config"
    config.mkdir()
    settings = config / "bootstrap.env"
    settings.write_text(f"{setting}=https://CHANGE-ME-repo-host/repository/simple\n")
    result = subprocess.run(
        ["sh", str(ROOT / "scripts/bootstrap.sh"), "--registries-only", "--dry-run"],
        env={
            "HOME": str(tmp_path / "home"),
            "PATH": os.defpath,
            "SETUP_WORK_USER_CONFIG_DIR": str(config),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert setting in result.stderr and str(settings) in result.stderr
    assert "замените CHANGE-ME" in result.stderr
    assert not (tmp_path / "home").exists()


def test_model_validation_names_file_field_and_required_action(tmp_path):
    config = tmp_path / "config"
    config.mkdir()
    (config / "defaults.yaml").write_text((ROOT / "config/defaults.yaml").read_text())
    model = {
        "family": "test",
        "display_name": "Тестовая модель",
        "license": "test",
        "architecture": "dense",
        "parameters_b": 1,
        "workloads": ["general"],
        "context_length": 512,
        "attention": {
            "num_layers": "CHANGE-ME-num-hidden-layers",
            "num_attention_heads": 1,
            "num_kv_heads": 1,
            "head_dim": 1,
        },
        "variants": [
            {
                "id": "full",
                "format": "safetensors",
                "dtype": "bf16",
                "source": "/test/model.zip",
                "size_gb": "не число",
                "engines": ["vllm"],
            }
        ],
    }
    settings = config / "models.yaml"
    settings.write_text(yaml.safe_dump({"models": {"sample": model}}, allow_unicode=True))
    problems = Config(builtin_dir=config, user_dir=tmp_path / "user").validate_all()
    detail = "\n".join(problems)
    assert str(settings) in detail
    assert "models.sample.attention.num_layers" in detail
    assert "замените CHANGE-ME" in detail
    assert "models.sample.variants.0.size_gb: требуется число" in detail
    assert "Input should" not in detail


def test_confirmation_accepts_russian_answer(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda: "да")
    assert Runner(interactive=True).confirm("Продолжить?")
