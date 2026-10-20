from __future__ import annotations

import json

import pytest

from conftest import HW_DIR
from setup_work.cli import main


def run(capsys, *args):
    code = main(list(args))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_version(capsys):
    code, out, _ = run(capsys, "version", "--json")
    assert code == 0 and json.loads(out)["version"]


def test_config_validate(capsys):
    code, out, _ = run(capsys, "config", "validate")
    assert code == 0 and "конфигурация корректна" in out


def test_hardware_json_from_file(capsys, monkeypatch):
    monkeypatch.setenv("SETUP_WORK_HARDWARE_FILE", str(HW_DIR / "2x-v100-32gb.json"))
    code, out, _ = run(capsys, "hardware", "--json")
    data = json.loads(out)
    assert code == 0 and data["summary"]["gpu_count"] == 2


def test_recommend_json_is_pure(capsys, monkeypatch):
    monkeypatch.setenv("SETUP_WORK_HARDWARE_FILE", str(HW_DIR / "1x-rtx4090.json"))
    code, out, _ = run(capsys, "--json", "recommend", "--task", "coding", "--context", "16384")
    data = json.loads(out)  # stdout содержит только JSON
    assert code == 0 and data["recommendations"]
    assert data["request"]["context"] == 16384


def test_recommend_via_env_like_makefile(capsys, monkeypatch):
    monkeypatch.setenv("SETUP_WORK_HARDWARE_FILE", str(HW_DIR / "cpu-only-32gb.json"))
    monkeypatch.setenv("SETUP_WORK_TASK", "agents")
    monkeypatch.setenv("SETUP_WORK_ENGINE", "llama.cpp")
    monkeypatch.setenv(
        "SETUP_WORK_CONTEXT", ""
    )  # пустое значение означает отсутствие параметра, как при экспорте из make
    code, out, _ = run(capsys, "--json", "recommend")
    data = json.loads(out)
    assert data["request"]["task"] == "agents" and data["request"]["engine"] == "llama.cpp"
    assert data["request"]["context"] == 8192


@pytest.mark.parametrize(
    "args,code",
    [
        (["recommend", "--task", "poetry"], 2),
        (["recommend", "--context", "abc"], 2),
        (["model", "info", "nope"], 6),
        (["serve", "qwen3-8b:q4_k_m"], 6),  # не загружено → не скачивать неявно
        (["harness", "install", "nope"], 6),
        (["bootstrap", "--tool", "git"], 2),
        (["no-such-command"], 2),
    ],
)
def test_exit_codes(capsys, monkeypatch, args, code):
    monkeypatch.setenv("SETUP_WORK_HARDWARE_FILE", str(HW_DIR / "1x-rtx4090.json"))
    got, _, err = run(capsys, *args)
    assert got == code, err


def test_errors_have_hints(capsys, monkeypatch):
    monkeypatch.setenv("SETUP_WORK_HARDWARE_FILE", str(HW_DIR / "1x-rtx4090.json"))
    code, _, err = run(capsys, "serve", "qwen3-8b:q4_k_m")
    assert "что сделать:" in err and "make model-pull MODEL=qwen3-8b:q4_k_m" in err


def test_serve_plan_json(capsys, monkeypatch):
    monkeypatch.setenv("SETUP_WORK_HARDWARE_FILE", str(HW_DIR / "4x-a100-80gb.json"))
    code, out, _ = run(
        capsys,
        "--json",
        "serve",
        "qwen3-32b:bf16",
        "--plan",
        "--context",
        "32768",
        "--concurrency",
        "4",
    )
    data = json.loads(out)
    assert code == 0
    assert "--tensor-parallel-size" in data["argv"]
    assert data["launch"]["host"] == "127.0.0.1"


def test_unsupported_serve_exit_code(capsys, monkeypatch):
    monkeypatch.setenv("SETUP_WORK_HARDWARE_FILE", str(HW_DIR / "m3-max-64gb.json"))
    code, _, err = run(capsys, "serve", "qwen3-8b:bf16", "--plan")
    assert code == 4 and "Linux" in err


def test_bootstrap_dry_run_json(capsys):
    code, out, _ = run(capsys, "--dry-run", "--json", "bootstrap", "--tool", "node")
    data = json.loads(out)
    assert code == 0 and [d["tool"] for d in data] == ["node"]


def test_harness_install_dry_run(capsys, monkeypatch):
    from setup_work import harness as hz

    monkeypatch.setattr(hz, "_check_requirements", lambda cfg, spec: [])
    code, out, _ = run(capsys, "--dry-run", "--json", "harness", "install", "codex")
    data = json.loads(out)
    assert code == 0 and data["version"] == "0.159.1"
    assert any("@openai/codex@0.159.1" in s["action"] for s in data["steps"])


def test_models_listing(capsys):
    code, out, _ = run(capsys, "--json", "models", "--task", "embedding")
    data = json.loads(out)
    assert code == 0 and [m["id"] for m in data] == ["qwen3-embedding-0.6b"]


def test_serve_status_empty(capsys):
    code, out, _ = run(capsys, "server", "status")
    assert code == 0 and "серверы не зарегистрированы" in out


def test_destructive_without_yes_aborts(capsys, tmp_path):
    from setup_work.config import Config

    cfg = Config()
    (cfg.tools_dir / "downloads").mkdir(parents=True)
    code, _, err = run(capsys, "clean")
    assert code == 5 and "требуется подтверждение" in err
    assert (cfg.tools_dir / "downloads").exists()
    code, _, _ = run(capsys, "--yes", "clean")
    assert code == 0 and not (cfg.tools_dir / "downloads").exists()


def test_setup_make_flags_and_json(capsys, monkeypatch):
    from setup_work import pipeline

    monkeypatch.setenv("SETUP_WORK_WITH_AGENTS", "0")
    monkeypatch.setenv("SETUP_WORK_LOCAL_MODEL", "0")
    monkeypatch.setenv("SETUP_WORK_WITH_UV", "1")
    monkeypatch.setenv("SETUP_WORK_HARDWARE_FILE", str(HW_DIR / "cpu-only-32gb.json"))
    seen = {}

    def fake_run(cfg, hardware, runner, **kwargs):
        seen.update(kwargs)
        return {"backend": "gigachat", "stages": {}}

    monkeypatch.setattr(pipeline, "run", fake_run)
    code, out, _ = run(capsys, "--json", "setup")
    assert code == 0 and json.loads(out)["backend"] == "gigachat"
    assert seen["agents"] is False and seen["local"] is False and seen["uv"] is True
