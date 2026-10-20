from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from setup_work import harness, pipeline
from setup_work.config import ArtifactRef
from setup_work.errors import SetupWorkError
from setup_work.runner import Runner


def test_pipeline_order_and_cpu_fallback(cfg, hw_cpu, monkeypatch):
    events = []
    monkeypatch.setattr(pipeline, "infrastructure", lambda *a: events.append("infra") or {})
    cfg.defaults.setup.harnesses = ["agent-b", "agent-a"]
    for name in cfg.defaults.setup.harnesses:
        cfg.raw["harnesses.yaml"]["harnesses"][name] = {
            "source": "git",
            "repository": f"https://git.corp/{name}.git",
            "ref": "main",
            "install": {"method": "setup"},
        }
    monkeypatch.setattr(
        harness,
        "setup_repository",
        lambda cfg, name, runner, **kw: events.append(name) or {"name": name},
    )

    def hardware():
        events.append("hardware")
        return hw_cpu

    state = pipeline.run(cfg, hardware, Runner(interactive=False), agents=True, local=True)
    assert events == ["infra", "agent-b", "agent-a", "hardware"]
    assert state["backend"] == "gigachat" and state["stages"]["local"]["status"] == "no-gpu"
    assert json.loads((cfg.state_dir / "setup-state.json").read_text()) == state


def test_pipeline_stops_on_failure_and_does_not_record_success(cfg, hw_cpu, monkeypatch):
    monkeypatch.setattr(
        pipeline, "infrastructure", lambda *a: (_ for _ in ()).throw(SetupWorkError("infra failed"))
    )
    with pytest.raises(SetupWorkError, match="infra failed"):
        pipeline.run(cfg, lambda: pytest.fail("hardware after failure"), Runner(), agents=False)
    assert not (cfg.state_dir / "setup-state.json").exists()


def test_dry_run_does_not_answer_questions_or_write_state(cfg, hw_cpu, monkeypatch):
    monkeypatch.setattr(pipeline, "infrastructure", lambda *a: {})
    runner = Runner(dry_run=True, interactive=False)
    monkeypatch.setattr(runner, "confirm", lambda *a, **kw: pytest.fail("prompt during dry-run"))
    pipeline.run(cfg, lambda: hw_cpu, runner)
    assert not cfg.state_dir.exists()


def test_gpu_local_is_additional_to_gigachat(cfg, hw_4090, monkeypatch):
    monkeypatch.setattr(pipeline, "infrastructure", lambda *a: {})
    monkeypatch.setattr(pipeline, "local_model", lambda *a: {"status": "running"})
    state = pipeline.run(cfg, lambda: hw_4090, Runner(), agents=False, local=True)
    assert state["backend"] == "gigachat" and state["stages"]["local"]["status"] == "running"


def test_unconfigured_jars_skip_and_configured_jar_reuses_install(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.bootstrap, "plan_tools", lambda *a, **kw: [])
    # Pin probes to avoid any runtime install/network.
    from setup_work.diagnostics import ToolStatus

    monkeypatch.setattr(
        pipeline,
        "detect_tool",
        lambda name, cfg: ToolStatus(name, True, cfg.defaults.tools[name].version),
    )
    jar = tmp_path / "driver.jar"
    jar.write_bytes(b"jar contents")
    cfg.corporate.jars = {
        "trino": ArtifactRef(uri=str(jar), format="file", version="1"),
        "iceberg": None,
    }
    runner = Runner(interactive=False)
    monkeypatch.setenv("PATH", os.environ["PATH"])
    result = pipeline.infrastructure(cfg, runner)
    assert result["jars"] == {"trino": "installed", "iceberg": "unconfigured"}
    installed = cfg.tools_dir / "jars/trino/driver.jar"
    assert installed.read_bytes() == b"jar contents"
    jar.unlink()
    pipeline.infrastructure(cfg, runner)  # completed artifact does not need its source again
    assert installed.exists()


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=True,
        env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull},
    ).strip()


@pytest.fixture
def internal_repo(cfg, tmp_path, monkeypatch):
    repo = tmp_path / "remote"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    (repo / "setup.sh").write_text('printf "%s\\n" "$SETUP_WORK_BACKEND" >> runs.log\n')
    _git(repo, "add", "setup.sh")
    _git(repo, "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "v1")
    url = "https://bitbucket.corp/scm/test/agent.git"
    cfg.raw["harnesses.yaml"]["harnesses"]["agent"] = {
        "source": "git",
        "repository": url,
        "ref": "main",
        "install": {"method": "setup"},
    }
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", f"url.{repo.as_uri()}.insteadOf")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", url)
    return repo


def _commit_setup(repo, text):
    (repo / "setup.sh").write_text(text)
    _git(repo, "add", "setup.sh")
    _git(
        repo, "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "update"
    )
    return _git(repo, "rev-parse", "HEAD")


def test_setup_repo_install_skip_decline_and_update(cfg, internal_repo, monkeypatch):
    runner = Runner(interactive=False)
    first = harness.setup_repository(cfg, "agent", runner)
    src = cfg.harness_dir / "agent/src"
    assert first["status"] == "installed"
    assert (src / "runs.log").read_text() == "gigachat\n"
    assert harness.setup_repository(cfg, "agent", runner)["status"] == "current"
    assert (src / "runs.log").read_text() == "gigachat\n"
    newer = _commit_setup(internal_repo, 'printf "%s:v2\\n" "$SETUP_WORK_BACKEND" >> runs.log\n')
    monkeypatch.setattr(runner, "confirm", lambda *a, **kw: False)
    assert harness.setup_repository(cfg, "agent", runner)["status"] == "declined"
    assert _git(src, "rev-parse", "HEAD") == first["commit"]
    monkeypatch.setattr(runner, "confirm", lambda *a, **kw: True)
    assert harness.setup_repository(cfg, "agent", runner)["commit"] == newer
    assert (src / "runs.log").read_text() == "gigachat\ngigachat:v2\n"


def test_setup_failure_is_retryable_and_keeps_last_success(cfg, internal_repo, monkeypatch):
    runner = Runner(assume_yes=True, interactive=False)
    first = harness.setup_repository(cfg, "agent", runner)
    _commit_setup(internal_repo, "echo failure >&2\nexit 8\n")
    for _ in range(2):
        with pytest.raises(SetupWorkError):
            harness.setup_repository(cfg, "agent", runner)
        assert harness.read_manifest(cfg, "agent")["commit"] == first["commit"]


def test_setup_dry_run_never_contacts_remote(cfg, internal_repo, monkeypatch):
    monkeypatch.setattr(
        harness, "remote_revision", lambda *a: pytest.fail("network during dry-run")
    )
    result = harness.setup_repository(cfg, "agent", Runner(dry_run=True, interactive=False))
    assert result["status"] == "planned"
    assert not cfg.harness_dir.exists()


def test_new_container_reuses_harness_and_only_starts_api(cfg, internal_repo):
    runner = Runner(interactive=False)
    harness.setup_repository(cfg, "agent", runner)
    cfg.runtime_dir.mkdir(parents=True)
    (cfg.runtime_dir / "model.gguf").write_bytes(b"weights")
    shutil.rmtree(cfg.runtime_dir)  # файловая система старого контейнера исчезла
    spec = harness.get_spec(cfg, "agent")
    spec.install.start = [
        "bash",
        "-c",
        'printf "%s\\n" "$SETUP_WORK_BACKEND" >> "$SETUP_WORK_HARNESS_HOME/starts.log"',
    ]
    result = harness.setup_repository(cfg, "agent", runner, check_updates=False)
    assert result["status"] == "kept" and result["runtime"] == "started"
    assert (cfg.harness_dir / "agent/src/runs.log").read_text() == "gigachat\n"
    assert (cfg.harness_dir / "agent/starts.log").read_text() == "gigachat\n"
    assert (cfg.harness_dir / "agent/config").is_dir()
    assert not cfg.runtime_dir.exists()


def test_local_dry_run_with_explicit_variant_never_fetches(cfg, hw_4090, monkeypatch):
    from setup_work.config import EngineArtifact

    cfg.defaults.setup.model = "qwen3-8b:q4_k_m"
    cfg.corporate.models["qwen3-8b:q4_k_m"] = ArtifactRef(uri="hdfs:///weights/model.gguf")
    cfg.corporate.engines["llama.cpp"] = EngineArtifact(uri="hdfs:///engines/llama.zip")
    monkeypatch.setattr(
        pipeline.artifacts, "inspect", lambda *a: pytest.fail("HDFS during dry-run")
    )
    result = pipeline.local_model(cfg, hw_4090, Runner(dry_run=True, interactive=False))
    assert result["model"] == "qwen3-8b:q4_k_m" and result["status"] == "planned"
    assert not cfg.runtime_dir.exists()


def test_setup_authenticates_before_fetch_even_for_pinned_commit(cfg, internal_repo, monkeypatch):
    spec = harness.get_spec(cfg, "agent")
    spec.ref = _git(internal_repo, "rev-parse", "HEAD")
    calls = []
    monkeypatch.setattr(harness, "prepare_git_auth", lambda *a: calls.append("login"))
    result = harness.setup_repository(cfg, "agent", Runner(interactive=False))
    assert result["status"] == "installed" and calls == ["login"]
