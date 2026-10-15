from __future__ import annotations

import json

import pytest

from setup_work import harness as hz
from setup_work.config import Config
from setup_work.diagnostics import ToolStatus
from setup_work.errors import MissingDependencyError, NotFoundError, UsageError
from setup_work.runner import Runner
from setup_work.steps import execute


@pytest.fixture
def no_req(monkeypatch):
    monkeypatch.setattr(hz, "_check_requirements", lambda cfg, spec: [])


def _write_user_harness(tmp_path, body: str) -> Config:
    d = tmp_path / "userconf"
    d.mkdir(exist_ok=True)
    (d / "harnesses.yaml").write_text(body)
    return Config()


def test_npm_install_plan_is_pinned_and_local(cfg, no_req):
    plan = hz.plan_install(cfg, "qwen-code")
    cmd = plan.steps[0].argv
    assert cmd[1:4] == ["install", "--prefix", str(cfg.harness_dir / "qwen-code")]
    assert cmd[-1] == "@qwen-code/qwen-code@0.24.7"
    assert "-g" not in cmd
    link = plan.steps[-1]
    assert link.kind == "link" and link.dest == cfg.bin_dir / "qwen"
    assert link.src == cfg.harness_dir / "qwen-code" / "node_modules" / ".bin" / "qwen"


def test_pypi_install_uses_dedicated_venv(cfg, no_req):
    plan = hz.plan_install(cfg, "aider", version="0.86.1")
    venv_step, install_step = plan.steps[0], plan.steps[1]
    assert venv_step.argv[1:4] == ["venv", "--python", "3.12"]
    assert install_step.argv[-1] == "aider-chat==0.86.1"


def test_version_override_validated(cfg, no_req):
    with pytest.raises(Exception, match="invalid version"):
        hz.plan_install(cfg, "aider", version="1.0; rm -rf ~")
    with pytest.raises(UsageError, match="REF applies"):
        hz.plan_install(cfg, "aider", ref="main")


def test_git_harness_plan(tmp_path, no_req):
    cfg = _write_user_harness(
        tmp_path,
        """
harnesses:
  custom-agent:
    source: git
    repository: https://github.com/example/agent.git
    ref: v1.2.0
    install: {method: uv}
    binaries: [agent]
    environment:
      required: [OPENAI_BASE_URL, OPENAI_API_KEY]
""",
    )
    plan = hz.plan_install(cfg, "custom-agent")
    clone = plan.steps[0]
    assert clone.argv[1:4] == ["clone", "--no-checkout", "--filter=blob:none"]
    assert clone.argv[4] == "--"  # URL can never be parsed as an option
    assert clone.env["GIT_TERMINAL_PROMPT"] == "0"
    assert plan.needs_confirmation
    assert any("build code" in w for w in plan.warnings)
    assert plan.steps[-1].src.name == "agent" and ".venv" in str(plan.steps[-1].src)


def test_build_command_requires_opt_in(tmp_path, no_req):
    cfg = _write_user_harness(
        tmp_path,
        """
harnesses:
  builder:
    source: git
    repository: https://github.com/example/agent.git
    ref: 0123abc
    install: {method: npm, build: [npm, run, build]}
""",
    )
    with pytest.raises(UsageError, match="allow-build"):
        hz.plan_install(cfg, "builder")
    plan = hz.plan_install(cfg, "builder", allow_build=True)
    assert any(s.argv == ["npm", "run", "build"] for s in plan.steps if s.argv)


def test_missing_runtime_reported(cfg, monkeypatch):
    monkeypatch.setattr(hz, "detect_tool", lambda name, cfg=None: ToolStatus(name, False))
    with pytest.raises(MissingDependencyError, match="bootstrap-node"):
        hz.plan_install(cfg, "claude-code")


def test_unknown_harness(cfg):
    with pytest.raises(NotFoundError, match="custom-harness.yaml"):
        hz.plan_install(cfg, "does-not-exist")


def test_install_remove_cycle_with_local_source(tmp_path, monkeypatch, no_req):
    """End-to-end with a `local` + `none` recipe: link, manifest, doctor, remove."""
    src = tmp_path / "myagent"
    src.mkdir()
    exe = src / "myagent"
    exe.write_text("#!/bin/sh\necho myagent 1.0\n")
    exe.chmod(0o755)
    cfg = _write_user_harness(
        tmp_path,
        f"""
harnesses:
  myagent:
    source: local
    path: {src}
    install: {{method: none}}
    binaries: [myagent]
    verify: [myagent, --version]
    environment:
      required: [MYAGENT_TOKEN]
""",
    )
    runner = Runner(assume_yes=True, interactive=False)
    plan = hz.plan_install(cfg, "myagent")
    execute(plan.steps, runner, cfg.data_dir)
    hz.write_manifest(cfg, "myagent", plan.spec, runner)
    link = cfg.bin_dir / "myagent"
    assert link.is_symlink() and link.resolve() == exe.resolve()
    manifest = json.loads((cfg.harness_dir / "myagent" / hz.MANIFEST).read_text())
    assert manifest["binaries"]["myagent"] == str(link)

    monkeypatch.setenv("PATH", f"{cfg.bin_dir}:{__import__('os').environ['PATH']}")
    report = hz.doctor(cfg, "myagent")
    checks = {c: ok for c, ok, _ in report.checks}
    assert checks["verify command"] is True
    assert checks["env MYAGENT_TOKEN"] is False  # required variable missing
    assert not report.ok
    monkeypatch.setenv("MYAGENT_TOKEN", "x")
    assert hz.doctor(cfg, "myagent").ok
    assert all(d != "x" for _, _, d in hz.doctor(cfg, "myagent").checks)  # value never shown

    steps = hz.plan_remove(cfg, "myagent")
    assert steps[-1].destructive
    with pytest.raises(Exception, match="confirmation required"):
        execute(steps, Runner(interactive=False), cfg.data_dir)
    execute(steps, runner, cfg.data_dir)
    assert not link.exists() and exe.exists()  # local source itself untouched


def test_list_reports_pinning(cfg):
    rows = {r["name"]: r for r in hz.list_harnesses(cfg)}
    assert rows["claude-code"]["pinned"]
    assert rows["claude-code"]["installed"] is None
