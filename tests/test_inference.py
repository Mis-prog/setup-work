from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from setup_work.engines import LaunchConfig
from setup_work.errors import (
    MissingDependencyError,
    NotFoundError,
    UnsupportedError,
    UsageError,
)
from setup_work.inference import (
    LaunchPlan,
    ServeOptions,
    ServerManager,
    ServerState,
    find_free_port,
    plan_launch,
    port_in_use,
)
from setup_work.runner import Runner

FAKE = Path(__file__).with_name("fake_server.py")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_plan_catalog_ref_without_cache_check(cfg, hw_4090):
    plan = plan_launch(
        cfg, hw_4090, "qwen3-8b:q4_k_m", ServeOptions(context=16384), check_cache=False
    )
    assert plan.launch.engine == "llama.cpp"
    assert plan.launch.gpu_layers == -1
    assert plan.launch.host == "127.0.0.1"
    assert "-c" in plan.argv and plan.argv[plan.argv.index("-c") + 1] == "16384"
    assert not plan.public
    assert any("оценка памяти" in r for r in plan.rationale)


def test_plan_auto_selects_variant_for_engine(cfg, hw_a100x4):
    plan = plan_launch(
        cfg, hw_a100x4, "qwen3-32b", ServeOptions(engine="vllm", context=32768), check_cache=False
    )
    assert plan.launch.engine == "vllm"
    assert "--served-model-name" in plan.argv
    assert any("автоматически выбран" in r for r in plan.rationale)


def test_plan_rejects_engine_variant_mismatch(cfg, hw_a100x4):
    with pytest.raises(UsageError, match="подготовлен для llama.cpp"):
        plan_launch(
            cfg, hw_a100x4, "qwen3-8b:q4_k_m", ServeOptions(engine="vllm"), check_cache=False
        )


def test_plan_rejects_unsupported_and_oversized(cfg, hw_v100, hw_cpu):
    with pytest.raises(UnsupportedError, match="7.5"):
        plan_launch(cfg, hw_v100, "qwen3-8b:awq", ServeOptions(), check_cache=False)
    with pytest.raises(UnsupportedError, match="недостаточно памяти"):
        plan_launch(cfg, hw_cpu, "qwen3-32b:q8_0", ServeOptions(context=32768), check_cache=False)
    with pytest.raises(UnsupportedError):
        plan_launch(cfg, hw_cpu, "qwen3-8b:bf16", ServeOptions(), check_cache=False)


def test_force_overrides_fit_check(cfg, hw_cpu):
    plan = plan_launch(
        cfg, hw_cpu, "qwen3-32b:q8_0", ServeOptions(context=32768, force=True), check_cache=False
    )
    assert any(w.startswith("ПРИНУДИТЕЛЬНО") for w in plan.warnings)


def test_public_bind_is_flagged(cfg, hw_4090):
    plan = plan_launch(
        cfg, hw_4090, "qwen3-8b:q4_k_m", ServeOptions(host="0.0.0.0"), check_cache=False
    )
    assert plan.public
    assert any("без аутентификации" in w for w in plan.warnings)


def test_serve_never_downloads_implicitly(cfg, hw_4090):
    with pytest.raises(NotFoundError, match="не скачивает модели"):
        plan_launch(cfg, hw_4090, "qwen3-8b:q4_k_m", ServeOptions())


def test_local_gguf_path(cfg, hw_mac, tmp_path):
    f = tmp_path / "tiny.gguf"
    f.write_bytes(b"GGUF")
    plan = plan_launch(cfg, hw_mac, str(f), ServeOptions())
    assert plan.launch.engine == "llama.cpp" and plan.launch.model_path == str(f.resolve())
    assert plan.launch.served_model_name == "tiny"


def test_port_helpers():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        busy = s.getsockname()[1]
        assert port_in_use("127.0.0.1", busy)
        assert find_free_port("127.0.0.1", busy) != busy


def _fake_plan(port: int, name: str = "t") -> LaunchPlan:
    launch = LaunchConfig(
        name=name,
        engine="llama.cpp",
        model_ref="fake",
        model_path="x",
        served_model_name="fake-model",
        port=port,
    )
    return LaunchPlan(launch, [sys.executable, str(FAKE), str(port)], {}, sys.executable)


def test_lifecycle_start_status_test_stop(cfg):
    runner = Runner(assume_yes=True, interactive=False)
    mgr = ServerManager(cfg, runner)
    port = free_port()
    st = mgr.start(_fake_plan(port), detach=True)
    try:
        assert st is not None and mgr.owns(st)
        status = mgr.status("t")[0]
        assert status["running"] and status["healthy"]
        res = mgr.smoke_test("t")
        assert res["ok"] and res["reply"] == "pong"
        with pytest.raises(UsageError, match="уже работает"):
            mgr.start(_fake_plan(port), detach=True)
        assert any("listening" in line for line in mgr.tail("t"))
    finally:
        assert mgr.stop("t")
    assert mgr.load("t") is None
    deadline = time.time() + 5
    while time.time() < deadline and port_in_use("127.0.0.1", port):
        time.sleep(0.1)
    assert not port_in_use("127.0.0.1", port)


def test_stop_never_kills_unrelated_process(cfg):
    """Не отправлять сигнал, если записанный PID теперь принадлежит другому процессу."""
    runner = Runner(assume_yes=True, interactive=False)
    mgr = ServerManager(cfg, runner)
    victim = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        st = ServerState(
            name="stale",
            pid=victim.pid,
            pgid=os.getpgid(victim.pid),
            started_at=0,
            proc_start="Mon Jan  1 00:00:00 2001",
            engine="vllm",
            model_ref="x",
            served_model_name="x",
            base_url="http://127.0.0.1:1",
            health_path="/health",
            log_file="/dev/null",
            argv=[],
        )
        mgr._save(st)
        assert mgr.stop("stale") is False
        assert victim.poll() is None  # процесс ещё работает
        assert mgr.load("stale") is None  # устаревшее состояние удалено
    finally:
        victim.kill()
        victim.wait()


def test_start_requires_engine_binary(cfg):
    mgr = ServerManager(cfg, Runner(assume_yes=True, interactive=False))
    plan = _fake_plan(free_port())
    plan.binary = None
    with pytest.raises(MissingDependencyError, match="engine-install"):
        mgr.start(plan, detach=True)


def test_start_detects_port_conflict(cfg):
    mgr = ServerManager(cfg, Runner(assume_yes=True, interactive=False))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        with pytest.raises(UsageError, match="уже используется"):
            mgr.start(_fake_plan(s.getsockname()[1]), detach=True)


def test_dry_run_does_not_start(cfg):
    mgr = ServerManager(cfg, Runner(dry_run=True, interactive=False))
    assert mgr.start(_fake_plan(free_port()), detach=True) is None
    assert mgr.all_states() == []


def test_crash_before_ready_reports_logs(cfg):
    mgr = ServerManager(cfg, Runner(assume_yes=True, interactive=False))
    launch = LaunchConfig(
        name="crash",
        engine="llama.cpp",
        model_ref="x",
        model_path="x",
        served_model_name="x",
        port=free_port(),
    )
    plan = LaunchPlan(
        launch,
        [sys.executable, "-c", "print('boom: bad model'); raise SystemExit(3)"],
        {},
        sys.executable,
    )
    with pytest.raises(Exception, match="завершился с кодом 3") as ei:
        mgr.start(plan, detach=True)
    assert "boom: bad model" in (ei.value.hint or "")
    assert mgr.load("crash") is None


def test_api_key_written_privately(cfg, monkeypatch):
    monkeypatch.setenv("SETUP_WORK_API_KEY", "topsecret")
    mgr = ServerManager(cfg, Runner(assume_yes=True, interactive=False))
    plan = _fake_plan(free_port())
    mgr.prepare_api_key(plan)
    kf = Path(plan.launch.api_key_file)
    assert kf.read_text().strip() == "topsecret"
    assert (kf.stat().st_mode & 0o777) == 0o600
    assert "topsecret" not in " ".join(plan.argv)
