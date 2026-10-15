from __future__ import annotations

import pytest

from setup_work.config import Config
from setup_work.errors import ConfigError, NotFoundError
from setup_work.profiles import resolve_profile


def test_minimal(cfg):
    p = resolve_profile(cfg, "minimal")
    assert p.chain == ["minimal"]
    assert p.tools_required == ["git", "curl"]
    assert p.harnesses_required == [] and p.engines == []


def test_composition_order_and_union(cfg):
    p = resolve_profile(cfg, "full")
    assert p.chain == ["minimal", "developer", "coding-agent", "local-llm", "gpu-inference", "full"]
    assert p.tools_required[:2] == ["git", "curl"]
    assert "node" in p.tools_required and p.tools_required.count("git") == 1
    assert p.engines == ["llama.cpp", "vllm"]
    assert "claude-code" in p.harnesses_required
    # child mapping wins: gpu-inference overrides local-llm's engine preference
    assert p.model_preferences["engine"] == "vllm"
    assert p.inference["port"] == 8000
    assert not set(p.tools_optional) & set(p.tools_required)


def test_profiles_never_download_by_default(cfg):
    for name in cfg.profiles.profiles:
        assert resolve_profile(cfg, name).model_pull == []


def test_cycle_and_unknown(tmp_path):
    d = tmp_path / "userconf"
    d.mkdir(exist_ok=True)
    (d / "profiles.yaml").write_text(
        "profiles:\n  a: {extends: [b]}\n  b: {extends: [a]}\n  c: {extends: [ghost]}\n"
    )
    cfg = Config()
    with pytest.raises(ConfigError, match="cycle"):
        resolve_profile(cfg, "a")
    with pytest.raises(ConfigError, match="unknown profile"):
        resolve_profile(cfg, "c")
    with pytest.raises(NotFoundError):
        resolve_profile(cfg, "zzz")


def test_custom_profile_extends_builtin(tmp_path):
    d = tmp_path / "userconf"
    d.mkdir(exist_ok=True)
    (d / "profiles.yaml").write_text(
        "profiles:\n  team:\n    extends: [coding-agent]\n    harnesses: {required: [aider]}\n"
    )
    p = resolve_profile(Config(), "team")
    assert p.harnesses_required[-1] == "aider"
    assert "aider" not in p.harnesses_optional
