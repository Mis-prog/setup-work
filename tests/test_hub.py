from __future__ import annotations

from types import SimpleNamespace

import pytest

from setup_work import hub
from setup_work.errors import SetupWorkError, UsageError
from setup_work.models import resolve
from setup_work.runner import Runner


def test_target_for_catalog_and_raw(cfg):
    repo, rev, allow, ignore = hub.target_for(cfg, "qwen3-8b:q4_k_m")
    assert repo == "Qwen/Qwen3-8B-GGUF" and allow == ["Qwen3-8B-Q4_K_M.gguf"] and ignore is None
    repo, rev, allow, ignore = hub.target_for(cfg, "Org/Model@abc123")
    assert (repo, rev, allow) == ("Org/Model", "abc123", None)
    assert "original/*" in ignore
    with pytest.raises(UsageError, match="specify which variant"):
        hub.target_for(cfg, "qwen3-8b")


def test_gguf_shard_entry(cfg, tmp_path):
    _, v = resolve(cfg, "qwen2.5-coder-32b:q4_k_m")
    for i in (1, 2, 3):
        (tmp_path / f"qwen2.5-coder-32b-instruct-q4_k_m-0000{i}-of-00003.gguf").write_bytes(b"")
    assert hub.gguf_entry_file(tmp_path, v).name.endswith("00001-of-00003.gguf")
    (tmp_path / "qwen2.5-coder-32b-instruct-q4_k_m-00003-of-00003.gguf").unlink()
    with pytest.raises(SetupWorkError, match="incomplete"):
        hub.gguf_entry_file(tmp_path, v)


def _fake_info(files):
    sib = [
        SimpleNamespace(rfilename=n, size=s, lfs=SimpleNamespace(sha256="a" * 64)) for n, s in files
    ]
    return SimpleNamespace(siblings=sib, sha="deadbeef" * 5)


def test_plan_pull_sizes_and_patterns(cfg, monkeypatch):
    import huggingface_hub

    files = [
        ("Qwen3-8B-Q4_K_M.gguf", 5_030_000_000),
        ("Qwen3-8B-Q8_0.gguf", 8_710_000_000),
        ("README.md", 100),
    ]
    monkeypatch.setattr(
        huggingface_hub.HfApi, "model_info", lambda self, repo, **kw: _fake_info(files)
    )
    plan = hub.plan_pull(cfg, "qwen3-8b:q4_k_m")
    assert [f.name for f in plan.files] == ["Qwen3-8B-Q4_K_M.gguf"]
    assert plan.total_bytes == 5_030_000_000 and plan.needed_bytes == plan.total_bytes
    assert plan.free_bytes is not None


def test_pull_refuses_when_disk_too_small(cfg, monkeypatch):
    plan = hub.PullPlan(
        "x",
        "a/b",
        None,
        None,
        None,
        [hub.RemoteFile("f", 10**12, None)],
        free_bytes=10**9,
        cache_dir=None,
    )
    with pytest.raises(SetupWorkError, match="not enough disk space"):
        hub.pull(cfg, plan, Runner(interactive=False))


def test_pull_large_download_requires_confirmation(cfg):
    plan = hub.PullPlan(
        "x",
        "a/b",
        None,
        None,
        None,
        [hub.RemoteFile("f", 5 * 10**9, None)],
        free_bytes=10**13,
        cache_dir=None,
    )
    with pytest.raises(Exception, match="confirmation required"):
        hub.pull(cfg, plan, Runner(interactive=False))


def test_verify_detects_mismatch(cfg, tmp_path):
    f = tmp_path / "w.gguf"
    f.write_bytes(b"weights")
    bad = hub.verify_files(
        cfg, tmp_path, [hub.RemoteFile("w.gguf", 7, "0" * 64)], Runner(interactive=False)
    )
    assert bad and "mismatch" in bad[0]
    import hashlib

    good = hashlib.sha256(b"weights").hexdigest()
    assert (
        hub.verify_files(
            cfg, tmp_path, [hub.RemoteFile("w.gguf", 7, good)], Runner(interactive=False)
        )
        == []
    )


def test_cache_dir_respects_env(cfg, monkeypatch, tmp_path):
    assert str(hub.cache_dir(cfg)).startswith(str(tmp_path))  # HF_HUB_CACHE from conftest
    monkeypatch.setenv("SETUP_WORK_MODEL_CACHE", str(tmp_path / "custom"))
    from setup_work.config import Config

    assert hub.cache_dir(Config()) == tmp_path / "custom"


def test_list_cached_empty(cfg):
    assert hub.list_cached(cfg) == []
