from __future__ import annotations

import hashlib
import io
import tarfile

import pytest

from setup_work import bootstrap as bs
from setup_work import steps as st
from setup_work.config import ArtifactRef
from setup_work.diagnostics import Environment, ToolStatus, version_satisfies
from setup_work.errors import NotFoundError, SetupWorkError, UnsupportedError
from setup_work.runner import Runner, redact_argv


def _tar_with(tmp_path, name, data=b"x", linkname=None):
    p = tmp_path / "a.tar.gz"
    with tarfile.open(p, "w:gz") as tf:
        info = tarfile.TarInfo(name)
        if linkname:
            info.type = tarfile.SYMTYPE
            info.linkname = linkname
            tf.addfile(info)
        else:
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return p


def test_safe_extract_rejects_traversal(tmp_path):
    with pytest.raises(SetupWorkError, match="выходит за пределы"):
        st.safe_extract(_tar_with(tmp_path, "../evil"), tmp_path / "out")
    with pytest.raises(SetupWorkError, match="выходит за пределы"):
        st.safe_extract(_tar_with(tmp_path, "ok/link", linkname="/etc/passwd"), tmp_path / "o2")


def test_safe_extract_ok(tmp_path):
    st.safe_extract(_tar_with(tmp_path, "pkg/bin/tool", b"hi"), tmp_path / "out")
    assert (tmp_path / "out" / "pkg" / "bin" / "tool").read_bytes() == b"hi"


def test_download_requires_https_and_checksum(tmp_path, monkeypatch):
    with pytest.raises(SetupWorkError, match="только по HTTPS"):
        st.download_verified("http://example.com/x", tmp_path / "x", "0" * 64)

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(st.urllib.request, "urlopen", lambda req, timeout: Resp(b"payload"))
    with pytest.raises(SetupWorkError, match="не совпадает SHA256"):
        st.download_verified("https://example.com/x", tmp_path / "x", "0" * 64)
    assert not (tmp_path / "x").exists()
    assert not list(tmp_path.glob("*.part"))
    good = hashlib.sha256(b"payload").hexdigest()
    st.download_verified("https://example.com/x", tmp_path / "x", good)
    assert (tmp_path / "x").read_bytes() == b"payload"


def test_make_link_refuses_foreign_files(tmp_path):
    managed = tmp_path / "managed"
    (managed / "t").mkdir(parents=True)
    target = managed / "t" / "tool"
    target.write_text("")
    dest = tmp_path / "bin" / "tool"
    dest.parent.mkdir()
    dest.write_text("user's own binary")
    with pytest.raises(SetupWorkError, match="не является символьной ссылкой"):
        st.make_link(target, dest, managed)
    dest.unlink()
    dest.symlink_to("/usr/bin/true")
    with pytest.raises(SetupWorkError, match="не управляет"):
        st.make_link(target, dest, managed)
    dest.unlink()
    st.make_link(target, dest, managed)
    st.make_link(target, dest, managed)  # повторное обновление без изменения результата
    assert dest.resolve() == target.resolve()


def test_dry_run_executes_nothing(tmp_path):
    steps = [
        st.Step("command", "touch", argv=["touch", str(tmp_path / "f")]),
        st.Step("write", "write", dest=tmp_path / "g", content="x"),
    ]
    st.execute(steps, Runner(dry_run=True, interactive=False), tmp_path)
    assert not (tmp_path / "f").exists() and not (tmp_path / "g").exists()


def test_privileged_step_needs_confirmation(tmp_path):
    step = st.Step("command", "sudo", argv=["sudo", "true"], privileged=True)
    with pytest.raises(Exception, match="требуется подтверждение"):
        st.execute([step], Runner(interactive=False), tmp_path)


def test_redaction():
    assert redact_argv(["x", "--api-key", "s", "--hf-token=abc"]) == [
        "x",
        "--api-key",
        "***",
        "--hf-token=***",
    ]


@pytest.mark.parametrize(
    "found,spec,ok",
    [
        ("v22.11.0", ">=22", True),
        ("20.1.0", ">=22", False),
        ("3.12.1", ">=3.11", True),
        (None, ">=1", False),
        ("1.2", "==1.2", True),
    ],
)
def test_version_satisfies(found, spec, ok):
    assert version_satisfies(found, spec) is ok


def _env(tools):
    return Environment(python_runtime="3.12", tools=tools, bin_dir_on_path=True, bin_dir="/b")


def _corp_tools(cfg, **uris):
    cfg.corporate.tools.update({k: ArtifactRef(uri=v) for k, v in uris.items()})
    return cfg


def test_plan_uv_from_corporate_source(cfg, monkeypatch):
    monkeypatch.setattr(bs, "_platform", lambda: ("Linux", "x86_64"))
    monkeypatch.setattr(bs, "_is_musl", lambda: False)
    steps = bs.plan_uv(_corp_tools(cfg, uv="hdfs:///tools/uv/{name}"))
    fetch = steps[0]
    assert fetch.kind == "fetch"
    assert fetch.artifact.uri == "hdfs:///tools/uv/uv-0.12.13-x86_64-unknown-linux-gnu.tar.gz"
    assert fetch.sha256 == cfg.defaults.tools["uv"].sha256["x86_64-unknown-linux-gnu"]
    assert [s.dest.name for s in steps if s.kind == "link"] == ["uv", "uvx"]


def test_tools_never_fall_back_to_public_hosts(cfg, monkeypatch):
    monkeypatch.setattr(bs, "_platform", lambda: ("Darwin", "arm64"))
    for plan in (bs.plan_uv, bs.plan_node):
        with pytest.raises(NotFoundError, match="не задан корпоративный источник"):
            plan(cfg)
    monkeypatch.setattr(bs, "detect_environment", lambda cfg, names=None: _env([]))
    (p,) = bs.plan_tools(cfg, ["node"])
    assert p.status == "manual" and "tools.node.uri" in p.steps[0].description


def test_plan_node_targets(cfg, monkeypatch):
    monkeypatch.setattr(bs, "_platform", lambda: ("Darwin", "arm64"))
    _corp_tools(cfg, node="/mirror/{name}")
    assert bs.plan_node(cfg)[0].artifact.uri == "/mirror/node-v24.20.0-darwin-arm64.tar.gz"
    monkeypatch.setattr(bs, "_platform", lambda: ("Linux", "riscv64"))
    with pytest.raises(UnsupportedError):
        bs.plan_node(cfg)


def test_idempotent_when_satisfied(cfg, monkeypatch):
    tools = [
        ToolStatus("uv", True, "0.12.0", "/x/uv"),
        ToolStatus("node", True, "24.1.0", "/n", satisfies=True),
    ]
    monkeypatch.setattr(bs, "detect_environment", lambda cfg, names=None: _env(tools))
    plans = bs.plan_tools(cfg, ["uv", "node"])
    assert all(p.status == "satisfied" and not p.steps for p in plans)


def test_engines_only_from_corporate_artifacts(cfg, hw_a100x4):
    for engine in ("llama.cpp", "vllm"):
        with pytest.raises(NotFoundError, match="не задан корпоративный артефакт"):
            bs.plan_engine(cfg, engine, hw_a100x4)
