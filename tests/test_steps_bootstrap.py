from __future__ import annotations

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


def test_download_is_atomic_and_requires_https(tmp_path, monkeypatch):
    with pytest.raises(SetupWorkError, match="только по HTTPS"):
        st.download("http://example.com/x", tmp_path / "x")
    monkeypatch.setattr(st.urllib.request, "urlopen", lambda req, timeout: io.BytesIO(b"payload"))
    st.download("https://example.com/x", tmp_path / "x")
    assert (tmp_path / "x").read_bytes() == b"payload"

    def fail(req, timeout):
        raise OSError("connection lost")

    monkeypatch.setattr(st.urllib.request, "urlopen", fail)
    with pytest.raises(OSError, match="connection lost"):
        st.download("https://example.com/x", tmp_path / "x")
    assert (tmp_path / "x").read_bytes() == b"payload"
    assert not list(tmp_path.glob("*.part"))


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


def test_tools_never_fall_back_to_public_hosts(cfg, monkeypatch):
    monkeypatch.setattr(bs, "_platform", lambda: ("Darwin", "arm64"))
    monkeypatch.setenv("NPM_CONFIG_USERCONFIG", str(cfg.user_dir / "npmrc"))
    for key in ("SETUP_WORK_NPM_REGISTRY", "NPM_CONFIG_REGISTRY", "npm_config_registry"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(NotFoundError, match="npm registry"):
        bs.plan_node(cfg)
    monkeypatch.setattr(bs, "detect_environment", lambda cfg, names=None: _env([]))
    (p,) = bs.plan_tools(cfg, ["node"])
    assert p.status == "manual" and "SETUP_WORK_NPM_REGISTRY" in p.steps[0].description


def test_plan_node_targets(cfg, monkeypatch):
    monkeypatch.setattr(bs, "_platform", lambda: ("Darwin", "arm64"))
    _corp_tools(cfg, node="/mirror/{name}")
    assert bs.plan_node(cfg)[0].artifact.uri == "/mirror/node-v24.20.0-darwin-arm64.tar.gz"
    monkeypatch.setattr(bs, "_platform", lambda: ("Linux", "riscv64"))
    with pytest.raises(UnsupportedError):
        bs.plan_node(cfg)


def test_idempotent_when_satisfied(cfg, monkeypatch):
    tools = [
        ToolStatus("python", True, "3.12.0", "/x/python", satisfies=True),
        ToolStatus("node", True, "24.1.0", "/n", satisfies=True),
        ToolStatus("npm", True, "12.2.0", "/npm"),
    ]
    monkeypatch.setattr(bs, "detect_environment", lambda cfg, names=None: _env(tools))
    plans = bs.plan_tools(cfg, ["python", "node"])
    assert all(p.status == "satisfied" and not p.steps for p in plans)


def test_engines_only_from_corporate_artifacts(cfg, hw_a100x4):
    for engine in ("llama.cpp", "vllm"):
        with pytest.raises(NotFoundError, match="не задан корпоративный артефакт"):
            bs.plan_engine(cfg, engine, hw_a100x4)


def test_optional_uv_uses_corporate_pip_and_separate_environment(cfg, monkeypatch):
    monkeypatch.setenv("PIP_INDEX_URL", "https://pypi.corp/simple")
    steps = bs.plan_uv(cfg)
    python = cfg.tools_dir / "uv" / cfg.defaults.tools["uv"].version / "bin/python"
    assert steps[0].argv[1:3] == ["-m", "venv"]
    assert steps[1].argv[:4] == [str(python), "-m", "pip", "install"]
    assert steps[1].argv[-1] == "uv==0.12.13"
    assert [step.dest.name for step in steps if step.kind == "link"] == ["uv", "uvx"]
