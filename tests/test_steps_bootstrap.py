from __future__ import annotations

import hashlib
import io
import tarfile

import pytest

from setup_work import bootstrap as bs
from setup_work import steps as st
from setup_work.diagnostics import Environment, ToolStatus, version_satisfies
from setup_work.errors import SetupWorkError, UnsupportedError
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
    with pytest.raises(SetupWorkError, match="escapes"):
        st.safe_extract(_tar_with(tmp_path, "../evil"), tmp_path / "out")
    with pytest.raises(SetupWorkError, match="escapes"):
        st.safe_extract(_tar_with(tmp_path, "ok/link", linkname="/etc/passwd"), tmp_path / "o2")


def test_safe_extract_ok(tmp_path):
    st.safe_extract(_tar_with(tmp_path, "pkg/bin/tool", b"hi"), tmp_path / "out")
    assert (tmp_path / "out" / "pkg" / "bin" / "tool").read_bytes() == b"hi"


def test_download_requires_https_and_checksum(tmp_path, monkeypatch):
    with pytest.raises(SetupWorkError, match="non-HTTPS"):
        st.download_verified("http://example.com/x", tmp_path / "x", "0" * 64)

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(st.urllib.request, "urlopen", lambda req, timeout: Resp(b"payload"))
    with pytest.raises(SetupWorkError, match="checksum mismatch"):
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
    with pytest.raises(SetupWorkError, match="not a symlink"):
        st.make_link(target, dest, managed)
    dest.unlink()
    dest.symlink_to("/usr/bin/true")
    with pytest.raises(SetupWorkError, match="does not manage"):
        st.make_link(target, dest, managed)
    dest.unlink()
    st.make_link(target, dest, managed)
    st.make_link(target, dest, managed)  # idempotent refresh
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
    with pytest.raises(Exception, match="confirmation required"):
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


def _env(tools, pms=(), root=False, sudo=False):
    return Environment(
        python_runtime="3.12",
        tools=tools,
        package_managers=list(pms),
        is_root=root,
        sudo_present=sudo,
        bin_dir_on_path=True,
        bin_dir="/b",
    )


def test_system_package_without_sudo_is_manual():
    plan = bs.system_package_plan(_env([], pms=["apt-get"], sudo=False), "wget", False)
    assert plan.status == "manual" and "administrator" in plan.detail


def test_system_package_with_sudo_needs_opt_in(monkeypatch):
    monkeypatch.setattr(bs, "_platform", lambda: ("Linux", "x86_64"))
    plan = bs.system_package_plan(_env([], pms=["apt-get"], sudo=True), "wget", False)
    assert plan.status == "manual" and "--allow-sudo" in plan.detail
    plan = bs.system_package_plan(_env([], pms=["apt-get"], sudo=True), "wget", True)
    assert plan.status == "install"
    assert all(s.privileged for s in plan.steps)
    assert plan.steps[-1].argv[:2] == ["sudo", "apt-get"]


def test_root_in_container_uses_package_manager(monkeypatch):
    monkeypatch.setattr(bs, "_platform", lambda: ("Linux", "x86_64"))
    plan = bs.system_package_plan(_env([], pms=["apt-get"], root=True), "git", False)
    assert plan.status == "install" and plan.steps[-1].argv[0] == "apt-get"


def test_plan_uv_pinned(cfg, monkeypatch):
    monkeypatch.setattr(bs, "_platform", lambda: ("Linux", "x86_64"))
    monkeypatch.setattr(bs, "_is_musl", lambda: False)
    steps = bs.plan_uv(cfg)
    dl = steps[0]
    assert dl.kind == "download" and "0.12.13" in dl.url
    assert dl.sha256 == cfg.defaults.tools["uv"].sha256["x86_64-unknown-linux-gnu"]
    assert [s.dest.name for s in steps if s.kind == "link"] == ["uv", "uvx"]


def test_plan_node_and_git_lfs_targets(cfg, monkeypatch):
    monkeypatch.setattr(bs, "_platform", lambda: ("Darwin", "arm64"))
    assert bs.plan_node(cfg)[0].url.endswith("node-v24.20.0-darwin-arm64.tar.gz")
    lfs = bs.plan_git_lfs(cfg)
    assert lfs[0].url.endswith("git-lfs-darwin-arm64-v3.8.0.zip")
    assert lfs[-1].kind == "manual"  # never edits ~/.gitconfig itself
    monkeypatch.setattr(bs, "_platform", lambda: ("Linux", "riscv64"))
    with pytest.raises(UnsupportedError):
        bs.plan_node(cfg)


def test_idempotent_when_satisfied(cfg, monkeypatch):
    tools = [ToolStatus("uv", True, "0.12.0", "/x/uv"), ToolStatus("git", True, "2.4", "/g")]
    monkeypatch.setattr(bs, "detect_environment", lambda cfg, names=None: _env(tools))
    plans = bs.plan_tools(cfg, ["uv", "git"])
    assert all(p.status == "satisfied" and not p.steps for p in plans)


def test_llama_cpp_asset_selection(hw_4090, hw_v100, hw_mac, hw_cpu):
    assert bs.llama_cpp_asset(hw_4090)[0] == "ubuntu-cuda-12.8-x64"
    asset, notes = bs.llama_cpp_asset(hw_v100)  # driver reports CUDA 12.2
    assert asset == "ubuntu-x64" and any("12.8" in n for n in notes)
    assert bs.llama_cpp_asset(hw_mac)[0] == "macos-arm64"
    assert bs.llama_cpp_asset(hw_cpu)[0] == "ubuntu-x64"


def test_vllm_unsupported_on_mac_and_cpu(cfg, hw_mac, hw_cpu, hw_a100x4):
    for hw in (hw_mac, hw_cpu):
        with pytest.raises(UnsupportedError):
            bs.plan_engine(cfg, "vllm", hw)
    steps, _ = bs.plan_engine(cfg, "vllm", hw_a100x4)
    assert any("vllm==0.30.0" in (s.argv or []) for s in steps)
