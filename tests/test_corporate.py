from __future__ import annotations

import hashlib
import os
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from conftest import load_hw
from setup_work import artifacts, corporate, hub
from setup_work import bootstrap as bs
from setup_work import harness as hz
from setup_work.config import Config, CorporateConfig, HarnessSpec
from setup_work.engines import VllmEngine
from setup_work.errors import SetupWorkError, UsageError
from setup_work.inference import ServeOptions, plan_launch
from setup_work.models import resolve
from setup_work.runner import Runner
from setup_work.steps import execute

ROOT = Path(__file__).resolve().parents[1]

FAKE_HDFS = """#!/bin/sh
# fake `hdfs dfs` backed by $FAKE_HDFS_ROOT: supports -du -s URI and -get URI DEST
shift  # dfs
op=$1; shift
case "$op" in
  -du) shift; p=$(printf '%s' "$1" | sed 's#^hdfs://[^/]*##; s#^hdfs:##'); f="$FAKE_HDFS_ROOT$p"
       [ -e "$f" ] || { echo "du: No such file or directory: $1" >&2; exit 1; }
       printf '%s  %s  %s\\n' "$(wc -c < "$f" | tr -d ' ')" 0 "$1" ;;
  -get) p=$(printf '%s' "$1" | sed 's#^hdfs://[^/]*##; s#^hdfs:##'); cp "$FAKE_HDFS_ROOT$p" "$2" ;;
  *) echo "unsupported $op" >&2; exit 2 ;;
esac
"""


@pytest.fixture
def fake_hdfs(tmp_path, monkeypatch):
    root = tmp_path / "hdfsroot"
    root.mkdir()
    bindir = tmp_path / "fakebin"
    bindir.mkdir()
    exe = bindir / "hdfs"
    exe.write_text(FAKE_HDFS)
    exe.chmod(0o755)
    monkeypatch.setenv("FAKE_HDFS_ROOT", str(root))
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    return root


@pytest.fixture(autouse=True)
def _clear_team(monkeypatch):
    monkeypatch.delenv("SETUP_WORK_TEAM_CONFIG_DIR", raising=False)
    for k in (
        "UV_DEFAULT_INDEX",
        "UV_INDEX_URL",
        "UV_INDEX",
        "UV_INSECURE_HOST",
        "PIP_INDEX_URL",
        "PIP_CONFIG_FILE",
    ):
        monkeypatch.delenv(k, raising=False)


def team(tmp_path, corporate_yaml: str = "", **extra: str) -> Config:
    d = tmp_path / "team"
    d.mkdir(exist_ok=True)
    (d / "corporate.yaml").write_text("hdfs: {kerberos: 'off'}\n" + corporate_yaml)
    for fname, body in extra.items():
        (d / fname).write_text(body)
    os.environ["SETUP_WORK_TEAM_CONFIG_DIR"] = str(d)
    return Config()


def make_zip(path: Path, files: dict[str, bytes]) -> str:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return hashlib.sha256(path.read_bytes()).hexdigest()


RUN = Runner(assume_yes=True, interactive=False)


# ------------------------------------------------------------------ config


def test_corporate_schema_rejects_secrets_and_bad_uris():
    with pytest.raises(ValidationError, match="secrets"):
        CorporateConfig.model_validate({"env": {"HF_TOKEN": "x"}})
    with pytest.raises(ValidationError, match="artifact URI"):
        CorporateConfig.model_validate({"models": {"a:b": {"uri": "-rf /"}}})
    with pytest.raises(ValidationError, match="artifact URI"):
        CorporateConfig.model_validate({"models": {"a:b": {"uri": "ftp://x/y.zip"}}})
    ok = CorporateConfig.model_validate(
        {"models": {"a:b": {"uri": "hdfs://nameservice1/ml/x.zip"}}}
    )
    assert ok.models["a:b"].effective_format == "zip"


def test_team_layer_and_unknown_model_mapping(tmp_path):
    cfg = team(tmp_path, "models:\n  'nope:x': {uri: hdfs:///a.zip}\n")
    assert any("not a catalog model:variant" in p for p in cfg.validate_all())


def test_example_corp_config_is_valid(monkeypatch):
    monkeypatch.setenv("SETUP_WORK_TEAM_CONFIG_DIR", str(ROOT / "examples" / "corp"))
    cfg = Config()
    assert cfg.validate_all() == []
    qwen = cfg.harnesses.harnesses["qwen-code"]
    assert qwen.source == "git" and qwen.package is None  # entry replaced, not merged
    assert "corp-coder-14b" in cfg.models.models and "qwen3-8b" in cfg.models.models
    with pytest.raises(UsageError, match="CHANGE-ME"):
        hz.plan_install(cfg, "qwen-code")


# ------------------------------------------------------------- artifacts


def test_model_pull_from_hdfs_gguf(tmp_path, fake_hdfs, hw_4090):
    (fake_hdfs / "ml").mkdir()
    sha = make_zip(fake_hdfs / "ml" / "q.zip", {"Qwen3-8B/Qwen3-8B-Q4_K_M.gguf": b"GGUF" * 100})
    cfg = team(
        tmp_path, f"models:\n  'qwen3-8b:q4_k_m': {{uri: 'hdfs:///ml/q.zip', sha256: {sha}}}\n"
    )
    spec, var = resolve(cfg, "qwen3-8b:q4_k_m")
    entry = hub.pull_artifact(cfg, "qwen3-8b", var, hub.model_artifact(cfg, "qwen3-8b", var), RUN)
    assert entry.name == "Qwen3-8B-Q4_K_M.gguf" and entry.is_file()
    assert hub.local_model(cfg, "qwen3-8b", var) == entry
    # archive removed after unpacking (weights are large)
    assert not list((cfg.artifacts_dir / "downloads").glob("*.zip"))
    # second run reuses
    res = artifacts.install(
        cfg,
        hub.model_artifact(cfg, "qwen3-8b", var),
        hub.artifact_model_dir(cfg, "qwen3-8b", var),
        RUN,
        name="x",
    )
    assert res.reused
    # serve resolves the unpacked file without touching Hugging Face
    plan = plan_launch(cfg, hw_4090, "qwen3-8b:q4_k_m", ServeOptions())
    assert plan.launch.model_path == str(entry)
    assert hub.list_artifact_models(cfg)[0]["ref"] == "qwen3-8b:q4_k_m"


def test_model_pull_safetensors_dir_and_validation(tmp_path, fake_hdfs):
    good = {"m/config.json": b"{}", "m/model.safetensors": b"x", "m/tokenizer.json": b"{}"}
    make_zip(fake_hdfs / "st.zip", good)
    make_zip(fake_hdfs / "bad.zip", {"readme.txt": b"no weights"})
    cfg = team(
        tmp_path,
        "models:\n  'qwen3-8b:awq': {uri: 'hdfs:///st.zip'}\n"
        "  'qwen3-8b:fp8': {uri: 'hdfs:///bad.zip'}\n",
    )
    _, awq = resolve(cfg, "qwen3-8b:awq")
    root = hub.pull_artifact(cfg, "qwen3-8b", awq, hub.model_artifact(cfg, "qwen3-8b", awq), RUN)
    assert (root / "config.json").is_file() and root.name == "m"
    _, fp8 = resolve(cfg, "qwen3-8b:fp8")
    with pytest.raises(SetupWorkError, match="config.json"):
        hub.pull_artifact(cfg, "qwen3-8b", fp8, hub.model_artifact(cfg, "qwen3-8b", fp8), RUN)


def test_checksum_mismatch_deletes_copy(tmp_path, fake_hdfs):
    make_zip(fake_hdfs / "x.zip", {"a.gguf": b"1"})
    cfg = team(
        tmp_path, "models:\n  'qwen3-8b:q8_0': {uri: 'hdfs:///x.zip', sha256: '" + "0" * 64 + "'}\n"
    )
    _, v = resolve(cfg, "qwen3-8b:q8_0")
    with pytest.raises(SetupWorkError, match="checksum mismatch"):
        hub.pull_artifact(cfg, "qwen3-8b", v, hub.model_artifact(cfg, "qwen3-8b", v), RUN)
    assert not list((cfg.artifacts_dir / "downloads").iterdir())
    assert not hub.artifact_model_dir(cfg, "qwen3-8b", v).exists()


def test_missing_hdfs_path_and_kerberos(tmp_path, fake_hdfs, monkeypatch):
    cfg = team(tmp_path, "models:\n  'qwen3-8b:q8_0': {uri: 'hdfs:///missing.zip'}\n")
    _, v = resolve(cfg, "qwen3-8b:q8_0")
    with pytest.raises(SetupWorkError, match="does not exist on HDFS"):
        hub.pull_artifact(cfg, "qwen3-8b", v, hub.model_artifact(cfg, "qwen3-8b", v), RUN)
    # Kerberos required and no ticket -> actionable error, nothing fetched
    klist = tmp_path / "fakebin" / "klist"
    klist.write_text("#!/bin/sh\nexit 1\n")
    klist.chmod(0o755)
    (tmp_path / "team" / "corporate.yaml").write_text(
        "hdfs: {kerberos: required}\nmodels:\n  'qwen3-8b:q8_0': {uri: 'hdfs:///missing.zip'}\n"
    )
    cfg = Config()
    with pytest.raises(SetupWorkError, match="kinit"):
        hub.pull_artifact(cfg, "qwen3-8b", v, hub.model_artifact(cfg, "qwen3-8b", v), RUN)


def test_unzip_traversal_and_symlinks(tmp_path):
    evil = tmp_path / "evil.zip"
    make_zip(evil, {"../../escape.txt": b"x"})
    with pytest.raises(SetupWorkError, match="escapes"):
        artifacts.safe_unzip(evil, tmp_path / "out")
    linkzip = tmp_path / "link.zip"
    with zipfile.ZipFile(linkzip, "w") as zf:
        zf.writestr("pkg/real.so.1", b"lib")
        info = zipfile.ZipInfo("pkg/real.so")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(info, "real.so.1")
        exe = zipfile.ZipInfo("pkg/tool")
        exe.external_attr = (stat.S_IFREG | 0o755) << 16
        zf.writestr(exe, b"#!/bin/sh\necho hi\n")
        bad = zipfile.ZipInfo("pkg/out")
        bad.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(bad, "../../../etc")
    with pytest.raises(SetupWorkError, match="escapes"):
        artifacts.safe_unzip(linkzip, tmp_path / "o2")
    with zipfile.ZipFile(linkzip) as zf, zipfile.ZipFile(tmp_path / "ok.zip", "w") as out:
        for i in zf.infolist():
            if i.filename != "pkg/out":
                out.writestr(i, zf.read(i))
    artifacts.safe_unzip(tmp_path / "ok.zip", tmp_path / "o3")
    assert (tmp_path / "o3" / "pkg" / "real.so").is_symlink()
    assert os.access(tmp_path / "o3" / "pkg" / "tool", os.X_OK)


# ---------------------------------------------------------------- engines


def _zip_tree(src: Path, dest: Path) -> None:
    """Like plain `zip -r` (no -y): symlinks are followed and stored as files."""
    with zipfile.ZipFile(dest, "w") as zf:
        for p in sorted(src.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(src.parent))


def test_vllm_venv_from_plain_zip_is_repaired(tmp_path, fake_hdfs):
    venv = tmp_path / "build" / "vllm-venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
    # Simulate a venv built on another machine: base interpreter path does not exist.
    pv = venv / "pyvenv.cfg"
    pv.write_text(
        "\n".join(
            "home = /nonexistent/python/bin" if ln.startswith("home") else ln
            for ln in pv.read_text().splitlines()
        )
        + "\n"
    )
    script = venv / "bin" / "vllm"
    script.write_text("#!/build/host/vllm-venv/bin/python\nprint('cli')\n")
    _zip_tree(venv, fake_hdfs / "vllm.zip")
    cfg = team(tmp_path, "engines:\n  vllm: {uri: 'hdfs:///vllm.zip'}\n")
    steps = bs.plan_engine_artifact(cfg, "vllm", cfg.corporate.engines["vllm"])
    execute(steps[:2], RUN, cfg.data_dir)  # skip `import vllm` verification
    current = cfg.engines_dir / "vllm" / "current"
    py = current / "bin" / "python"
    out = subprocess.run(
        [str(py), "-c", "import sys; print(sys.prefix)"], capture_output=True, text=True, check=True
    )
    assert Path(out.stdout.strip()).resolve() == current.resolve()
    assert (current / "bin" / "vllm").read_text().startswith(f"#!{current.resolve()}/bin/python")
    eng = VllmEngine()
    binary = eng.find_binary(cfg)
    assert binary == str(py)
    from setup_work.engines import LaunchConfig

    lc = LaunchConfig(
        name="d",
        engine="vllm",
        model_ref="m",
        model_path="/m",
        served_model_name="m",
        binary=binary,
    )
    argv = eng.build_command(lc, binary)
    assert argv[0] == binary and argv[1] == "-c" and "vllm.entrypoints.cli.main" in argv[2]
    assert argv[3:5] == ["serve", "/m"]
    assert (
        eng.build_env(lc)["VIRTUAL_ENV"] == str(current / "bin" / "python")[: -len("/bin/python")]
    )


def test_llama_cpp_artifact_activation(tmp_path, fake_hdfs):
    files = {
        "build/bin/llama-server": b"#!/bin/sh\necho 'version: corp'\n",
        "build/bin/libggml.so": b"lib",
    }
    with zipfile.ZipFile(fake_hdfs / "llama.zip", "w") as zf:
        for n, d in files.items():
            i = zipfile.ZipInfo(n)
            i.external_attr = (stat.S_IFREG | 0o644) << 16  # exec bit lost on purpose
            zf.writestr(i, d)
    cfg = team(tmp_path, "engines:\n  llama.cpp: {uri: 'hdfs:///llama.zip'}\n")
    steps, notes = bs.plan_engine(cfg, "llama.cpp", load_hw("1x-rtx4090"))
    assert any("corporate artifact" in n for n in notes)
    execute(steps, RUN, cfg.data_dir)  # includes `llama-server --version`
    from setup_work.engines import LlamaCppEngine

    b = LlamaCppEngine().find_binary(cfg)
    assert b and os.access(b, os.X_OK)
    assert "LD_LIBRARY_PATH" in LlamaCppEngine.runtime_env(b)
    with pytest.raises(SetupWorkError, match="no corporate artifact"):
        bs.plan_engine(team(tmp_path, ""), "vllm", load_hw("1x-rtx4090"), source="corporate")


def test_tool_archive_from_corporate_source(tmp_path, monkeypatch):
    monkeypatch.setattr(bs, "_platform", lambda: ("Linux", "x86_64"))
    cfg = team(tmp_path, "tools:\n  node: {uri: 'hdfs:///tools/node-v{version}-{target}.tar.xz'}\n")
    step = bs.plan_node(cfg)[0]
    assert step.kind == "fetch"
    assert step.artifact.uri == "hdfs:///tools/node-v24.20.0-linux-x64.tar.xz"
    assert step.sha256 == cfg.defaults.tools["node"].sha256["linux-x64"]  # pin still enforced


# ------------------------------------------------------------ pip / scripts


def test_pip_conf_bridge(tmp_path, monkeypatch):
    xdg = tmp_path / "xdgconf"
    (xdg / "pip").mkdir(parents=True)
    (xdg / "pip" / "pip.conf").write_text(
        "[global]\nindex-url = https://user:pw@nexus.corp/pypi/simple\n"
        "extra-index-url = https://nexus.corp/ml/simple\n  https://nexus.corp/other/simple\n"
        "trusted-host = nexus.corp\ncert = /etc/corp-ca.pem\n"
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    env = corporate.pip_conf_bridge()
    assert env["UV_DEFAULT_INDEX"] == "https://user:pw@nexus.corp/pypi/simple"
    assert env["UV_INDEX"] == "https://nexus.corp/ml/simple https://nexus.corp/other/simple"
    assert env["UV_INSECURE_HOST"] == "nexus.corp"
    assert corporate.mask(env["UV_DEFAULT_INDEX"]) == "https://***@nexus.corp/pypi/simple"
    monkeypatch.setenv("UV_DEFAULT_INDEX", "https://mine/simple")
    assert "UV_DEFAULT_INDEX" not in corporate.pip_conf_bridge()  # user setting wins


def test_corporate_scripts_are_confirmed_and_idempotent(tmp_path):
    marker = tmp_path / "pip.conf"
    script = tmp_path / "setup-pypi.sh"
    script.write_text(f"#!/bin/sh\necho configured > {marker}\n")
    script.chmod(0o755)
    cfg = team(
        tmp_path, f"scripts:\n  - name: pypi\n    argv: [{script}]\n    creates: [{marker}]\n"
    )
    steps = corporate.plan_scripts(cfg)
    with pytest.raises(Exception, match="confirmation required"):
        execute(steps, Runner(interactive=False), cfg.data_dir)
    execute(steps, RUN, cfg.data_dir)
    assert marker.read_text().strip() == "configured"
    assert steps[0].when is not None and steps[0].when() is False  # now skipped


def test_corporate_env_applied_without_overriding(tmp_path, monkeypatch):
    monkeypatch.setenv("NPM_CONFIG_REGISTRY", "https://mine/")
    cfg = team(
        tmp_path, "env:\n  NPM_CONFIG_REGISTRY: https://corp/\n  SW_TEST_MIRROR: https://m/\n"
    )
    monkeypatch.delenv("SW_TEST_MIRROR", raising=False)
    applied = corporate.apply_environment(cfg)
    assert os.environ["NPM_CONFIG_REGISTRY"] == "https://mine/"
    assert applied["SW_TEST_MIRROR"] == "https://m/"
    monkeypatch.delenv("SW_TEST_MIRROR")


# ---------------------------------------------------------------- harnesses


def test_wrapper_binaries_and_trusted_hosts(tmp_path, monkeypatch):
    monkeypatch.setattr(hz, "_check_requirements", lambda cfg, spec: [])
    src = tmp_path / "agent"
    (src / "dist").mkdir(parents=True)
    (src / "dist" / "cli.sh").write_text('echo agent-ok "$@"\n')
    cfg = team(
        tmp_path,
        "trusted_git_hosts: [git.corp.example]\n",
        **{
            "harnesses.yaml": f"""
harnesses:
  local-agent:
    source: local
    path: {src}
    install:
      method: none
      build: [[sh, -c, "echo built > dist/BUILT"], ["true"]]
    binaries:
      - {{name: lagent, path: dist/cli.sh, interpreter: sh}}
  forked:
    source: git
    repository: git@git.corp.example:ai/forked.git
    ref: v1.0.0
    install: {{method: none, build: [[npm, ci], [npm, run, build]]}}
    binaries: [{{name: forked, path: dist/cli.js, interpreter: node}}]
"""
        },
    )
    with pytest.raises(UsageError, match="allow-build"):
        hz.plan_install(cfg, "local-agent")  # local paths are never implicitly trusted
    plan = hz.plan_install(cfg, "local-agent", allow_build=True)
    execute(plan.steps, RUN, cfg.data_dir)
    hz.write_manifest(cfg, "local-agent", plan.spec, RUN)
    assert (src / "dist" / "BUILT").exists()
    out = subprocess.run([str(cfg.bin_dir / "lagent"), "x"], capture_output=True, text=True)
    assert out.stdout.strip() == "agent-ok x"
    forked = hz.plan_install(cfg, "forked")  # trusted host: no --allow-build needed
    assert any("trusted host" in w for w in forked.warnings)
    assert [s.argv for s in forked.steps if s.description.startswith("build step")] == [
        ["npm", "ci"],
        ["npm", "run", "build"],
    ]
    steps = hz.plan_remove(cfg, "local-agent")
    execute(steps, RUN, cfg.data_dir)
    assert not (cfg.bin_dir / "lagent").exists() and src.exists()


def test_requires_commands(tmp_path):
    spec = HarnessSpec.model_validate(
        {
            "source": "local",
            "path": str(tmp_path),
            "install": {"method": "none"},
            "requires_commands": ["definitely-not-a-cmd"],
        }
    )
    assert any("definitely-not-a-cmd" in p for p in hz._check_requirements(Config(), spec))


def test_binary_path_must_stay_inside():
    with pytest.raises(ValidationError, match="inside the checkout"):
        HarnessSpec.model_validate(
            {
                "source": "local",
                "path": "/x",
                "install": {"method": "none"},
                "binaries": [{"name": "a", "path": "../../bin/sh"}],
            }
        )


# ---------------------------------------------------------------- stage 0


def test_bootstrap_sh_team_env(tmp_path):
    teamdir = tmp_path / "t"
    teamdir.mkdir()
    (teamdir / "bootstrap.env").write_text(
        "UV_DEFAULT_INDEX=https://nexus.corp/simple\nSETUP_WORK_UV_FROM_PIP=1\n"
        "evil key=1\nCORP_TOKEN=abc\n"
    )
    env = {
        "HOME": str(tmp_path / "h"),
        "PATH": "/usr/bin:/bin",
        "SETUP_WORK_TEAM_CONFIG_DIR": str(teamdir),
    }
    out = subprocess.run(
        ["sh", str(ROOT / "scripts" / "bootstrap.sh"), "--dry-run"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    assert "refusing secret-like key CORP_TOKEN" in out.stdout
    assert "pip install --user" in out.stdout and "uv==0.12.13" in out.stdout
    assert "re-resolved" in out.stdout
