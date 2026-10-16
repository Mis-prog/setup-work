"""Tool and engine installers (plan first, then execute).

Policy
------
* Corporate sources only: uv, Node.js and Git LFS archives and the llama.cpp / vLLM
  builds come from ``corporate.yaml`` (``tools`` / ``engines``; HDFS, NFS or internal
  HTTPS). Tool archives are checked against the sha256 pinned in ``defaults.yaml``.
  Nothing is fetched from public hosts; a missing mapping is an error with a hint.
* User-space first: everything is unpacked into ``data_dir`` and linked into
  ``bin_dir`` (default ``~/.local/bin``). No remote install scripts are run.
* System packages (git, curl, wget, compilers) are only installed through a
  package manager. With Homebrew this needs confirmation; with apt/dnf/... it needs
  root or ``--allow-sudo`` *and* confirmation. Otherwise setup-work prints the exact
  command for an administrator.
* Already-satisfied tools are skipped (idempotent). Shell startup files are never
  edited; if ``bin_dir`` is not on PATH we print the line to add.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config, EngineArtifact
from .diagnostics import Environment, detect_environment, detect_tool
from .errors import NotFoundError, SetupWorkError, UnsupportedError, UsageError
from .hardware import HardwareInfo
from .runner import Runner, probe, which
from .steps import Step

BOOTSTRAP_TOOLS = [
    "uv",
    "python",
    "pip",
    "node",
    "npm",
    "git",
    "git-lfs",
    "curl",
    "wget",
    "build-tools",
]
GROUPS = {
    "python": ["uv", "python"],
    "node": ["node", "npm"],
    "tools": ["git", "git-lfs", "curl", "wget", "build-tools"],
    "all": BOOTSTRAP_TOOLS,
}

SYSTEM_PACKAGES: dict[str, dict[str, list[str]]] = {
    "git": {
        "brew": ["git"],
        "apt-get": ["git"],
        "dnf": ["git"],
        "yum": ["git"],
        "pacman": ["git"],
        "zypper": ["git"],
        "apk": ["git"],
    },
    "curl": {
        "brew": ["curl"],
        "apt-get": ["curl", "ca-certificates"],
        "dnf": ["curl"],
        "yum": ["curl"],
        "pacman": ["curl"],
        "zypper": ["curl"],
        "apk": ["curl"],
    },
    "wget": {
        "brew": ["wget"],
        "apt-get": ["wget"],
        "dnf": ["wget"],
        "yum": ["wget"],
        "pacman": ["wget"],
        "zypper": ["wget"],
        "apk": ["wget"],
    },
    "build-tools": {
        "apt-get": ["build-essential", "cmake", "pkg-config"],
        "dnf": ["gcc", "gcc-c++", "make", "cmake"],
        "yum": ["gcc", "gcc-c++", "make", "cmake"],
        "pacman": ["base-devel", "cmake"],
        "zypper": ["gcc", "gcc-c++", "make", "cmake"],
        "apk": ["build-base", "cmake"],
        "brew": ["cmake"],
    },
}
_INSTALL_ARGV = {
    "brew": ["brew", "install"],
    "apt-get": ["apt-get", "install", "-y", "--no-install-recommends"],
    "dnf": ["dnf", "install", "-y"],
    "yum": ["yum", "install", "-y"],
    "pacman": ["pacman", "-S", "--needed", "--noconfirm"],
    "zypper": ["zypper", "--non-interactive", "install"],
    "apk": ["apk", "add"],
}


@dataclass
class ToolPlan:
    tool: str
    status: str  # satisfied | install | manual | unsupported
    steps: list[Step] = field(default_factory=list)
    detail: str = ""


def _platform() -> tuple[str, str]:
    system = platform.system()
    arch = platform.machine().lower()
    arch = {"aarch64": "arm64", "amd64": "x86_64"}.get(arch, arch)
    return system, arch


def _is_musl() -> bool:
    res = probe(["ldd", "--version"])
    return "musl" in (res.stdout + res.stderr).lower()


def uv_target() -> str:
    system, arch = _platform()
    a = {"arm64": "aarch64", "x86_64": "x86_64"}.get(arch)
    if a is None:
        raise UnsupportedError(f"no pinned uv build for architecture {arch}")
    if system == "Darwin":
        return f"{a}-apple-darwin"
    if system == "Linux":
        if _is_musl():
            raise UnsupportedError(
                "musl libc (e.g. Alpine) has no pinned uv build in defaults.yaml"
            )
        return f"{a}-unknown-linux-gnu"
    raise UnsupportedError(f"unsupported OS {system}")


def node_target() -> tuple[str, str]:
    system, arch = _platform()
    a = {"arm64": "arm64", "x86_64": "x64"}.get(arch)
    s = {"Darwin": "darwin", "Linux": "linux"}.get(system)
    if not a or not s:
        raise UnsupportedError(f"no pinned Node.js build for {system}/{arch}")
    return f"{s}-{a}", "tar.gz" if s == "darwin" else "tar.xz"


def git_lfs_target() -> tuple[str, str]:
    system, arch = _platform()
    a = {"arm64": "arm64", "x86_64": "amd64"}.get(arch)
    s = {"Darwin": "darwin", "Linux": "linux"}.get(system)
    if not a or not s:
        raise UnsupportedError(f"no pinned git-lfs build for {system}/{arch}")
    return f"{s}-{a}", "zip" if s == "darwin" else "tar.gz"


def _archive_steps(
    cfg: Config,
    tool: str,
    sha: str | None,
    archive_name: str,
    bins: dict[str, str],
    target: str = "",
) -> list[Step]:
    """fetch (corporate) -> extract -> link each `bins` entry (link name -> path inside)."""
    version = cfg.defaults.tools[tool].version
    corp = cfg.corporate.tools.get(tool)
    if corp is None:
        raise NotFoundError(
            f"no corporate source configured for {tool}",
            hint=f"add tools.{tool}.uri to corporate.yaml (see examples/corp/corporate.yaml)",
        )
    sha = corp.sha256 or sha
    if not sha:
        raise UnsupportedError(f"no pinned sha256 for {tool} target {target}")
    root = cfg.tools_dir / tool / version
    archive = cfg.tools_dir / "downloads" / archive_name
    uri = corp.uri.format(version=version, target=target, name=archive_name)
    ref = corp.model_copy(update={"uri": uri, "format": "file"})
    steps = [
        Step("fetch", f"fetch {tool} {version}", artifact=ref, dest=archive, sha256=sha, cfg=cfg),
        Step("extract", f"unpack {tool} into {root}", src=archive, dest=root),
    ]
    for link, rel in bins.items():
        steps.append(
            Step("link", f"link {link} into {cfg.bin_dir}", src=root / rel, dest=cfg.bin_dir / link)
        )
    return steps


def plan_uv(cfg: Config) -> list[Step]:
    pin = cfg.defaults.tools["uv"]
    target = uv_target()
    inner = f"uv-{target}"
    return _archive_steps(
        cfg,
        "uv",
        pin.sha256.get(target),
        f"uv-{pin.version}-{target}.tar.gz",
        {"uv": f"{inner}/uv", "uvx": f"{inner}/uvx"},
        target,
    )


def plan_node(cfg: Config) -> list[Step]:
    pin = cfg.defaults.tools["node"]
    target, ext = node_target()
    inner = f"node-v{pin.version}-{target}/bin"
    return _archive_steps(
        cfg,
        "node",
        pin.sha256.get(target),
        f"node-v{pin.version}-{target}.{ext}",
        {b: f"{inner}/{b}" for b in ("node", "npm", "npx", "corepack")},
        target,
    )


def plan_git_lfs(cfg: Config) -> list[Step]:
    pin = cfg.defaults.tools["git-lfs"]
    target, ext = git_lfs_target()
    steps = _archive_steps(
        cfg,
        "git-lfs",
        pin.sha256.get(target),
        f"git-lfs-{target}-v{pin.version}.{ext}",
        {"git-lfs": f"git-lfs-{pin.version}/git-lfs"},
        target,
    )
    steps.append(
        Step(
            "manual",
            "enable Git LFS hooks for your user (edits ~/.gitconfig, so not done automatically)",
            argv=["git", "lfs", "install", "--skip-repo"],
        )
    )
    steps[-1].description += ": git lfs install --skip-repo"
    return steps


def plan_python(cfg: Config) -> list[Step]:
    """Python is never downloaded (no corporate mirror): it must come with the machine."""
    version = cfg.defaults.tools.get("python")
    v = version.version if version else "3.12"
    return [
        Step(
            "manual",
            f"install Python {v} on this machine (system package or ask the admins); "
            "setup-work and uv do not download interpreters",
        )
    ]


def system_package_plan(env: Environment, tool: str, allow_sudo: bool) -> ToolPlan:
    pkgs_by_pm = SYSTEM_PACKAGES[tool]
    system, _ = _platform()
    if system == "Darwin" and tool == "build-tools":
        has_clt = probe(["xcode-select", "-p"]).ok
        if not has_clt:
            return ToolPlan(
                tool,
                "manual",
                [Step("manual", "install the Xcode Command Line Tools: xcode-select --install")],
                "Apple compilers come from the Command Line Tools (GUI installer)",
            )
    for pm in env.package_managers:
        if pm not in pkgs_by_pm or pm not in _INSTALL_ARGV:
            continue
        argv = _INSTALL_ARGV[pm] + pkgs_by_pm[pm]
        if pm == "brew":
            return ToolPlan(
                tool,
                "install",
                [Step("command", f"install {tool} with Homebrew", argv=argv, confirm=True)],
            )
        if env.is_root:
            pre = (
                [
                    Step(
                        "command",
                        "refresh package index",
                        argv=["apt-get", "update"],
                        privileged=True,
                    )
                ]
                if pm == "apt-get"
                else []
            )
            return ToolPlan(
                tool,
                "install",
                [*pre, Step("command", f"install {tool} with {pm}", argv=argv, privileged=True)],
            )
        sudo_cmd = ["sudo", *argv]
        if env.sudo_present and allow_sudo:
            pre = (
                [
                    Step(
                        "command",
                        "refresh package index",
                        argv=["sudo", "apt-get", "update"],
                        privileged=True,
                    )
                ]
                if pm == "apt-get"
                else []
            )
            return ToolPlan(
                tool,
                "install",
                [
                    *pre,
                    Step(
                        "command", f"install {tool} with sudo {pm}", argv=sudo_cmd, privileged=True
                    ),
                ],
            )
        hint = " ".join(sudo_cmd)
        detail = (
            "requires administrator rights; re-run with --allow-sudo (ALLOW_SUDO=1) or ask an admin"
            if env.sudo_present
            else "no sudo available: ask an administrator, or use a user-space package manager "
            "(e.g. micromamba/conda-forge)"
        )
        return ToolPlan(tool, "manual", [Step("manual", f"{tool}: {hint}")], detail)
    return ToolPlan(
        tool,
        "manual",
        [Step("manual", f"install {tool} with your system package manager")],
        "no supported package manager found",
    )


def plan_tools(
    cfg: Config, tools: list[str], *, allow_sudo: bool = False, force: bool = False
) -> list[ToolPlan]:
    for t in tools:
        if t not in BOOTSTRAP_TOOLS:
            raise UsageError(
                f"unknown tool {t!r}", hint=f"known tools: {', '.join(BOOTSTRAP_TOOLS)}"
            )
    env = detect_environment(
        cfg,
        [
            n
            for n in (
                "python",
                "uv",
                "pip",
                "node",
                "npm",
                "git",
                "git-lfs",
                "curl",
                "wget",
                "make",
                "cc",
            )
        ],
    )
    plans: list[ToolPlan] = []
    for t in tools:
        try:
            plans.append(_plan_one(cfg, env, t, allow_sudo, force))
        except NotFoundError as e:
            plans.append(ToolPlan(t, "manual", [Step("manual", e.hint or e.message)], e.message))
        except UnsupportedError as e:
            plans.append(ToolPlan(t, "unsupported", detail=e.message))
    return plans


def _plan_one(cfg: Config, env: Environment, t: str, allow_sudo: bool, force: bool) -> ToolPlan:
    if t == "build-tools":
        cc, mk = env.tool("cc"), env.tool("make")
        if cc and cc.installed and mk and mk.installed and not force:
            return ToolPlan(t, "satisfied", detail=f"cc {cc.version}, make {mk.version}")
        return system_package_plan(env, t, allow_sudo)
    st = env.tool(t) or detect_tool(t, cfg)
    if t == "uv":
        if st.installed and not force:
            return ToolPlan(t, "satisfied", detail=f"{st.version} at {st.path}")
        return ToolPlan(t, "install", plan_uv(cfg), "corporate archive, sha256-verified")
    if t == "python":
        if st.installed and st.satisfies and not force:
            return ToolPlan(t, "satisfied", detail=f"{st.version} at {st.path}")
        return ToolPlan(t, "manual", plan_python(cfg), "Python must be preinstalled")
    if t == "pip":
        if st.installed:
            return ToolPlan(t, "satisfied", detail=f"{st.version}")
        return ToolPlan(
            t,
            "manual",
            [
                Step(
                    "manual",
                    "pip is optional with uv (use `uv pip`); for a plain interpreter run "
                    "`python3 -m ensurepip --user`",
                )
            ],
            "uv replaces pip for setup-work",
        )
    if t in ("node", "npm"):
        node = env.tool("node")
        if node and node.installed and node.satisfies and not force:
            if t == "npm":
                npm = env.tool("npm")
                if npm and npm.installed:
                    return ToolPlan(t, "satisfied", detail=f"{npm.version}")
            else:
                return ToolPlan(t, "satisfied", detail=f"{node.version} at {node.path}")
        if t == "npm":
            return ToolPlan(t, "satisfied", detail="installed together with node")
        why = (
            f"found {node.version}, need >= {cfg.defaults.tools['node'].min_version}"
            if node and node.installed
            else "not installed"
        )
        return ToolPlan(t, "install", plan_node(cfg), f"{why}; corporate archive, sha256-verified")
    if t == "git-lfs":
        if st.installed and not force:
            return ToolPlan(t, "satisfied", detail=f"{st.version} at {st.path}")
        return ToolPlan(t, "install", plan_git_lfs(cfg), "corporate archive, sha256-verified")
    if t in ("git", "curl", "wget"):
        if st.installed and not force:
            return ToolPlan(t, "satisfied", detail=f"{st.version} at {st.path}")
        return system_package_plan(env, t, allow_sudo)
    raise UsageError(f"unknown tool {t}")


# ------------------------------------------------------------------ engines


def plan_engine(cfg: Config, engine: str, hw: HardwareInfo) -> tuple[list[Step], list[str]]:
    """Install an engine from its corporate artifact (corporate.yaml `engines`)."""
    if engine not in ("llama.cpp", "vllm"):
        raise UsageError(f"unknown engine {engine!r}", hint="llama.cpp or vllm")
    art = cfg.corporate.engines.get(engine)
    if art is None:
        raise NotFoundError(
            f"no corporate artifact configured for {engine}",
            hint=f"add engines['{engine}'].uri to corporate.yaml "
            "(see examples/corp/corporate.yaml)",
        )
    notes = [f"using corporate artifact {art.uri}"]
    if engine == "vllm" and (hw.os.system != "Linux" or hw.accelerator != "nvidia"):
        notes.append("this machine is not Linux + NVIDIA: vLLM will not be able to serve here")
    return plan_engine_artifact(cfg, engine, art), notes


def path_hint(cfg: Config) -> str:
    return f'export PATH="{cfg.bin_dir}:$PATH"'


# ------------------------------------------------------- corporate engine artifacts


def _activate_llama(cfg: Config, dest: Path, art: EngineArtifact) -> None:
    from .artifacts import find_one
    from .steps import ensure_executable, make_link

    binary = dest / art.path if art.path else find_one(dest, "llama-server")
    if binary is None or not binary.is_file():
        raise SetupWorkError(
            f"no llama-server found in {dest}",
            hint="set corporate.engines['llama.cpp'].path to the binary inside the archive",
        )
    for f in binary.parent.iterdir():
        if f.is_file() and f.name.startswith("llama-"):
            ensure_executable(f)
    make_link(binary.parent, cfg.engines_dir / "llama.cpp" / "current", cfg.data_dir)


def _read_pyvenv(cfgfile: Path) -> dict[str, str]:
    out = {}
    for line in cfgfile.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _python_works(py: Path) -> bool:
    return probe([str(py), "-c", "import sys; print(sys.prefix)"], timeout=60).ok


def _find_interpreter(version: str, home: str | None, cfg: Config) -> Path | None:
    cands: list[Path] = []
    if home:
        cands += [Path(home) / f"python{version}", Path(home) / "python3"]
    uv = which("uv", [cfg.bin_dir])
    if uv:
        res = probe([uv, "python", "find", "--no-project", version], timeout=30)
        if res.ok and res.stdout.strip():
            cands.append(Path(res.stdout.strip().splitlines()[0]))
    w = which(f"python{version}")
    if w:
        cands.append(Path(w))
    for c in cands:
        if c.is_file():
            res = probe([str(c), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"])
            if res.ok and res.stdout.strip() == version:
                return c.resolve()
    return None


def repair_venv(venv: Path, cfg: Config, runner: Runner) -> list[str]:
    """Make a venv unpacked from a plain zip usable on this machine.

    Fixes the base interpreter (pyvenv.cfg `home` + bin/python* links) and rewrites
    absolute shebangs of bin/ scripts to the new location. Returns notes.
    """
    notes: list[str] = []
    pv = venv / "pyvenv.cfg"
    if not pv.is_file():
        raise SetupWorkError(f"{venv} is not a virtualenv (no pyvenv.cfg)")
    meta = _read_pyvenv(pv)
    full = meta.get("version_info") or meta.get("version") or ""
    version = ".".join(full.split(".")[:2])
    if not version:
        raise SetupWorkError(f"cannot read the Python version from {pv}")
    py = venv / "bin" / "python"
    if not _python_works(py):
        interp = _find_interpreter(version, meta.get("home"), cfg)
        if interp is None:
            raise UnsupportedError(
                f"the vLLM venv was built with Python {version} (home={meta.get('home')}), "
                "which is not installed on this machine",
                hint=f"install Python {version} here (system package / admins) or rebuild the "
                "venv zip with the Python version the machines have",
            )
        lines = pv.read_text(encoding="utf-8").splitlines()
        lines = [
            f"home = {interp.parent}" if ln.split("=")[0].strip() == "home" else ln for ln in lines
        ]
        pv.write_text("\n".join(lines) + "\n", encoding="utf-8")
        for name in ("python", "python3", f"python{version}"):
            link = venv / "bin" / name
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(interp)
        notes.append(f"re-linked venv interpreter to {interp}")
        if not _python_works(py):
            raise SetupWorkError(f"{py} still does not start after re-linking to {interp}")
    rewritten = 0
    new_shebang = f"#!{venv / 'bin' / 'python'}"
    for f in (venv / "bin").iterdir():
        if not f.is_file() or f.is_symlink() or f.stat().st_size > 1 << 20:
            continue
        try:
            head = f.read_bytes()[:2]
            if head != b"#!":
                continue
            text = f.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        first, _, rest = text.partition("\n")
        if "/bin/python" in first and first.strip() != new_shebang:
            f.write_text(new_shebang + "\n" + rest, encoding="utf-8")
            rewritten += 1
    if rewritten:
        notes.append(f"rewrote {rewritten} script shebang(s) in {venv / 'bin'}")
    return notes


def _activate_vllm(cfg: Config, dest: Path, art: EngineArtifact, runner: Runner) -> None:
    from .artifacts import find_one
    from .steps import make_link

    if art.path:
        venv = dest / art.path
    else:
        marker = find_one(dest, "pyvenv.cfg")
        venv = marker.parent if marker else dest
    for note in repair_venv(venv, cfg, runner):
        runner.info(f"  {note}")
    make_link(venv, cfg.engines_dir / "vllm" / "current", cfg.data_dir)


def plan_engine_artifact(cfg: Config, engine: str, art: EngineArtifact) -> list[Step]:
    if engine == "llama.cpp":
        dest = cfg.engines_dir / "llama.cpp" / "corporate"
        return [
            Step("artifact", "llama.cpp (corporate build)", artifact=art, dest=dest, cfg=cfg),
            Step(
                "call",
                "locate llama-server and point engines/llama.cpp/current at it",
                action=lambda r: _activate_llama(cfg, dest, art),
            ),
            Step(
                "command",
                "verify llama-server runs",
                argv=[str(cfg.engines_dir / "llama.cpp" / "current" / "llama-server"), "--version"],
            ),
        ]
    dest = cfg.engines_dir / "vllm" / "corporate"
    py = cfg.engines_dir / "vllm" / "current" / "bin" / "python"
    return [
        Step("artifact", "vLLM venv (corporate)", artifact=art, dest=dest, cfg=cfg),
        Step(
            "call",
            "repair the unpacked venv and point engines/vllm/current at it",
            action=lambda r: _activate_vllm(cfg, dest, art, r),
        ),
        Step(
            "command",
            "verify vLLM imports",
            argv=[str(py), "-c", "import vllm; print('vllm', vllm.__version__)"],
            timeout=900,
        ),
    ]
