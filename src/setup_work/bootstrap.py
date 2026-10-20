"""Установка инструментов и движков: сначала план, затем выполнение.

Node.js и npm берутся из корпоративного npm registry.
Источники движков задаются в engines. Python должен быть
установлен заранее. Файлы распаковываются в data_dir, ссылки создаются в bin_dir.
Системные пакеты, sudo и профили shell не используются."""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config, EngineArtifact
from .diagnostics import Environment, detect_environment, detect_tool
from .errors import NotFoundError, SetupWorkError, UnsupportedError, UsageError
from .hardware import HardwareInfo
from .runner import Runner, probe, which
from .steps import Step

BOOTSTRAP_TOOLS = ["python", "node"]
AVAILABLE_TOOLS = [*BOOTSTRAP_TOOLS, "uv"]


@dataclass
class ToolPlan:
    tool: str
    status: str  # Статусы: satisfied, install, manual, unsupported.
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


def node_target() -> tuple[str, str]:
    system, arch = _platform()
    a = {"arm64": "arm64", "x86_64": "x64"}.get(arch)
    s = {"Darwin": "darwin", "Linux": "linux"}.get(system)
    if not a or not s:
        raise UnsupportedError(f"нет закреплённой сборки Node.js для {system}/{arch}")
    return f"{s}-{a}", "tar.gz" if s == "darwin" else "tar.xz"


def _archive_steps(
    cfg: Config,
    tool: str,
    archive_name: str,
    bins: dict[str, str],
    target: str = "",
) -> list[Step]:
    """Загрузить корпоративный архив, распаковать и создать ссылки для записей bins."""
    version = cfg.defaults.tools[tool].version
    corp = cfg.corporate.tools.get(tool)
    if corp is None:
        raise NotFoundError(
            f"не задан корпоративный источник для {tool}",
            hint=f"заполните tools.{tool}.uri в config/corporate.yaml",
        )
    root = cfg.tools_dir / tool / version
    archive = cfg.tools_dir / "downloads" / archive_name
    uri = corp.uri.format(version=version, target=target, name=archive_name)
    ref = corp.model_copy(update={"uri": uri, "format": "file"})
    steps = [
        Step("fetch", f"загрузить {tool} {version}", artifact=ref, dest=archive, cfg=cfg),
        Step("extract", f"распаковать {tool} в {root}", src=archive, dest=root),
    ]
    for link, rel in bins.items():
        steps.append(
            Step(
                "link",
                f"создать ссылку {link} в {cfg.bin_dir}",
                src=root / rel,
                dest=cfg.bin_dir / link,
            )
        )
    return steps


def plan_node(cfg: Config) -> list[Step]:
    pin = cfg.defaults.tools["node"]
    target, ext = node_target()
    if "node" not in cfg.corporate.tools:
        from . import node_registry

        if target.startswith("linux-") and _is_musl():
            raise UnsupportedError("для установки Node.js из npm требуется Linux с glibc")
        node_version, npm_version = node_registry.versions(cfg)
        registry = node_registry.read_registry(cfg)
        package = "node-bin-darwin-arm64" if target == "darwin-arm64" else f"node-{target}"
        return [
            Step(
                "call",
                f"установить {package}@{node_version} и npm@{npm_version} из {registry.url}",
                action=lambda runner: node_registry.install(cfg, registry, package, target, runner),
            )
        ]
    inner = f"node-v{pin.version}-{target}/bin"
    return _archive_steps(
        cfg,
        "node",
        f"node-v{pin.version}-{target}.{ext}",
        {b: f"{inner}/{b}" for b in ("node", "npm", "npx", "corepack")},
        target,
    )


def plan_python(cfg: Config) -> list[Step]:
    """Python не скачивается: он должен быть заранее установлен на машине."""
    version = cfg.defaults.tools.get("python")
    v = version.version if version else "3.12"
    return [
        Step(
            "manual",
            f"установите Python {v} через системный пакет или администратора; "
            "setup-work не скачивает интерпретаторы",
        )
    ]


def plan_uv(cfg: Config) -> list[Step]:
    """Необязательный uv устанавливается через корпоративный pip в отдельный venv."""
    from .corporate import pip_index

    if not pip_index():
        raise NotFoundError("для uv настройте корпоративный PyPI через make registries")
    version = cfg.defaults.tools["uv"].version
    root = cfg.tools_dir / "uv" / version
    python = root / "bin" / "python"
    return [
        Step(
            "command",
            "создать окружение инструмента uv",
            argv=[sys.executable, "-m", "venv", str(root)],
            when=lambda: not python.exists(),
        ),
        Step(
            "command",
            f"установить uv {version} из корпоративного PyPI",
            argv=[
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--only-binary=:all:",
                f"uv=={version}",
            ],
        ),
        *[
            Step("link", f"создать ссылку {name}", src=root / "bin" / name, dest=cfg.bin_dir / name)
            for name in ("uv", "uvx")
        ],
    ]


def plan_tools(cfg: Config, tools: list[str], *, force: bool = False) -> list[ToolPlan]:
    for t in tools:
        if t not in AVAILABLE_TOOLS:
            raise UsageError(
                f"неизвестный инструмент {t!r}",
                hint=f"доступные инструменты: {', '.join(AVAILABLE_TOOLS)}",
            )
    env = detect_environment(cfg, ["python", "node", "npm", "uv"])
    plans: list[ToolPlan] = []
    for t in tools:
        try:
            plans.append(_plan_one(cfg, env, t, force))
        except NotFoundError as e:
            plans.append(ToolPlan(t, "manual", [Step("manual", e.hint or e.message)], e.message))
        except UnsupportedError as e:
            plans.append(ToolPlan(t, "unsupported", detail=e.message))
    return plans


def _plan_one(cfg: Config, env: Environment, t: str, force: bool) -> ToolPlan:
    st = env.tool(t) or detect_tool(t, cfg)
    if t == "uv":
        if st.installed and st.version == cfg.defaults.tools["uv"].version and not force:
            return ToolPlan(t, "satisfied", detail=f"{st.version} по пути {st.path}")
        return ToolPlan(t, "install", plan_uv(cfg))
    if t == "python":
        if st.installed and st.satisfies and not force:
            return ToolPlan(t, "satisfied", detail=f"{st.version} по пути {st.path}")
        return ToolPlan(t, "manual", plan_python(cfg), "Python должен быть установлен заранее")
    if t == "node":
        node = env.tool("node")
        npm = env.tool("npm")
        if node and node.installed and node.satisfies and npm and npm.installed and not force:
            return ToolPlan(t, "satisfied", detail=f"{node.version} по пути {node.path}")
        if force:
            why = "принудительная установка закреплённых версий"
        elif node and node.installed and node.satisfies:
            why = "Node.js найден, но npm отсутствует или не запускается"
        elif node and node.installed:
            why = (
                f"найдена версия {node.version}, требуется >= "
                f"{cfg.defaults.tools['node'].min_version}"
            )
        else:
            why = "Node.js не установлен"
        return ToolPlan(
            t,
            "install",
            plan_node(cfg),
            f"{why}; установка Node.js и npm из корпоративного реестра",
        )
    raise UsageError(f"неизвестный инструмент {t}")


# ------------------------------------------------------------------ движки


def plan_engine(cfg: Config, engine: str, hw: HardwareInfo) -> tuple[list[Step], list[str]]:
    """Установить движок из артефакта, указанного в corporate.yaml: engines."""
    if engine not in ("llama.cpp", "vllm"):
        raise UsageError(f"неизвестный движок {engine!r}", hint="укажите llama.cpp или vllm")
    art = cfg.corporate.engines.get(engine)
    if art is None:
        raise NotFoundError(
            f"не задан корпоративный артефакт для {engine}",
            hint=f"заполните engines['{engine}'].uri в config/corporate.yaml",
        )
    notes = [f"используется корпоративный артефакт {art.uri}"]
    if engine == "vllm" and (hw.os.system != "Linux" or hw.accelerator != "nvidia"):
        notes.append(
            "для управляемого запуска vLLM нужна Linux-машина с "
            "NVIDIA GPU; на этой машине используйте подходящий движок"
        )
    return plan_engine_artifact(cfg, engine, art), notes


def path_hint(cfg: Config) -> str:
    return f'export PATH="{cfg.bin_dir}:$PATH"'


# ------------------------------------------------------- корпоративные артефакты движков


def _activate_llama(cfg: Config, dest: Path, art: EngineArtifact) -> None:
    from .artifacts import find_one
    from .steps import ensure_executable, make_link

    binary = dest / art.path if art.path else find_one(dest, "llama-server")
    if binary is None or not binary.is_file():
        raise SetupWorkError(
            f"llama-server не найден в {dest}",
            hint=(
                "задайте engines.llama.cpp.path в config/corporate.yaml: "
                "путь к llama-server относительно корня архива"
            ),
        )
    for f in binary.parent.iterdir():
        if f.is_file() and f.name.startswith("llama-"):
            ensure_executable(f)
    make_link(binary.parent, cfg.engines_dir / "llama.cpp" / "current", cfg.runtime_dir)


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
    for name in (f"python{version}", "python3"):
        w = which(name, [cfg.bin_dir])
        if w:
            cands.append(Path(w))
    for c in cands:
        if c.is_file():
            res = probe([str(c), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"])
            if res.ok and res.stdout.strip() == version:
                return c.resolve()
    return None


def repair_venv(venv: Path, cfg: Config, runner: Runner) -> list[str]:
    """Подготовить распакованное venv к работе на этой машине.

    Исправить home в pyvenv.cfg, ссылки bin/python* и строки запуска скриптов bin/."""
    notes: list[str] = []
    pv = venv / "pyvenv.cfg"
    if not pv.is_file():
        raise SetupWorkError(
            f"{venv}: нет pyvenv.cfg; проверьте engines.vllm.path в corporate.yaml"
        )
    meta = _read_pyvenv(pv)
    full = meta.get("version_info") or meta.get("version") or ""
    version = ".".join(full.split(".")[:2])
    if not version:
        raise SetupWorkError(f"не удалось определить версию Python по файлу {pv}")
    py = venv / "bin" / "python"
    if not _python_works(py):
        interp = _find_interpreter(version, meta.get("home"), cfg)
        if interp is None:
            raise UnsupportedError(
                f"окружение vLLM собрано с Python {version} (home={meta.get('home')}), "
                "который не установлен на этой машине",
                hint=f"установите Python {version} через администратора либо пересоберите "
                "архив venv под установленную версию Python",
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
        notes.append(f"интерпретатор venv перепривязан к {interp}")
        if not _python_works(py):
            raise SetupWorkError(f"{py} по-прежнему не запускается после перепривязки к {interp}")
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
        notes.append(f"исправлено {rewritten} строк запуска скриптов в {venv / 'bin'}")
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
    make_link(venv, cfg.engines_dir / "vllm" / "current", cfg.runtime_dir)


def plan_engine_artifact(cfg: Config, engine: str, art: EngineArtifact) -> list[Step]:
    if engine == "llama.cpp":
        dest = cfg.engines_dir / "llama.cpp" / "corporate"
        return [
            Step("artifact", "llama.cpp (корпоративная сборка)", artifact=art, dest=dest, cfg=cfg),
            Step(
                "call",
                "найти llama-server и создать ссылку engines/llama.cpp/current",
                action=lambda r: _activate_llama(cfg, dest, art),
            ),
            Step(
                "command",
                "проверить запуск llama-server",
                argv=[str(cfg.engines_dir / "llama.cpp" / "current" / "llama-server"), "--version"],
            ),
        ]
    dest = cfg.engines_dir / "vllm" / "corporate"
    py = cfg.engines_dir / "vllm" / "current" / "bin" / "python"
    return [
        Step("artifact", "venv vLLM из корпоративного архива", artifact=art, dest=dest, cfg=cfg),
        Step(
            "call",
            "исправить пути venv и создать ссылку engines/vllm/current",
            action=lambda r: _activate_vllm(cfg, dest, art, r),
        ),
        Step(
            "command",
            "проверить импорт vLLM",
            argv=[str(py), "-c", "import vllm; print('vllm', vllm.__version__)"],
            timeout=900,
        ),
    ]
