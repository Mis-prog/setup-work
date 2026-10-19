"""Установка и обслуживание харнессов агентов по декларативным рецептам.

Каждый харнесс находится в <data_dir>/harnesses/<name>, команды доступны через
ссылки в bin_dir. Глобальная установка пакетов не используется.
Git использует существующие учётные данные без интерактивных запросов.
Сборка выполняет код проекта и требует разрешения согласно настройкам доверия."""

from __future__ import annotations

import datetime as _dt
import json
import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import BinarySpec, Config, HarnessSpec
from .diagnostics import detect_tool, version_satisfies
from .errors import MissingDependencyError, NotFoundError, UsageError
from .runner import Runner, probe, which
from .steps import Step, ensure_executable

MANIFEST = ".setup-work.json"
_GIT_ENV = {"GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "", "SSH_ASKPASS": ""}


def get_spec(cfg: Config, name: str) -> HarnessSpec:
    specs = cfg.harnesses.harnesses
    if name not in specs:
        raise NotFoundError(
            f"неизвестный харнесс {name!r}",
            hint=f"known: {', '.join(sorted(specs))}; добавьте рецепт в "
            f"{cfg.user_dir / 'harnesses.yaml'}(пример: examples/corp/harnesses.yaml)",
        )
    return specs[name]


def harness_root(cfg: Config, name: str) -> Path:
    return cfg.harness_dir / name


def read_manifest(cfg: Config, name: str) -> dict[str, Any] | None:
    try:
        return json.loads((harness_root(cfg, name) / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def checkout_dir(cfg: Config, name: str, spec: HarnessSpec) -> Path:
    if spec.source == "git":
        return harness_root(cfg, name) / "src"
    if spec.source == "local":
        return Path(spec.path or "").expanduser().resolve()
    return harness_root(cfg, name)


def binary_file(cfg: Config, name: str, spec: HarnessSpec, b: BinarySpec) -> Path:
    """Путь к настоящему исполняемому файлу до обёртки."""
    if b.path:
        return checkout_dir(cfg, name, spec) / b.path
    return binary_target(cfg, name, spec, b.name)


def link_target(cfg: Config, name: str, spec: HarnessSpec, b: BinarySpec) -> Path:
    """Цель ссылки bin_dir/<name>: при заданном интерпретаторе используется обёртка."""
    if b.interpreter:
        return harness_root(cfg, name) / "bin" / b.name
    return binary_file(cfg, name, spec, b)


def _interpreter(cfg: Config, name: str, spec: HarnessSpec, kind: str) -> str:
    if kind == "sh":
        return "/bin/sh"
    if kind == "python" and (spec.source == "pypi" or spec.install.method == "uv"):
        return str(harness_root(cfg, name) / ".venv" / "bin" / "python")
    exe = {"node": "node", "python": "python3", "bun": "bun"}[kind]
    return which(exe, [cfg.bin_dir]) or exe


def wrapper_script(interpreter: str, target: Path, harness: str) -> str:
    return (
        "#!/bin/sh\n"
        f"# создано setup-work для харнесса {harness}; не редактируйте вручную\n"
        f'exec {shlex.quote(interpreter)} {shlex.quote(str(target))} "$@"\n'
    )


def binary_target(cfg: Config, name: str, spec: HarnessSpec, binary: str) -> Path:
    root = harness_root(cfg, name)
    if spec.source in ("npm",) or (spec.install.method == "npm"):
        return root / "node_modules" / ".bin" / binary
    if spec.source == "pypi" or spec.install.method == "uv":
        return root / ".venv" / "bin" / binary
    src = root / "src" if spec.source == "git" else Path(spec.path or "").expanduser()
    return src / binary


@dataclass
class HarnessPlan:
    name: str
    spec: HarnessSpec
    steps: list[Step] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    needs_confirmation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.spec.source,
            "version": self.spec.version or self.spec.ref,
            "steps": [s.to_dict() for s in self.steps],
            "warnings": self.warnings,
        }


def _check_requirements(cfg: Config, spec: HarnessSpec) -> list[str]:
    problems = []
    if spec.source == "npm" or spec.install.method == "npm":
        node = detect_tool("node", cfg)
        npm = detect_tool("npm", cfg)
        if not node.installed or not npm.installed:
            problems.append("Node.js/npm не найдены; выполните make bootstrap")
        elif "node" in spec.requires and not version_satisfies(node.version, spec.requires["node"]):
            problems.append(
                f"нужен Node.js {spec.requires['node']}, найден {node.version}; make bootstrap"
            )
    if (spec.source == "pypi" or spec.install.method == "uv") and not which("uv", [cfg.bin_dir]):
        problems.append("uv не найден; выполните make bootstrap")
    if spec.source == "git" and not which("git"):
        problems.append("git не найден; установите его на этой машине")
    for cmd in spec.requires_commands:
        if not which(cmd, [cfg.bin_dir]):
            problems.append(f"обязательная команда {cmd!r} не найдена в PATH")
    return problems


def plan_install(
    cfg: Config,
    name: str,
    *,
    version: str | None = None,
    ref: str | None = None,
    update: bool = False,
    allow_build: bool = False,
) -> HarnessPlan:
    spec = get_spec(cfg, name).model_copy(deep=True)
    if version:
        if spec.source not in ("npm", "pypi"):
            raise UsageError("VERSION применяется к пакетам npm/PyPI; для Git укажите REF")
        spec.version = version
    if ref:
        if spec.source != "git":
            raise UsageError("REF применяется только к Git-рецептам")
        spec.ref = ref
    spec = HarnessSpec.model_validate(spec.model_dump())  # повторная проверка переопределений
    if "CHANGE-ME" in f"{spec.ref} {spec.repository} {spec.path}":
        sources = cfg.sources.get("harnesses.yaml") or [cfg.builtin_dir / "harnesses.yaml"]
        raise UsageError(
            f"рецепт {name} содержит незаполненный CHANGE-ME",
            hint=f"заполните реальные repository и ref в {sources[-1]}",
        )
    root = harness_root(cfg, name)
    plan = HarnessPlan(name, spec)
    if not spec.pinned:
        plan.warnings.append("в рецепте не закреплена точная версия, тег или коммит")
    problems = _check_requirements(cfg, spec)
    if problems:
        raise MissingDependencyError(f"не удалось установить {name}: " + "; ".join(problems))

    existing = read_manifest(cfg, name)
    if existing and not update:
        same = existing.get("version") == (spec.version or spec.ref)
        if same:
            plan.warnings.append(f"{name} {existing.get('version')} уже установлен")
            return plan

    uv = which("uv", [cfg.bin_dir]) or "uv"
    npm = which("npm", [cfg.bin_dir]) or "npm"
    git = which("git") or "git"
    s = plan.steps

    if spec.source == "npm":
        s.append(
            Step(
                "command",
                f"установить {spec.package}@{spec.version} в {root}",
                argv=[
                    npm,
                    "install",
                    "--prefix",
                    str(root),
                    "--no-fund",
                    "--no-audit",
                    "--save-exact",
                    f"{spec.package}@{spec.version}",
                ],
                timeout=1800,
            )
        )
    elif spec.source == "pypi":
        s += _uv_env_steps(uv, root, spec)
        s.append(
            Step(
                "command",
                f"установить {spec.package}=={spec.version}",
                argv=[
                    uv,
                    "pip",
                    "install",
                    "--python",
                    str(root / ".venv" / "bin" / "python"),
                    f"{spec.package}=={spec.version}",
                ],
                timeout=1800,
            )
        )
    else:
        if spec.source == "git":
            src = root / "src"
            if not (spec.repository and spec.ref):  # гарантируется проверкой HarnessSpec
                raise UsageError(
                    f"Git-харнесс {name} требует repository и ref в config/harnesses.yaml"
                )
            if not (src / ".git").is_dir():
                s.append(
                    Step(
                        "command",
                        f"клонировать {spec.repository}",
                        argv=[
                            git,
                            "clone",
                            "--no-checkout",
                            "--filter=blob:none",
                            "--",
                            spec.repository,
                            str(src),
                        ],
                        env=_GIT_ENV,
                        timeout=900,
                    )
                )
            s.append(
                Step(
                    "command",
                    f"загрузить {spec.ref}",
                    argv=[git, "-C", str(src), "fetch", "--tags", "origin", "--", spec.ref],
                    env=_GIT_ENV,
                    timeout=900,
                )
            )
            s.append(
                Step(
                    "command",
                    f"переключить на {spec.ref}",
                    argv=[
                        git,
                        "-C",
                        str(src),
                        "-c",
                        "advice.detachedHead=false",
                        "checkout",
                        "--detach",
                        "FETCH_HEAD",
                    ],
                )
            )
        else:
            src = Path(spec.path or "").expanduser().resolve()
            if not src.is_dir():
                raise NotFoundError(f"каталог {src} не найден; исправьте path в harnesses.yaml")
        plan.needs_confirmation = True
        plan.warnings.append(
            f"установка из {'Git' if spec.source == 'git' else 'локального каталога'} "
            "запускает код сборки самого проекта"
        )
        builds = spec.install.build_steps
        if builds:
            trusted = bool(
                spec.git_host
                and spec.git_host in {h.lower() for h in cfg.corporate.trusted_git_hosts}
            )
            if not (allow_build or trusted):
                raise UsageError(
                    f"рецепт {name} содержит команды сборки {builds}",
                    hint="проверьте команды, укажите --allow-build (ALLOW_BUILD=1) либо добавьте "
                    "хост Git в trusted_git_hosts файла corporate.yaml",
                )
            if trusted:
                plan.warnings.append(
                    f"команды сборки разрешены: {spec.git_host} является доверенным хостом"
                )
            for i, argv in enumerate(builds, 1):
                s.append(
                    Step(
                        "command",
                        f"шаг сборки {i}/{len(builds)}",
                        argv=list(argv),
                        cwd=src,
                        timeout=3600,
                    )
                )
        if spec.install.method == "uv":
            s += _uv_env_steps(uv, root, spec)
            s.append(
                Step(
                    "command",
                    f"установить {src} в окружение харнесса",
                    argv=[
                        uv,
                        "pip",
                        "install",
                        "--python",
                        str(root / ".venv" / "bin" / "python"),
                        str(src),
                    ],
                    timeout=1800,
                )
            )
        elif spec.install.method == "npm":
            s.append(
                Step(
                    "command",
                    f"установить {src} через npm",
                    argv=[
                        npm,
                        "install",
                        "--prefix",
                        str(root),
                        "--no-fund",
                        "--no-audit",
                        "--install-links",
                        str(src),
                    ],
                    timeout=1800,
                )
            )

    for b in spec.binary_specs:
        real = binary_file(cfg, name, spec, b)
        if b.interpreter:
            interp = _interpreter(cfg, name, spec, b.interpreter)
            s.append(
                Step(
                    "write",
                    f"создать обёртку {b.name} ({b.interpreter} {b.path or b.name})",
                    dest=link_target(cfg, name, spec, b),
                    content=wrapper_script(interp, real, name),
                    mode=0o755,
                )
            )
        elif b.path:
            s.append(
                Step(
                    "call",
                    f"дать право исполнения {real}",
                    action=lambda r, f=real: ensure_executable(f),
                )
            )
        s.append(
            Step(
                "link",
                f"создать ссылку {b.name} в {cfg.bin_dir}",
                src=link_target(cfg, name, spec, b),
                dest=cfg.bin_dir / b.name,
            )
        )
    return plan


def _uv_env_steps(uv: str, root: Path, spec: HarnessSpec) -> list[Step]:
    py = spec.install.python or "3.12"
    venv = root / ".venv"
    return [
        Step(
            "command",
            f"создать окружение Python {py} environment",
            argv=[uv, "venv", "--python", py, str(venv)],
            when=lambda: not (venv / "bin" / "python").exists(),
        )
    ]


def write_manifest(cfg: Config, name: str, spec: HarnessSpec, runner: Runner) -> None:
    if runner.dry_run:
        return
    root = harness_root(cfg, name)
    commit = None
    if spec.source == "git":
        res = probe(["git", "-C", str(root / "src"), "rev-parse", "HEAD"])
        commit = res.stdout.strip() if res.ok else None
    data = {
        "name": name,
        "source": spec.source,
        "package": spec.package,
        "version": spec.version or spec.ref,
        "commit": commit,
        "binaries": {b.name: str(cfg.bin_dir / b.name) for b in spec.binary_specs},
        "links": {
            str(cfg.bin_dir / b.name): str(link_target(cfg, name, spec, b))
            for b in spec.binary_specs
        },
        "installed_at": _dt.datetime.now(_dt.UTC).isoformat(),
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / MANIFEST).write_text(json.dumps(data, indent=2), encoding="utf-8")


def plan_remove(cfg: Config, name: str) -> list[Step]:
    root = harness_root(cfg, name)
    manifest = read_manifest(cfg, name)
    if manifest is None and not root.exists():
        raise NotFoundError(f"{name} не установлен", hint="выполните make harness-list")
    steps: list[Step] = []
    for link, target in (manifest or {}).get("links", {}).items():
        p = Path(link)
        # Только ссылки, чья цель совпадает с созданной setup-work.
        if p.is_symlink() and os.readlink(p) == target:
            steps.append(Step("remove", f"удалить ссылку {p.name}", dest=p))
    steps.append(Step("remove", f"удалить {root}", dest=root, destructive=True))
    return steps


@dataclass
class HarnessCheck:
    name: str
    ok: bool
    checks: list[tuple[str, bool, str]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "checks": [{"check": c, "ok": o, "detail": d} for c, o, d in self.checks],
        }


def doctor(cfg: Config, name: str) -> HarnessCheck:
    spec = get_spec(cfg, name)
    checks: list[tuple[str, bool, str]] = [("рецепт корректен", True, spec.source)]
    checks.append(("версия закреплена", spec.pinned, str(spec.version or spec.ref or spec.path)))
    manifest = read_manifest(cfg, name)
    checks.append(
        (
            "установлен через setup-work",
            manifest is not None,
            f"{manifest.get('version')} at {harness_root(cfg, name)}"
            if manifest
            else "выполните make harness-install NAME=" + name,
        )
    )
    if manifest and manifest.get("version") != (spec.version or spec.ref):
        checks.append(
            (
                "версия совпадает с рецептом",
                False,
                f"установлено {manifest.get('version')}, в рецепте {spec.version or spec.ref}",
            )
        )
    for problem in _check_requirements(cfg, spec):
        checks.append(("требование окружения", False, problem))
    for bs in spec.binary_specs:
        b = bs.name
        target = link_target(cfg, name, spec, bs)
        real = binary_file(cfg, name, spec, bs)
        link = cfg.bin_dir / b
        resolved = which(b)
        checks.append((f"команда {b}", real.exists() and target.exists(), str(real)))
        if resolved and Path(resolved).resolve() != target.resolve() and link.exists():
            checks.append(
                (f"{b} в PATH", False, f"PATH для {b} указывает на {resolved}, ожидается {link}")
            )
        elif not resolved:
            checks.append((f"{b} в PATH", False, f"добавьте {cfg.bin_dir} в PATH"))
        else:
            checks.append((f"{b} в PATH", True, resolved))
    if spec.verify and manifest:
        exe = (
            str(cfg.bin_dir / spec.verify[0])
            if (cfg.bin_dir / spec.verify[0]).exists()
            else spec.verify[0]
        )
        res = probe([exe, *spec.verify[1:]], timeout=60)
        out = (res.stdout or res.stderr).strip().splitlines()
        checks.append(
            ("проверочная команда", res.ok, out[0][:100] if out else f"код {res.returncode}")
        )
    for var in spec.environment.required:
        checks.append(
            (
                f"переменная {var}",
                bool(os.environ.get(var)),
                "set" if os.environ.get(var) else "отсутствует обязательное значение",
            )
        )
    for var in spec.environment.optional:
        if os.environ.get(var):
            checks.append((f"переменная {var}", True, "set"))
    ok = all(o for c, o, _ in checks if c != "версия закреплена")
    return HarnessCheck(name, ok, checks)


def list_harnesses(cfg: Config) -> list[dict[str, Any]]:
    out = []
    for name, spec in sorted(cfg.harnesses.harnesses.items()):
        m = read_manifest(cfg, name)
        out.append(
            {
                "name": name,
                "source": spec.source,
                "package": spec.package or spec.repository or spec.path,
                "version": spec.version or spec.ref,
                "pinned": spec.pinned,
                "installed": m.get("version") if m else None,
                "binaries": [b.name for b in spec.binary_specs],
                "description": spec.description,
            }
        )
    return out
