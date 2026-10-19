"""План установки и его выполнение.

План можно вывести через --dry-run или --json. Шаги: command — команда argv,
download — загрузка с SHA256, extract — безопасная распаковка, link — ссылка,
manual — инструкция для ручного выполнения. Произвольный код shell не используется."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tarfile
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .errors import SetupWorkError
from .runner import Runner, format_argv

StepKind = Literal[
    "command",
    "extract",
    "link",
    "manual",
    "remove",
    "write",
    "artifact",
    "fetch",
    "call",
]


@dataclass
class Step:
    kind: StepKind
    description: str
    argv: list[str] | None = None
    cwd: Path | None = None
    env: dict[str, str] | None = None
    sha256: str | None = None
    src: Path | None = None
    dest: Path | None = None
    content: str | None = None
    privileged: bool = False
    confirm: bool = False
    destructive: bool = False
    timeout: float | None = None
    when: Callable[[], bool] | None = field(default=None, repr=False)
    artifact: Any = field(default=None, repr=False)  # config.ArtifactRef для шагов artifact/fetch
    action: Callable[[Runner], None] | None = field(default=None, repr=False)  # для шага call
    mode: int | None = None  # права файла для шага write
    cfg: Any = field(default=None, repr=False)  # config.Config для шагов artifact/fetch

    def render(self) -> str:
        if self.kind == "command" and self.argv:
            return format_argv(self.argv) + (f"   (в {self.cwd})" if self.cwd else "")
        if self.kind == "extract":
            return f"распаковать {self.src} -> {self.dest}"
        if self.kind == "link":
            return f"ln -sfn {self.src} {self.dest}"
        if self.kind == "remove":
            return f"удалить {self.dest}"
        if self.kind == "write":
            return f"записать {self.dest}"
        if self.kind in ("artifact", "fetch") and self.artifact is not None:
            return f"загрузить {self.artifact.uri} -> {self.dest}"
        return self.description

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "description": self.description,
            "action": self.render(),
            "privileged": self.privileged,
            "destructive": self.destructive,
        }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_verified(url: str, dest: Path, sha256: str, timeout: float = 120) -> Path:
    """Загрузить с учётом прокси и проверить перед перемещением на место."""
    if not url.startswith("https://"):
        raise SetupWorkError(f"загрузка разрешена только по HTTPS: {url}")
    if dest.exists() and sha256_file(dest) == sha256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=dest.name + ".", suffix=".part", dir=dest.parent)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "setup-work"})  # noqa: S310
        with urllib.request.urlopen(req, timeout=timeout) as resp, os.fdopen(fd, "wb") as out:  # noqa: S310
            shutil.copyfileobj(resp, out, 1 << 20)
        got = sha256_file(Path(tmp))
        if got != sha256:
            raise SetupWorkError(
                f"не совпадает SHA256 для {url}\n  ожидается {sha256}\n  получено {got}",
                hint="файл удалён; проверьте прокси, зеркало и закреплённую контрольную сумму",
            )
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return dest


def _safe_member(base: Path, name: str) -> Path:
    target = (base / name).resolve()
    if not str(target).startswith(str(base.resolve()) + os.sep) and target != base.resolve():
        raise SetupWorkError(f"путь в архиве выходит за пределы каталога распаковки: {name}")
    return target


def safe_extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    name = archive.name
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                _safe_member(dest, info.filename)
            zf.extractall(dest)  # noqa: S202 - элементы архива проверены выше
            for info in zf.infolist():
                mode = (info.external_attr >> 16) & 0o777
                if mode:
                    os.chmod(dest / info.filename, mode)
        return
    with tarfile.open(archive) as tf:
        for m in tf.getmembers():
            _safe_member(dest, m.name)
            if m.issym() or m.islnk():
                link_target = (
                    (dest / m.name).parent / m.linkname if m.issym() else dest / m.linkname
                )
                _safe_member(dest, os.path.relpath(link_target, dest))
            if m.isdev():
                raise SetupWorkError(f"в архиве недопустимый файл устройства: {m.name}")
        if hasattr(tarfile, "data_filter"):
            tf.extractall(dest, filter="data")
        else:  # pragma: no cover - Python < 3.11.4
            tf.extractall(dest)  # noqa: S202 - элементы архива проверены выше


def make_link(src: Path, dest: Path, managed_root: Path) -> None:
    """Создать ссылку dest на src.

    Чужие файлы и ссылки за пределы управляемых каталогов не заменяются."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink():
        current = Path(os.readlink(dest))
        if not current.is_absolute():
            current = dest.parent / current
        if not str(current.resolve(strict=False)).startswith(str(managed_root.resolve())):
            raise SetupWorkError(
                f"{dest} уже ссылается на {current}, которым setup-work не управляет",
                hint=(
                    "проверьте ссылку и удалите её вручную, "
                    "если нужно передать управление setup-work"
                ),
            )
        dest.unlink()
    elif dest.exists():
        raise SetupWorkError(
            f"{dest} существует и не является символьной ссылкой; перезапись отменена",
            hint="переместите файл либо задайте другой paths.bin_dir в config/defaults.yaml",
        )
    dest.symlink_to(src)


def _need(*values: object) -> None:
    if not all(values):
        raise SetupWorkError("внутренняя ошибка: неполный шаг установки")


def execute(steps: list[Step], runner: Runner, managed_root: Path) -> None:
    for i, step in enumerate(steps, 1):
        if step.when is not None and not step.when():
            runner.info(f"[dim]{i}/{len(steps)} skip:[/] {step.description}(уже выполнено)")
            continue
        runner.info(f"[bold]{i}/{len(steps)}[/] {step.description}")
        if step.kind == "manual":
            runner.info(f"  [yellow]требуется действие вручную:[/] {step.render()}")
            continue
        if step.privileged:
            runner.require_confirmation(
                f"Выполнить команду с повышенными правами: {step.render()} ?"
            )
        elif step.confirm:
            runner.require_confirmation(f"Run: {step.render()} ?")
        if step.destructive:
            runner.require_confirmation(f"{step.description}: {step.render()} ?", destructive=True)
        if step.kind == "command":
            _need(step.argv)
            runner.run(step.argv, cwd=step.cwd, env=step.env, timeout=step.timeout)
            continue
        if step.kind in ("artifact", "fetch", "call"):
            _run_special(step, runner)
            runner.event("step", kind=step.kind, description=step.description)
            continue
        if runner.dry_run:
            runner.info(f"  [dim](dry-run) {step.render()}[/]")
            continue
        if step.kind == "extract":
            _need(step.src, step.dest)
            safe_extract(step.src, step.dest)
        elif step.kind == "link":
            _need(step.src, step.dest)
            make_link(step.src, step.dest, managed_root)
        elif step.kind == "remove":
            _need(step.dest)
            if step.dest.is_symlink() or step.dest.is_file():
                step.dest.unlink()
            elif step.dest.is_dir():
                shutil.rmtree(step.dest)
        elif step.kind == "write":
            _need(step.dest, step.content is not None)
            step.dest.parent.mkdir(parents=True, exist_ok=True)
            step.dest.write_text(step.content, encoding="utf-8")
            if step.mode is not None:
                step.dest.chmod(step.mode)
        runner.event("step", kind=step.kind, description=step.description)


def _run_special(step: Step, runner: Runner) -> None:
    from . import artifacts

    if step.kind == "call":
        _need(step.action)
        if runner.dry_run:
            runner.info(f"  [dim](dry-run) {step.description}[/]")
            return
        step.action(runner)  # type: ignore[misc]
        return
    _need(step.artifact, step.dest)
    cfg = step.cfg
    _need(cfg)
    if step.kind == "artifact":
        artifacts.install(
            cfg, step.artifact, step.dest, runner, name=step.description, keep_archive_default=False
        )
        return
    # fetch: сохранить установочный архив в кэш под закреплённым именем
    if step.dest.exists() and step.sha256 and sha256_file(step.dest) == step.sha256:
        runner.info(f"  используется кэш {step.dest.name}")
        return
    got = artifacts.fetch_archive(cfg, step.artifact, runner, name=step.dest.name)
    if runner.dry_run:
        return
    step.dest.parent.mkdir(parents=True, exist_ok=True)
    if got != step.dest:
        shutil.copyfile(got, step.dest)
        if got.parent == step.dest.parent or "artifacts" in got.parts:
            got.unlink(missing_ok=True)
    want = step.artifact.sha256 or step.sha256
    if want and sha256_file(step.dest) != want:
        step.dest.unlink()
        raise SetupWorkError(f"не совпадает SHA256 для {step.artifact.uri}(ожидается {want})")


def ensure_executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
