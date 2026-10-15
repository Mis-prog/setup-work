"""Plan/execute primitives: every installer first builds a list of Steps.

A plan can be printed (``--dry-run``), serialised (``--json``) and executed.
Only the step kinds below exist; there is intentionally no "run this shell
snippet" step.

* ``command``   run an argv list (optionally in a directory, with extra env)
* ``download``  fetch a URL to a file and verify its sha256 before use
* ``extract``   unpack a verified archive with path-traversal protection
* ``link``      create/refresh a symlink inside bin_dir (never overwrites foreign files)
* ``manual``    an instruction the user must perform (privileged or interactive)
"""

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
    "download",
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
    url: str | None = None
    sha256: str | None = None
    src: Path | None = None
    dest: Path | None = None
    content: str | None = None
    privileged: bool = False
    confirm: bool = False
    destructive: bool = False
    timeout: float | None = None
    when: Callable[[], bool] | None = field(default=None, repr=False)
    artifact: Any = field(default=None, repr=False)  # config.ArtifactRef for artifact/fetch
    action: Callable[[Runner], None] | None = field(default=None, repr=False)  # for "call"
    mode: int | None = None  # file mode for "write"
    cfg: Any = field(default=None, repr=False)  # config.Config for artifact/fetch steps

    def render(self) -> str:
        if self.kind == "command" and self.argv:
            return format_argv(self.argv) + (f"   (in {self.cwd})" if self.cwd else "")
        if self.kind == "download":
            return f"download {self.url} -> {self.dest} (sha256 {str(self.sha256)[:12]}…)"
        if self.kind == "extract":
            return f"extract {self.src} -> {self.dest}"
        if self.kind == "link":
            return f"ln -sfn {self.src} {self.dest}"
        if self.kind == "remove":
            return f"remove {self.dest}"
        if self.kind == "write":
            return f"write {self.dest}"
        if self.kind in ("artifact", "fetch") and self.artifact is not None:
            return f"fetch {self.artifact.uri} -> {self.dest}"
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
    """Download (honouring HTTP(S)_PROXY / NO_PROXY) and verify before moving into place."""
    if not url.startswith("https://"):
        raise SetupWorkError(f"refusing non-HTTPS download: {url}")
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
                f"checksum mismatch for {url}\n  expected {sha256}\n  got      {got}",
                hint="the download was discarded; check proxies/mirrors or update the pinned "
                "digest",
            )
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return dest


def _safe_member(base: Path, name: str) -> Path:
    target = (base / name).resolve()
    if not str(target).startswith(str(base.resolve()) + os.sep) and target != base.resolve():
        raise SetupWorkError(f"archive member escapes destination: {name}")
    return target


def safe_extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    name = archive.name
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                _safe_member(dest, info.filename)
            zf.extractall(dest)  # noqa: S202 - members validated above
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
                raise SetupWorkError(f"archive contains a device file: {m.name}")
        if hasattr(tarfile, "data_filter"):
            tf.extractall(dest, filter="data")
        else:  # pragma: no cover - Python < 3.11.4
            tf.extractall(dest)  # noqa: S202 - members validated above


def make_link(src: Path, dest: Path, managed_root: Path) -> None:
    """Symlink dest -> src. Existing non-symlinks, or links pointing outside
    setup-work's managed directories, are never replaced."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink():
        current = Path(os.readlink(dest))
        if not current.is_absolute():
            current = dest.parent / current
        if not str(current.resolve(strict=False)).startswith(str(managed_root.resolve())):
            raise SetupWorkError(
                f"{dest} already links to {current}, which setup-work does not manage",
                hint="remove it yourself if you want setup-work to manage this binary",
            )
        dest.unlink()
    elif dest.exists():
        raise SetupWorkError(
            f"{dest} exists and is not a symlink; refusing to overwrite",
            hint="move it away or set paths.bin_dir to another directory",
        )
    dest.symlink_to(src)


def _need(*values: object) -> None:
    if not all(values):
        raise SetupWorkError("internal error: incomplete installation step")


def execute(steps: list[Step], runner: Runner, managed_root: Path) -> None:
    for i, step in enumerate(steps, 1):
        if step.when is not None and not step.when():
            runner.info(f"[dim]{i}/{len(steps)} skip:[/] {step.description} (already satisfied)")
            continue
        runner.info(f"[bold]{i}/{len(steps)}[/] {step.description}")
        if step.kind == "manual":
            runner.info(f"  [yellow]manual action required:[/] {step.render()}")
            continue
        if step.privileged:
            runner.require_confirmation(f"Run privileged command: {step.render()} ?")
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
        if step.kind == "download":
            _need(step.url, step.dest, step.sha256)
            download_verified(step.url, step.dest, step.sha256)
        elif step.kind == "extract":
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
    # fetch: bring an installer archive into the download cache under its pinned name
    if step.dest.exists() and step.sha256 and sha256_file(step.dest) == step.sha256:
        runner.info(f"  using cached {step.dest.name}")
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
        raise SetupWorkError(f"checksum mismatch for {step.artifact.uri} (expected {want})")


def ensure_executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
