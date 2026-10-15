"""Corporate artifact sources: HDFS (via `hdfs dfs`), local/NFS paths and HTTPS.

An artifact is fetched into ``<data>/artifacts/downloads/`` and unpacked into its
final directory through a temporary sibling, then marked with
``.setup-work-artifact.json``. Re-running with the same URI/size/sha256 reuses
the result, so interrupted runs can simply be repeated.

* HDFS access uses the cluster's own client (``hdfs dfs -stat/-get``) so
  Kerberos, HA name services and site configuration work exactly as for the
  user's shell. setup-work never handles passwords or keytabs: without a valid
  ticket it stops and asks for ``kinit``.
* Zip archives keep Unix permissions and symlinks (venvs need both); every
  member is checked for path traversal before anything is written.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import json
import os
import shutil
import stat
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ArtifactRef, Config, HdfsConfig
from .errors import MissingDependencyError, NotFoundError, SetupWorkError
from .memory import gib
from .runner import Runner, probe, which
from .steps import _safe_member, download_verified, sha256_file

MARKER = ".setup-work-artifact.json"


def scheme(uri: str) -> str:
    if uri.startswith(("hdfs:", "viewfs:")):
        return "hdfs"
    if uri.startswith("https://"):
        return "https"
    return "local"


def local_path(uri: str) -> Path:
    if uri.startswith("file://"):
        uri = uri[len("file://") :]
    return Path(uri).expanduser()


# ------------------------------------------------------------------------- HDFS


def _safe_uri(uri: str) -> str:
    # FsShell has no "--" separator; schema validation already guarantees this.
    if uri.startswith("-"):
        raise SetupWorkError(f"refusing option-like HDFS path {uri!r}")
    return uri


def hdfs_cli(h: HdfsConfig) -> str:
    cli = which(h.cli)
    if not cli:
        raise MissingDependencyError(
            f"HDFS client {h.cli!r} not found",
            hint="install the Hadoop client or set corporate.hdfs.cli to its full path",
        )
    return cli


def kerberos_ok(h: HdfsConfig) -> tuple[bool, str]:
    """`klist -s` exits 0 when the credential cache holds a valid ticket."""
    if h.kerberos == "off":
        return True, "kerberos check disabled"
    if not which("klist"):
        if h.kerberos == "required":
            return False, "klist not found but kerberos is required"
        return True, "klist not found (assuming no Kerberos)"
    res = probe(["klist", "-s"], timeout=10)
    if res.ok:
        return True, "valid Kerberos ticket"
    return False, "no valid Kerberos ticket"


def require_kerberos(h: HdfsConfig) -> None:
    ok, why = kerberos_ok(h)
    if not ok:
        raise SetupWorkError(
            f"cannot read HDFS: {why}",
            hint="run `kinit <user>@<REALM>` (setup-work never asks for your password), then retry",
        )


def hdfs_size(h: HdfsConfig, uri: str) -> int | None:
    cli = hdfs_cli(h)
    res = probe([cli, "dfs", *h.extra_args, "-du", "-s", _safe_uri(uri)], timeout=120)
    if not res.ok:
        msg = (res.stderr or res.stdout).strip().splitlines()
        tail = msg[-1] if msg else f"exit {res.returncode}"
        if "No such file" in (res.stderr + res.stdout):
            raise NotFoundError(f"{uri} does not exist on HDFS")
        raise SetupWorkError(f"hdfs dfs -du failed for {uri}: {tail}")
    first = res.stdout.strip().split()
    return int(first[0]) if first and first[0].isdigit() else None


# ------------------------------------------------------------------ inspection


@dataclass
class ArtifactInfo:
    uri: str
    scheme: str
    size_bytes: int | None
    format: str


def inspect(cfg: Config, ref: ArtifactRef) -> ArtifactInfo:
    sch = scheme(ref.uri)
    size: int | None = None
    if sch == "hdfs":
        require_kerberos(cfg.corporate.hdfs)
        size = hdfs_size(cfg.corporate.hdfs, ref.uri)
    elif sch == "local":
        p = local_path(ref.uri)
        if not p.exists():
            raise NotFoundError(f"artifact {p} does not exist")
        size = (
            p.stat().st_size
            if p.is_file()
            else sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        )
    return ArtifactInfo(ref.uri, sch, size, ref.effective_format)


def read_marker(dest: Path) -> dict[str, Any] | None:
    try:
        return json.loads((dest / MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def is_current(dest: Path, ref: ArtifactRef) -> bool:
    m = read_marker(dest)
    return bool(m and m.get("uri") == ref.uri and m.get("sha256") == ref.sha256)


# ------------------------------------------------------------------ extraction


def _zip_is_symlink(info: zipfile.ZipInfo) -> bool:
    return stat.S_ISLNK(info.external_attr >> 16)


def safe_unzip(archive: Path, dest: Path) -> None:
    """Extract with traversal checks, Unix modes and symlinks (zip -y) preserved."""
    with zipfile.ZipFile(archive) as zf:
        infos = zf.infolist()
        for info in infos:
            _safe_member(dest, info.filename)
            if _zip_is_symlink(info):
                target = zf.read(info).decode("utf-8", "replace")
                if os.path.isabs(target):
                    # Absolute links (e.g. venv bin/python -> /usr/bin/python3.12) are
                    # recreated as-is; they point outside but are only *read* at runtime.
                    continue
                _safe_member(dest, os.path.relpath((dest / info.filename).parent / target, dest))
        dest.mkdir(parents=True, exist_ok=True)
        for info in infos:
            out = dest / info.filename
            if info.is_dir():
                out.mkdir(parents=True, exist_ok=True)
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            if _zip_is_symlink(info):
                with contextlib.suppress(FileNotFoundError):
                    out.unlink()
                out.symlink_to(zf.read(info).decode("utf-8", "replace"))
                continue
            with zf.open(info) as src, out.open("wb") as fh:
                shutil.copyfileobj(src, fh, 8 << 20)
            mode = (info.external_attr >> 16) & 0o777
            if mode:
                os.chmod(out, mode & ~0o022 | 0o600)


def uncompressed_size(archive: Path, fmt: str) -> int:
    if fmt == "zip":
        with zipfile.ZipFile(archive) as zf:
            return sum(i.file_size for i in zf.infolist())
    if fmt in ("tar", "tar.gz"):
        with tarfile.open(archive) as tf:
            return sum(m.size for m in tf.getmembers())
    return archive.stat().st_size


def _free(path: Path) -> int:
    p = path
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free


# --------------------------------------------------------------------- install


@dataclass
class FetchResult:
    dest: Path
    reused: bool
    archive: Path | None = None


def fetch_archive(cfg: Config, ref: ArtifactRef, runner: Runner, name: str) -> Path:
    """Bring the archive (or file/dir) into the local download cache."""
    downloads = cfg.artifacts_dir / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    base = ref.uri.rstrip("/").rsplit("/", 1)[-1] or name
    target = downloads / f"{name}--{base}"
    sch = scheme(ref.uri)
    if sch == "local":
        src = local_path(ref.uri)
        if ref.effective_format == "dir":
            return src  # use in place; copied during install
        if not src.exists():
            raise NotFoundError(f"artifact {src} does not exist")
        if target.exists() and target.stat().st_size == src.stat().st_size:
            return target
        runner.run(["cp", "--", str(src), str(target) + ".part"])
    elif sch == "https":
        if not ref.sha256:
            raise SetupWorkError(f"https artifact {ref.uri} needs a pinned sha256")
        if runner.dry_run:
            runner.info(f"  [dim](dry-run) download {ref.uri} -> {target}[/]")
            return target
        return download_verified(ref.uri, target, ref.sha256, timeout=3600)
    else:
        h = cfg.corporate.hdfs
        require_kerberos(h)
        cli = hdfs_cli(h)
        size = hdfs_size(h, ref.uri)
        if target.exists() and size is not None and _size_of(target) == size:
            runner.info(f"  reusing {target}")
            return target
        part = Path(str(target) + ".part")
        if part.exists():
            if part.is_dir():
                shutil.rmtree(part)
            else:
                part.unlink()
        runner.run(
            [cli, "dfs", *h.extra_args, "-get", _safe_uri(ref.uri), str(part)],
            timeout=h.timeout_s,
        )
    if runner.dry_run:
        return target
    part = Path(str(target) + ".part")
    if target.exists():
        shutil.rmtree(target) if target.is_dir() else target.unlink()
    os.replace(part, target)
    return target


def _size_of(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def install(
    cfg: Config,
    ref: ArtifactRef,
    dest: Path,
    runner: Runner,
    *,
    name: str,
    keep_archive_default: bool = False,
    confirm_large: bool = True,
) -> FetchResult:
    """Fetch + verify + unpack `ref` into `dest` (idempotent)."""
    if is_current(dest, ref):
        runner.info(f"[green]up to date[/] {name} at {dest}")
        return FetchResult(dest, reused=True)
    info = inspect(cfg, ref)
    size = info.size_bytes
    fmt = info.format
    # Archive + unpacked copy (weights barely compress) + margin.
    need = (size or 0) * (2 if fmt in ("zip", "tar", "tar.gz") else 1) + 1024**3
    free = _free(dest.parent)
    runner.info(
        f"{name}: {info.uri} ({gib(size) if size else 'size unknown'}, {fmt}) -> {dest}; "
        f"{gib(free)} free"
    )
    if size and need > free:
        raise SetupWorkError(
            f"not enough disk space for {name}: need ~{gib(need)}, "
            f"{gib(free)} free at {dest.parent}",
            hint="set SETUP_WORK_MODEL_DIR / XDG_DATA_HOME to a larger volume",
        )
    threshold = cfg.defaults.downloads.confirm_above_gb * 1e9
    if confirm_large and (size is None or size > threshold):
        runner.require_confirmation(f"Fetch {gib(size) if size else 'unknown size'} for {name}?")
    archive = fetch_archive(cfg, ref, runner, name)
    if runner.dry_run:
        runner.info(f"  [dim](dry-run) unpack {archive} -> {dest}[/]")
        return FetchResult(dest, reused=False, archive=archive)
    if ref.sha256 and archive.is_file():
        runner.info(f"  verifying sha256 of {archive.name} ...")
        got = sha256_file(archive)
        if got != ref.sha256:
            archive.unlink()
            raise SetupWorkError(
                f"checksum mismatch for {ref.uri}\n  expected {ref.sha256}\n  got      {got}",
                hint="the local copy was deleted; check the artifact or its pinned sha256",
            )
    staging = dest.parent / f".{dest.name}.unpacking"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        if fmt == "zip" or (fmt in ("tar", "tar.gz")):
            if archive.is_file():
                total = uncompressed_size(archive, fmt)
                if total + 512 * 1024**2 > _free(staging):
                    raise SetupWorkError(
                        f"not enough space to unpack {gib(total)} into {dest.parent}"
                    )
            runner.info(f"  unpacking {archive.name} ...")
            if fmt == "zip":
                safe_unzip(archive, staging)
            else:
                from .steps import safe_extract

                safe_extract(archive, staging)
        elif fmt == "dir":
            shutil.copytree(archive, staging, symlinks=True, dirs_exist_ok=True)
        else:  # single file
            shutil.copy2(archive, staging / archive.name.split("--", 1)[-1])
        (staging / MARKER).write_text(
            json.dumps(
                {
                    "name": name,
                    "uri": ref.uri,
                    "sha256": ref.sha256,
                    "size_bytes": size,
                    "installed_at": _dt.datetime.now(_dt.UTC).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if dest.exists():
            old = dest.parent / f".{dest.name}.old"
            if old.exists():
                shutil.rmtree(old)
            os.replace(dest, old)
            shutil.rmtree(old, ignore_errors=True)
        os.replace(staging, dest)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    keep = ref.keep_archive if ref.keep_archive is not None else keep_archive_default
    if not keep and archive.is_file() and archive.parent == cfg.artifacts_dir / "downloads":
        archive.unlink()
    runner.event("artifact.install", name=name, uri=ref.uri, dest=str(dest))
    return FetchResult(dest, reused=False, archive=archive if keep else None)


def find_one(root: Path, pattern: str) -> Path | None:
    """Shallowest match of a glob below root (deterministic)."""
    hits = sorted(root.rglob(pattern), key=lambda p: (len(p.parts), str(p)))
    return hits[0] if hits else None
