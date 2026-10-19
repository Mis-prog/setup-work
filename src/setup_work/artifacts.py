"""Корпоративные артефакты: HDFS, локальные/NFS-каталоги и HTTPS.

Архивы загружаются в <data>/artifacts/downloads и распаковываются через временный
каталог. Маркер .setup-work-artifact.json позволяет переиспользовать установку.
HDFS читается штатным клиентом кластера с его настройками Kerberos и HA.
ZIP сохраняет права Unix и символьные ссылки; пути проверяются до распаковки."""

from __future__ import annotations

import contextlib
import datetime as _dt
import json
import os
import re
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
    # В FsShell нет разделителя --; допустимость пути уже проверена схемой.
    if uri.startswith("-"):
        raise SetupWorkError(f"путь HDFS похож на параметр команды и отклонён: {uri!r}")
    return uri


def hdfs_cli(h: HdfsConfig) -> str:
    cli = which(h.cli)
    if not cli:
        raise MissingDependencyError(
            f"клиент HDFS {h.cli!r} не найден",
            hint=(
                "установите клиент Hadoop или задайте полный "
                "путь в hdfs.cli файла config/corporate.yaml"
            ),
        )
    return cli


def kerberos_ok(h: HdfsConfig) -> tuple[bool, str]:
    """`klist -s` возвращает 0, если в кэше есть действующий билет."""
    if h.kerberos == "off":
        return True, "проверка Kerberos отключена"
    if not which("klist"):
        if h.kerberos == "required":
            return False, "klist не найден, но Kerberos обязателен; установите клиент Kerberos"
        return True, "klist не найден; предполагается доступ без Kerberos"
    res = probe(["klist", "-s"], timeout=10)
    if res.ok:
        return True, "действующий билет Kerberos"
    return False, "нет действующего билета Kerberos"


def require_kerberos(h: HdfsConfig) -> None:
    ok, why = kerberos_ok(h)
    if not ok:
        raise SetupWorkError(
            f"не удалось прочитать HDFS: {why}",
            hint="выполните kinit с вашим пользователем и REALM, затем повторите команду",
        )


def hdfs_size(h: HdfsConfig, uri: str) -> int | None:
    cli = hdfs_cli(h)
    res = probe([cli, "dfs", *h.extra_args, "-du", "-s", _safe_uri(uri)], timeout=120)
    if not res.ok:
        msg = (res.stderr or res.stdout).strip().splitlines()
        tail = msg[-1] if msg else f"код {res.returncode}"
        if "No such file" in (res.stderr + res.stdout):
            raise NotFoundError(f"{uri} нет на HDFS; проверьте uri в corporate.yaml и hdfs dfs -ls")
        raise SetupWorkError(f"команда hdfs dfs -du завершилась с ошибкой для {uri}: {tail}")
    first = res.stdout.strip().split()
    return int(first[0]) if first and first[0].isdigit() else None


# ------------------------------------------------------------------ проверка


@dataclass
class ArtifactInfo:
    uri: str
    scheme: str
    size_bytes: int | None
    format: str


def inspect(cfg: Config, ref: ArtifactRef) -> ArtifactInfo:
    sch = scheme(ref.uri)
    size: int | None = None
    if ref.effective_format == "zip-parts":
        parts = zip_parts(cfg, ref)
        sizes = [inspect(cfg, ArtifactRef(uri=uri, format="file")).size_bytes for uri in parts]
        size = sum(s for s in sizes if s is not None) if all(s is not None for s in sizes) else None
        return ArtifactInfo(ref.uri, sch, size, "zip-parts")
    if sch == "hdfs":
        require_kerberos(cfg.corporate.hdfs)
        size = hdfs_size(cfg.corporate.hdfs, ref.uri)
    elif sch == "local":
        p = local_path(ref.uri)
        if not p.exists():
            raise NotFoundError(f"артефакт {p} не найден; проверьте uri в corporate.yaml")
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
    return bool(
        m
        and m.get("uri") == ref.uri
        and m.get("sha256") == ref.sha256
        and m.get("format", "" if ref.effective_format == "zip-parts" else ref.effective_format)
        == ref.effective_format
    )


def zip_parts(cfg: Config, ref: ArtifactRef) -> list[str]:
    """Найти полный комплект <имя>-1-of-N.zip в каталоге, в числовом порядке."""
    if scheme(ref.uri) == "hdfs":
        h = cfg.corporate.hdfs
        require_kerberos(h)
        result = probe(
            [hdfs_cli(h), "dfs", *h.extra_args, "-ls", _safe_uri(ref.uri)], timeout=h.timeout_s
        )
        if not result.ok:
            raise NotFoundError(
                f"не удалось прочитать каталог частей {ref.uri}",
                hint="проверьте путь models в corporate.yaml, права HDFS и билет Kerberos",
            )
        names = [
            line.split(maxsplit=7)[7].rsplit("/", 1)[-1]
            for line in result.stdout.splitlines()
            if line.startswith("-") and len(line.split(maxsplit=7)) == 8
        ]
    elif scheme(ref.uri) == "local":
        folder = local_path(ref.uri)
        if not folder.is_dir():
            raise NotFoundError(
                f"не найден каталог частей {folder}; исправьте uri в corporate.yaml"
            )
        names = [p.name for p in folder.iterdir() if p.is_file()]
    else:
        raise SetupWorkError("для zip-parts задайте каталог HDFS или локальный каталог")
    groups: dict[tuple[str, int], dict[int, str]] = {}
    for filename in names:
        match = re.fullmatch(r"(.+)-(\d+)-of-(\d+)\.zip", filename, re.IGNORECASE)
        if not match:
            continue
        prefix, number, total = match.groups()
        index, count = int(number), int(total)
        group = groups.setdefault((prefix, count), {})
        if index < 1 or index > count or index in group:
            raise SetupWorkError(f"неверный или повторяющийся номер части: {filename}")
        group[index] = filename
    if len(groups) != 1:
        raise SetupWorkError(
            f"в {ref.uri} нужен один комплект <имя>-1-of-N.zip; найдено комплектов: {len(groups)}",
            hint="укажите отдельную папку одной модели, например с Qwen_Qwen3.8-27B-1-of-19.zip",
        )
    (_, count), group = next(iter(groups.items()))
    missing = [str(n) for n in range(1, count + 1) if n not in group]
    if missing:
        raise SetupWorkError(
            f"в {ref.uri} отсутствуют части: {', '.join(missing)} из {count}",
            hint="доложите недостающие ZIP-файлы в HDFS и повторите загрузку модели",
        )
    return [ref.uri.rstrip("/") + "/" + group[n] for n in range(1, count + 1)]


def _independent_zips(parts: list[Path]) -> bool:
    try:
        for part in parts:
            with zipfile.ZipFile(part) as archive:
                archive.infolist()
        return True
    except (zipfile.BadZipFile, OSError):
        return False


def _unpack_parts(parts: list[Path], staging: Path, joined: Path, runner: Runner) -> None:
    if _independent_zips(parts):
        seen: dict[str, tuple[int, int]] = {}
        for part in parts:
            with zipfile.ZipFile(part) as archive:
                for member in archive.infolist():
                    if member.is_dir():
                        continue
                    signature = (member.CRC, member.file_size)
                    key = str(Path(member.filename))
                    if key in seen and seen[key] != signature:
                        raise SetupWorkError(f"ZIP-части содержат разные версии файла {key}")
                    seen[key] = signature
        required = sum(size for _, size in seen.values())
        if required + 512 * 1024**2 > _free(staging):
            raise SetupWorkError(f"недостаточно места для распаковки {gib(required)}")
        for part in parts:
            runner.info(f"  распаковка {part.name} ...")
            safe_unzip(part, staging)
        return
    runner.info(f"  сборка {len(parts)} частей ZIP в числовом порядке ...")
    try:
        with joined.open("wb") as output:
            for part in parts:
                with part.open("rb") as source:
                    shutil.copyfileobj(source, output, 8 << 20)
        if uncompressed_size(joined, "zip") + 512 * 1024**2 > _free(staging):
            raise SetupWorkError("недостаточно места для распаковки собранного ZIP")
        safe_unzip(joined, staging)
    except (zipfile.BadZipFile, NotImplementedError) as exc:
        raise SetupWorkError(
            "не удалось распаковать ZIP-части",
            hint="проверьте целостность файлов: нужны отдельные ZIP "
            "либо последовательные части одного ZIP",
        ) from exc
    finally:
        joined.unlink(missing_ok=True)


# ------------------------------------------------------------------ распаковка


def _zip_is_symlink(info: zipfile.ZipInfo) -> bool:
    return stat.S_ISLNK(info.external_attr >> 16)


def safe_unzip(archive: Path, dest: Path) -> None:
    """Распаковать ZIP с проверкой путей и сохранением прав Unix и ссылок (zip -y)."""
    with zipfile.ZipFile(archive) as zf:
        infos = zf.infolist()
        for info in infos:
            _safe_member(dest, info.filename)
            if _zip_is_symlink(info):
                target = zf.read(info).decode("utf-8", "replace")
                if os.path.isabs(target):
                    # Абсолютные ссылки, например bin/python на /usr/bin/python3.12,
                    # воссоздаются как есть; внешние цели используются только для чтения.
                    continue
                _safe_member(dest, os.path.relpath((dest / info.filename).parent / target, dest))
        dest.mkdir(parents=True, exist_ok=True)
        for info in infos:
            _safe_member(dest, info.filename)
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


# --------------------------------------------------------------------- установка


@dataclass
class FetchResult:
    dest: Path
    reused: bool
    archive: Path | None = None


def fetch_archive(cfg: Config, ref: ArtifactRef, runner: Runner, name: str) -> Path:
    """Поместить архив, файл или каталог в локальный кэш загрузок."""
    downloads = cfg.artifacts_dir / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    base = ref.uri.rstrip("/").rsplit("/", 1)[-1] or name
    target = downloads / f"{name}--{base}"
    sch = scheme(ref.uri)
    if sch == "local":
        src = local_path(ref.uri)
        if ref.effective_format == "dir":
            return src  # использовать исходный каталог; копирование при установке
        if not src.exists():
            raise NotFoundError(f"артефакт {src} не найден; проверьте uri в corporate.yaml")
        if target.exists() and target.stat().st_size == src.stat().st_size:
            return target
        runner.run(["cp", "--", str(src), str(target) + ".part"])
    elif sch == "https":
        if not ref.sha256:
            raise SetupWorkError(
                f"HTTPS-артефакт {ref.uri} требует SHA256; заполните sha256 в config/corporate.yaml"
            )
        if runner.dry_run:
            runner.info(f"  [dim](dry-run) загрузить {ref.uri} -> {target}[/]")
            return target
        return download_verified(ref.uri, target, ref.sha256, timeout=3600)
    else:
        h = cfg.corporate.hdfs
        require_kerberos(h)
        cli = hdfs_cli(h)
        size = hdfs_size(h, ref.uri)
        if target.exists() and size is not None and _size_of(target) == size:
            runner.info(f"  используется установленная копия {target}")
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
    """Загрузить, проверить и распаковать ref в dest; повторная установка переиспользуется."""
    if is_current(dest, ref):
        runner.info(f"[green]уже установлен[/] {name} at {dest}")
        return FetchResult(dest, reused=True)
    info = inspect(cfg, ref)
    size = info.size_bytes
    fmt = info.format
    # Архив, распакованная копия весов и запас места.
    need = (size or 0) * (
        3 if fmt == "zip-parts" else 2 if fmt in ("zip", "tar", "tar.gz") else 1
    ) + 1024**3
    free = _free(dest.parent)
    runner.info(
        f"{name}: {info.uri} ({gib(size) if size else 'размер неизвестен'}, {fmt}) -> {dest}; "
        f"{gib(free)} free"
    )
    if size and need > free:
        raise SetupWorkError(
            f"недостаточно места на диске для {name}: требуется примерно{gib(need)}, "
            f"{gib(free)} свободно в {dest.parent}",
            hint="укажите том с достаточным местом через SETUP_WORK_MODEL_DIR или SETUP_WORK_HOME",
        )
    threshold = cfg.defaults.downloads.confirm_above_gb * 1e9
    if confirm_large and (size is None or size > threshold):
        runner.require_confirmation(
            f"Загрузить {gib(size) if size else 'неизвестный размер'} для {name}?"
        )
    parts: list[Path] = []
    if fmt == "zip-parts":
        if ref.sha256:
            raise SetupWorkError("sha256 каталога zip-parts не поддерживается; уберите sha256")
        parts = [
            fetch_archive(cfg, ArtifactRef(uri=uri, format="file"), runner, f"{name}-{index}")
            for index, uri in enumerate(zip_parts(cfg, ref), 1)
        ]
        archive = cfg.artifacts_dir / "downloads" / f"{name}--joined.zip"
    else:
        archive = fetch_archive(cfg, ref, runner, name)
    if runner.dry_run:
        runner.info(f"  [dim](dry-run) распаковать {archive} -> {dest}[/]")
        return FetchResult(dest, reused=False, archive=archive)
    if ref.sha256 and archive.is_file():
        runner.info(f"  проверка SHA256 архива {archive.name} ...")
        got = sha256_file(archive)
        if got != ref.sha256:
            archive.unlink()
            raise SetupWorkError(
                f"не совпадает SHA256 для {ref.uri}\n  ожидается {ref.sha256}\n  получено {got}",
                hint="локальная копия удалена; проверьте архив и sha256 в config/corporate.yaml",
            )
    staging = dest.parent / f".{dest.name}.unpacking"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        if fmt == "zip-parts":
            _unpack_parts(parts, staging, archive, runner)
        elif fmt == "zip" or (fmt in ("tar", "tar.gz")):
            if archive.is_file():
                total = uncompressed_size(archive, fmt)
                if total + 512 * 1024**2 > _free(staging):
                    raise SetupWorkError(
                        f"недостаточно места для распаковки {gib(total)} в {dest.parent}"
                    )
            runner.info(f"  распаковка {archive.name} ...")
            if fmt == "zip":
                safe_unzip(archive, staging)
            else:
                from .steps import safe_extract

                safe_extract(archive, staging)
        elif fmt == "dir":
            shutil.copytree(archive, staging, symlinks=True, dirs_exist_ok=True)
        else:  # один файл
            shutil.copy2(archive, staging / archive.name.split("--", 1)[-1])
        (staging / MARKER).write_text(
            json.dumps(
                {
                    "name": name,
                    "uri": ref.uri,
                    "sha256": ref.sha256,
                    "format": fmt,
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
    except (zipfile.BadZipFile, NotImplementedError) as exc:
        shutil.rmtree(staging, ignore_errors=True)
        for part in parts:
            part.unlink(missing_ok=True)
        raise SetupWorkError(
            "ZIP повреждён или использует неподдерживаемый способ сжатия",
            hint="проверьте ZIP-файлы в HDFS и повторите загрузку модели",
        ) from exc
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        for part in parts:
            part.unlink(missing_ok=True)
        raise
    keep = ref.keep_archive if ref.keep_archive is not None else keep_archive_default
    if not keep and archive.is_file() and archive.parent == cfg.artifacts_dir / "downloads":
        archive.unlink()
    if not keep:
        for part in parts:
            part.unlink(missing_ok=True)
    runner.event("artifact.install", name=name, uri=ref.uri, dest=str(dest))
    return FetchResult(dest, reused=False, archive=archive if keep else None)


def find_one(root: Path, pattern: str) -> Path | None:
    """Найти ближайший к корню путь по шаблону с постоянным порядком выбора."""
    hits = sorted(root.rglob(pattern), key=lambda p: (len(p.parts), str(p)))
    return hits[0] if hits else None
