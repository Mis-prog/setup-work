"""Хранение моделей из корпоративных артефактов.

Источник выбирается из corporate.models или source варианта. Архивы
распаковываются в <models_dir>/<id>/<variant> и помечаются для повторного
использования. doctor и recommend не загружают модели."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .config import ArtifactRef, Config, Variant
from .errors import NotFoundError, SetupWorkError, UsageError
from .memory import gib
from .runner import Runner


def require_artifact(cfg: Config, model_id: str, variant: Variant) -> ArtifactRef:
    ref = model_artifact(cfg, model_id, variant)
    if ref is None:
        raise NotFoundError(
            f"{model_id}:{variant.id} не имеет источника артефакта",
            hint=f'добавьте "{model_id}:{variant.id}" в models файла corporate.yaml либо задайте '
            "source варианта в models.yaml",
        )
    return ref


def model_artifact(cfg: Config, model_id: str, variant: Variant) -> ArtifactRef | None:
    """Источник варианта: сначала corporate.models, затем source."""
    ref = cfg.corporate.models.get(f"{model_id}:{variant.id}")
    if ref is not None:
        return ref
    if variant.source:
        return ArtifactRef(uri=variant.source, sha256=variant.sha256)
    return None


def artifact_model_dir(cfg: Config, model_id: str, variant: Variant) -> Path:
    return cfg.models_dir / model_id / variant.id


def model_root(root: Path, variant: Variant) -> Path:
    """Каталог или файл распакованной модели для загрузки движком."""
    if variant.format == "gguf":
        return gguf_entry_file(root, variant)
    configs = sorted(root.rglob("config.json"), key=lambda p: (len(p.parts), str(p)))
    for c in configs:
        if any(c.parent.glob("*.safetensors")):
            return c.parent
    raise NotFoundError(
        f"не найден каталог модели с config.json и *.safetensors в {root}",
        hint="проверьте ZIP: внутри нужен каталог с config.json, *.safetensors и токенизатором",
    )


def local_model(cfg: Config, model_id: str, variant: Variant) -> Path | None:
    """Локальный путь модели: файл GGUF или каталог весов."""
    from .artifacts import read_marker

    adir = artifact_model_dir(cfg, model_id, variant)
    if read_marker(adir) is None:
        return None
    try:
        return model_root(adir, variant)
    except NotFoundError:
        return None


def pull_artifact(
    cfg: Config, model_id: str, variant: Variant, ref: ArtifactRef, runner: Runner
) -> Path:
    from . import artifacts

    dest = artifact_model_dir(cfg, model_id, variant)
    artifacts.install(cfg, ref, dest, runner, name=f"{model_id}:{variant.id}")
    if runner.dry_run:
        return dest
    entry = model_root(dest, variant)  # проверка содержимого архива
    runner.info(f"  путь модели для запуска: {entry}")
    return entry


def list_artifact_models(cfg: Config) -> list[dict[str, Any]]:
    from .artifacts import read_marker

    out = []
    root = cfg.models_dir
    if not root.is_dir():
        return out
    for mdir in sorted(p for p in root.iterdir() if p.is_dir()):
        for vdir in sorted(p for p in mdir.iterdir() if p.is_dir() and not p.name.startswith(".")):
            m = read_marker(vdir)
            if m is None:
                continue
            size = sum(f.stat().st_size for f in vdir.rglob("*") if f.is_file())
            out.append(
                {
                    "ref": f"{mdir.name}:{vdir.name}",
                    "path": str(vdir),
                    "uri": m.get("uri"),
                    "size_bytes": size,
                    "installed_at": m.get("installed_at"),
                }
            )
    return out


def remove_artifact_model(cfg: Config, ref: str, runner: Runner) -> int | None:
    """Удалить модель и вернуть объём освобождённой памяти; None, если модели нет."""
    mid, _, vid = ref.partition(":")
    vdir = cfg.models_dir / mid / vid
    if not vid or not (vdir / ".setup-work-artifact.json").exists():
        return None
    size = sum(f.stat().st_size for f in vdir.rglob("*") if f.is_file())
    runner.info(f"Будет удалено [bold]{vdir}[/] ({gib(size)})")
    runner.require_confirmation(f"Удалить распакованную модель {ref}?", destructive=True)
    if not runner.dry_run:
        shutil.rmtree(vdir)
        runner.event("model.remove", ref=ref, bytes=size)
    return size


def gguf_entry_file(snapshot: Path, variant: Variant) -> Path:
    """Найти GGUF или первый шард модели.

    Поиск рекурсивный; при несовпадении шаблона допускается единственная модель.
    Проекторы mmproj* не считаются самостоятельной моделью."""
    matches = sorted({p for pat in variant.files or [] for p in snapshot.rglob(pat)})
    if not matches:
        matches = sorted(p for p in snapshot.rglob("*.gguf") if not p.name.startswith("mmproj"))
    if not matches:
        raise NotFoundError(f"не найдены файлы GGUF по шаблону {variant.files} in {snapshot}")
    shards = [p for p in matches if "-00001-of-" in p.name]
    if shards:
        expected = int(shards[0].name.split("-of-")[1].split(".")[0])
        if len([p for p in matches if "-of-" in p.name]) < expected:
            raise SetupWorkError(
                f"отсутствуют части GGUF: {len(matches)}/{expected} частей найдено",
                hint="проверьте комплект шардов в архиве и повторите make model-pull",
            )
        return shards[0]
    if len(matches) > 1:
        raise UsageError(
            f"несколько GGUF: {[p.name for p in matches]}; уточните files в models.yaml"
        )
    return matches[0]
