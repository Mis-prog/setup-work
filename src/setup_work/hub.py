"""Model storage: weights come from corporate artifacts only.

* A variant's artifact is taken from ``corporate.models`` (``"<id>:<variant>"``) or
  from the variant's own ``source:``; see ``artifacts.py`` for HDFS, local/NFS
  paths and HTTPS.
* Archives are unpacked into ``<models_dir>/<id>/<variant>`` and marked, so re-runs
  reuse them.
* Nothing in this module is called by ``doctor`` or ``recommend``.
"""

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
            f"{model_id}:{variant.id} has no artifact source",
            hint=f'map "{model_id}:{variant.id}" in corporate.yaml (models:) or give the '
            "variant a `source:`",
        )
    return ref


def model_artifact(cfg: Config, model_id: str, variant: Variant) -> ArtifactRef | None:
    """Artifact source for a variant: corporate.models mapping first, then `source:`."""
    ref = cfg.corporate.models.get(f"{model_id}:{variant.id}")
    if ref is not None:
        return ref
    if variant.source:
        return ArtifactRef(uri=variant.source, sha256=variant.sha256)
    return None


def artifact_model_dir(cfg: Config, model_id: str, variant: Variant) -> Path:
    return cfg.models_dir / model_id / variant.id


def model_root(root: Path, variant: Variant) -> Path:
    """Directory/file the engine should load from an unpacked artifact."""
    if variant.format == "gguf":
        return gguf_entry_file(root, variant)
    configs = sorted(root.rglob("config.json"), key=lambda p: (len(p.parts), str(p)))
    for c in configs:
        if any(c.parent.glob("*.safetensors")):
            return c.parent
    raise NotFoundError(
        f"no model directory (config.json + *.safetensors) found in {root}",
        hint="the archive must contain a model directory (config.json + *.safetensors)",
    )


def local_model(cfg: Config, model_id: str, variant: Variant) -> Path | None:
    """Local path to load (GGUF entry file or model dir) of an unpacked artifact."""
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
    entry = model_root(dest, variant)  # validates the archive content
    runner.info(f"  model entry point: {entry}")
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
    """Remove an unpacked artifact model. Returns freed bytes, or None if not present."""
    mid, _, vid = ref.partition(":")
    vdir = cfg.models_dir / mid / vid
    if not vid or not (vdir / ".setup-work-artifact.json").exists():
        return None
    size = sum(f.stat().st_size for f in vdir.rglob("*") if f.is_file())
    runner.info(f"Will delete [bold]{vdir}[/] ({gib(size)})")
    runner.require_confirmation(f"Delete unpacked model {ref}?", destructive=True)
    if not runner.dry_run:
        shutil.rmtree(vdir)
        runner.event("model.remove", ref=ref, bytes=size)
    return size


def gguf_entry_file(snapshot: Path, variant: Variant) -> Path:
    """The file to hand to llama.cpp: the only file, or shard 00001 of a split model.

    Searches recursively (archives often contain a top-level directory). If the
    catalog patterns do not match (internally renamed files), a single GGUF model
    in the directory is accepted; multimodal projectors (mmproj*) are ignored.
    """
    matches = sorted({p for pat in variant.files or [] for p in snapshot.rglob(pat)})
    if not matches:
        matches = sorted(p for p in snapshot.rglob("*.gguf") if not p.name.startswith("mmproj"))
    if not matches:
        raise NotFoundError(f"no GGUF files matching {variant.files} in {snapshot}")
    shards = [p for p in matches if "-00001-of-" in p.name]
    if shards:
        expected = int(shards[0].name.split("-of-")[1].split(".")[0])
        if len([p for p in matches if "-of-" in p.name]) < expected:
            raise SetupWorkError(
                f"split GGUF is incomplete: {len(matches)}/{expected} shards present",
                hint="re-run make model-pull to finish the download",
            )
        return shards[0]
    if len(matches) > 1:
        raise UsageError(f"ambiguous GGUF files {[p.name for p in matches]}; narrow `files`")
    return matches[0]
