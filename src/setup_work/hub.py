"""Hugging Face download and cache management.

* The cache location follows huggingface_hub (``HF_HOME`` / ``HF_HUB_CACHE``)
  unless ``paths.model_cache_dir`` or ``$SETUP_WORK_MODEL_CACHE`` is set.
* Authentication uses existing credentials (``HF_TOKEN`` or ``hf auth login``);
  tokens are never read, printed or stored by setup-work.
* Downloads are planned first (sizes from Hub metadata, disk check, already-cached
  bytes), confirmed above a threshold, resumable (huggingface_hub keeps
  ``.incomplete`` files), and verified against LFS sha256 digests.
* Nothing in this module is called by ``doctor`` or ``recommend``.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import ArtifactRef, Config, Variant
from .errors import (
    AbortedError,
    NotFoundError,
    SetupWorkError,
    UsageError,
)
from .memory import gib
from .models import resolve
from .runner import Runner

DEFAULT_IGNORE = ["original/*", "metal/*", "*.pth", "*.pt", "*.bin.index.json"]


def cache_dir(cfg: Config) -> Path:
    if cfg.model_cache_dir:
        return cfg.model_cache_dir
    # Read the environment at call time (huggingface_hub caches it at import).
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser() / "hub"
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE)


@dataclass
class RemoteFile:
    name: str
    size: int
    sha256: str | None


@dataclass
class PullPlan:
    ref: str
    repository: str
    revision: str | None
    allow_patterns: list[str] | None
    ignore_patterns: list[str] | None
    files: list[RemoteFile] = field(default_factory=list)
    cached_bytes: int = 0
    cache_dir: Path | None = None
    free_bytes: int | None = None
    resolved_revision: str | None = None

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.files)

    @property
    def needed_bytes(self) -> int:
        return max(0, self.total_bytes - self.cached_bytes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "repository": self.repository,
            "revision": self.revision,
            "resolved_revision": self.resolved_revision,
            "files": [f.__dict__ for f in self.files],
            "total_bytes": self.total_bytes,
            "cached_bytes": self.cached_bytes,
            "needed_bytes": self.needed_bytes,
            "cache_dir": str(self.cache_dir),
            "free_bytes": self.free_bytes,
        }


def _matches(name: str, allow: list[str] | None, ignore: list[str] | None) -> bool:
    if allow and not any(fnmatch.fnmatch(name, p) for p in allow):
        return False
    return not (ignore and any(fnmatch.fnmatch(name, p) for p in ignore))


def target_for(cfg: Config, ref: str) -> tuple[str, str | None, list[str] | None, list[str] | None]:
    """Map a ref to (repository, revision, allow_patterns, ignore_patterns).

    Accepts ``catalog-id:variant`` or a raw ``org/repo`` (optionally ``org/repo@rev``).
    """
    if "/" in ref and ":" not in ref:
        repo, _, rev = ref.partition("@")
        return repo, rev or None, None, DEFAULT_IGNORE
    model, variant = resolve(cfg, ref)
    if variant is not None and not variant.repository:
        raise UsageError(
            f"{ref} has no Hugging Face repository; it is distributed as an artifact",
            hint="pull it with the corporate source (default) instead of SOURCE=hf",
        )
    if variant is None:
        raise UsageError(
            f"specify which variant of {ref} to download",
            hint="variants: "
            + ", ".join(f"{ref}:{v.id} ({v.size_gb:g} GB)" for v in model.variants),
        )
    allow = list(variant.files) if variant.files else None
    if allow and variant.format == "safetensors":
        allow += ["*.json", "*.txt", "*.model", "*.jinja"]
    return variant.repository, variant.revision, allow, None if allow else DEFAULT_IGNORE


def _hub_errors() -> tuple[type[BaseException], ...]:
    from huggingface_hub import errors

    return (
        errors.GatedRepoError,
        errors.RepositoryNotFoundError,
        errors.RevisionNotFoundError,
        errors.HfHubHTTPError,
    )


def _translate(e: BaseException, repo: str) -> SetupWorkError:
    from huggingface_hub import errors

    if isinstance(e, errors.GatedRepoError):
        return SetupWorkError(
            f"{repo} is gated: your account has not been granted access",
            hint=f"accept the license at https://huggingface.co/{repo} and authenticate with "
            "`hf auth login` or HF_TOKEN",
        )
    if isinstance(e, errors.RepositoryNotFoundError):
        return NotFoundError(
            f"repository {repo} not found (or private without credentials)",
            hint="check the id; for private repos run `hf auth login`",
        )
    if isinstance(e, errors.RevisionNotFoundError):
        return NotFoundError(f"revision not found in {repo}")
    return SetupWorkError(f"Hugging Face request failed for {repo}: {e}")


def plan_pull(cfg: Config, ref: str, revision: str | None = None) -> PullPlan:
    from huggingface_hub import HfApi

    repo, rev, allow, ignore = target_for(cfg, ref)
    rev = revision or rev
    try:
        info = HfApi().model_info(repo, revision=rev, files_metadata=True, timeout=30)
    except _hub_errors() as e:
        raise _translate(e, repo) from e
    files = []
    for s in info.siblings or []:
        if _matches(s.rfilename, allow, ignore):
            sha = s.lfs.sha256 if s.lfs else None
            files.append(RemoteFile(s.rfilename, int(s.size or 0), sha))
    if not files:
        raise NotFoundError(f"no files in {repo} match {allow or 'the default patterns'}")
    cdir = cache_dir(cfg)
    plan = PullPlan(
        ref, repo, rev, allow, ignore, files, cache_dir=cdir, resolved_revision=info.sha
    )
    probe_dir = cdir
    while not probe_dir.exists() and probe_dir != probe_dir.parent:
        probe_dir = probe_dir.parent
    plan.free_bytes = shutil.disk_usage(probe_dir).free
    snap = cdir / f"models--{repo.replace('/', '--')}" / "snapshots" / (info.sha or "")
    if info.sha and snap.is_dir():
        for f in files:
            p = snap / f.name
            if p.exists() and p.stat().st_size == f.size:
                plan.cached_bytes += f.size
    return plan


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verified_db(cfg: Config) -> Path:
    return cfg.state_dir / "verified-files.json"


def verify_files(cfg: Config, root: Path, files: list[RemoteFile], runner: Runner) -> list[str]:
    """Compare LFS sha256 digests. Results are cached by (resolved path, size, mtime)."""
    db_path = _verified_db(cfg)
    try:
        db = json.loads(db_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        db = {}
    bad: list[str] = []
    for f in files:
        if not f.sha256:
            continue
        p = (root / f.name).resolve()
        if not p.is_file():
            bad.append(f"{f.name}: missing after download")
            continue
        st = p.stat()
        key = f"{p}|{st.st_size}|{int(st.st_mtime)}"
        if db.get(key) == f.sha256:
            continue
        runner.info(f"  verifying sha256 of {f.name} ({gib(f.size)}) ...")
        digest = _sha256(p)
        if digest != f.sha256:
            bad.append(f"{f.name}: sha256 mismatch (expected {f.sha256[:12]}…, got {digest[:12]}…)")
        else:
            db[key] = digest
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db_path.write_text(json.dumps(db, indent=1), encoding="utf-8")
    except OSError:
        pass
    return bad


def pull(cfg: Config, plan: PullPlan, runner: Runner, verify: bool | None = None) -> Path:
    from huggingface_hub import snapshot_download

    threshold = cfg.defaults.downloads.confirm_above_gb * 1e9
    margin = 1 * 1024**3
    if plan.free_bytes is not None and plan.needed_bytes + margin > plan.free_bytes:
        raise SetupWorkError(
            f"not enough disk space in {plan.cache_dir}: need {gib(plan.needed_bytes)} "
            f"(+1 GiB margin), {gib(plan.free_bytes)} free",
            hint="free space or set SETUP_WORK_MODEL_CACHE / HF_HUB_CACHE to a larger volume",
        )
    if plan.needed_bytes == 0:
        runner.info(f"[green]already cached[/] {plan.repository} ({gib(plan.total_bytes)})")
    elif plan.needed_bytes > threshold:
        runner.require_confirmation(
            f"Download {gib(plan.needed_bytes)} from {plan.repository} into {plan.cache_dir}?"
        )
    if runner.dry_run:
        runner.info(f"(dry-run) would download {len(plan.files)} file(s) from {plan.repository}")
        return plan.cache_dir or Path()
    runner.event("model.pull.start", repo=plan.repository, bytes=plan.needed_bytes)
    try:
        path = snapshot_download(
            plan.repository,
            revision=plan.resolved_revision or plan.revision,
            cache_dir=str(cache_dir(cfg)),
            allow_patterns=plan.allow_patterns,
            ignore_patterns=plan.ignore_patterns,
        )
    except KeyboardInterrupt:
        runner.event("model.pull.interrupted", repo=plan.repository)
        raise AbortedError(
            "download interrupted", hint="re-run the same command to resume; partial files are kept"
        ) from None
    except _hub_errors() as e:
        raise _translate(e, plan.repository) from e
    root = Path(path)
    do_verify = cfg.defaults.downloads.verify_checksums if verify is None else verify
    if do_verify:
        bad = verify_files(cfg, root, plan.files, runner)
        if bad:
            raise SetupWorkError(
                "checksum verification failed:\n  " + "\n  ".join(bad),
                hint=f"remove the cached repo (make model-remove MODEL={plan.repository}) "
                "and retry",
            )
    runner.event("model.pull.end", repo=plan.repository, path=str(root))
    return root


def local_snapshot(cfg: Config, variant: Variant) -> Path | None:
    """Return the cached HF snapshot directory if every wanted file is present."""
    if not variant.repository:
        return None
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    allow = list(variant.files) if variant.files else None
    try:
        path = Path(
            snapshot_download(
                variant.repository,
                revision=variant.revision,
                cache_dir=str(cache_dir(cfg)),
                allow_patterns=allow,
                ignore_patterns=None if allow else DEFAULT_IGNORE,
                local_files_only=True,
            )
        )
    except (LocalEntryNotFoundError, FileNotFoundError, OSError):
        return None
    except Exception:
        return None
    if allow and not any(any(path.glob(p)) for p in allow):
        return None
    return path


# ------------------------------------------------------------ corporate artifacts


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
        hint="the archive must contain a Hugging Face style model directory",
    )


def local_model(cfg: Config, model_id: str, variant: Variant) -> Path | None:
    """Local path to load (GGUF entry file or model dir), from artifacts or the HF cache."""
    from .artifacts import read_marker

    adir = artifact_model_dir(cfg, model_id, variant)
    if read_marker(adir) is not None:
        try:
            return model_root(adir, variant)
        except NotFoundError:
            return None
    if not variant.repository:
        return None
    snap = local_snapshot(cfg, variant)
    if snap is None:
        return None
    return gguf_entry_file(snap, variant) if variant.format == "gguf" else snap


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


@dataclass
class CachedRepo:
    repo_id: str
    size_bytes: int
    revisions: list[str]
    path: str
    catalog_refs: list[str]


def list_cached(cfg: Config) -> list[CachedRepo]:
    from huggingface_hub import scan_cache_dir

    cdir = cache_dir(cfg)
    if not cdir.exists():
        return []
    try:
        info = scan_cache_dir(cdir)
    except Exception as e:  # corrupted caches raise CacheNotFound / CorruptedCacheException
        raise SetupWorkError(f"cannot scan cache {cdir}: {e}") from e
    by_repo: dict[str, list[str]] = {}
    for mid, spec in cfg.models.models.items():
        for v in spec.variants:
            by_repo.setdefault(v.repository, []).append(f"{mid}:{v.id}")
    out = []
    for r in sorted(info.repos, key=lambda r: r.repo_id):
        if r.repo_type != "model":
            continue
        refs = by_repo.get(r.repo_id, [])
        present = []
        for ref in refs:
            spec, var = resolve(cfg, ref)
            if var and var.files:
                if any(
                    fnmatch.fnmatch(f.file_name, p)
                    for rev in r.revisions
                    for f in rev.files
                    for p in var.files
                ):
                    present.append(ref)
            else:
                present.append(ref)
        out.append(
            CachedRepo(
                r.repo_id,
                r.size_on_disk,
                [rev.commit_hash[:12] for rev in r.revisions],
                str(r.repo_path),
                present,
            )
        )
    return out


def remove(cfg: Config, ref: str, runner: Runner) -> int:
    from huggingface_hub import scan_cache_dir

    repo = ref if "/" in ref and ":" not in ref else target_for(cfg, ref)[0]
    cdir = cache_dir(cfg)
    if not cdir.exists():
        raise NotFoundError(f"model cache {cdir} does not exist")
    info = scan_cache_dir(cdir)
    match = [r for r in info.repos if r.repo_id == repo and r.repo_type == "model"]
    if not match:
        raise NotFoundError(f"{repo} is not in the cache {cdir}", hint="see `make model-list`")
    r = match[0]
    shared = [x for x in list_cached(cfg) if x.repo_id == repo]
    refs = shared[0].catalog_refs if shared else []
    runner.info(f"Will delete [bold]{r.repo_path}[/] ({gib(r.size_on_disk)})")
    if len(refs) > 1:
        runner.warn(f"this repository holds several catalog variants: {', '.join(refs)}")
    runner.require_confirmation(f"Delete cached repository {repo}?", destructive=True)
    if runner.dry_run:
        return r.size_on_disk
    strategy = info.delete_revisions(*[rev.commit_hash for rev in r.revisions])
    strategy.execute()
    runner.event("model.remove", repo=repo, bytes=r.size_on_disk)
    return r.size_on_disk
