"""Declarative configuration: schemas, loading and layering.

Configuration is read from two layers:

1. the built-in directory (``$SETUP_WORK_CONFIG_DIR``, the repository ``config/``
   directory, or the copy bundled into the wheel), and
2. the user directory (``$SETUP_WORK_USER_CONFIG_DIR``, default
   ``$XDG_CONFIG_HOME/setup-work`` or ``~/.config/setup-work``).

Files with the same name are deep-merged: mappings merge key by key and the
user layer wins; lists and scalars are replaced. Secrets never live here; recipes
only reference environment variable *names*.
"""

from __future__ import annotations

import os
import re
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .errors import ConfigError

CONFIG_FILES = (
    "defaults.yaml",
    "models.yaml",
    "harnesses.yaml",
    "profiles.yaml",
    "corporate.yaml",
)
ENGINE_NAMES = ("llama.cpp", "vllm")
WORKLOADS = ("general", "coding", "reasoning", "agents", "embedding")
QUANTIZATIONS = ("none", "fp8", "awq", "mxfp4", "q4_k_m", "q5_k_m", "q6_k", "q8_0", "f16")
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)


# --------------------------------------------------------------------------- defaults


class PathsConfig(Strict):
    data_dir: str
    state_dir: str
    bin_dir: str


class InferenceDefaults(Strict):
    host: str = "127.0.0.1"
    port: int = Field(8000, ge=1, le=65535)
    gpu_memory_utilization: float = Field(0.90, gt=0.1, le=0.99)
    vram_safety_margin_gb: float = Field(1.0, ge=0)
    runtime_overhead_gb: dict[str, float] = Field(default_factory=dict)
    ready_timeout_s: int = Field(900, ge=5)
    stop_timeout_s: int = Field(30, ge=1)
    default_kv_cache_dtype: Literal["fp16", "bf16", "fp8", "q8_0"] = "fp16"


class RecommendDefaults(Strict):
    default_context: int = Field(8192, ge=256)
    default_concurrency: int = Field(1, ge=1)
    max_results: int = Field(5, ge=1)
    ram_safety_margin_gb: float = Field(4.0, ge=0)


class DownloadDefaults(Strict):
    confirm_above_gb: float = Field(1.0, ge=0)


class ToolPin(Strict):
    version: str
    min_version: str | None = None
    sha256: dict[str, str] = Field(default_factory=dict)

    @field_validator("sha256")
    @classmethod
    def _hex(cls, v: dict[str, str]) -> dict[str, str]:
        for target, digest in v.items():
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"sha256 for {target!r} is not a 64-char hex digest")
        return v


class Defaults(Strict):
    paths: PathsConfig
    inference: InferenceDefaults = Field(default_factory=InferenceDefaults)
    recommend: RecommendDefaults = Field(default_factory=RecommendDefaults)
    downloads: DownloadDefaults = Field(default_factory=DownloadDefaults)
    tools: dict[str, ToolPin] = Field(default_factory=dict)


# --------------------------------------------------------------------------- models


class Attention(Strict):
    num_layers: int = Field(ge=1)
    num_attention_heads: int = Field(ge=1)
    num_kv_heads: int = Field(ge=1)
    head_dim: int = Field(ge=1)
    sliding_window: int | None = Field(None, ge=1)
    sliding_window_layers: int = Field(0, ge=0)

    @model_validator(mode="after")
    def _check(self) -> Attention:
        if self.num_attention_heads % self.num_kv_heads:
            raise ValueError("num_attention_heads must be a multiple of num_kv_heads")
        if self.sliding_window_layers > self.num_layers:
            raise ValueError("sliding_window_layers cannot exceed num_layers")
        if self.sliding_window_layers and not self.sliding_window:
            raise ValueError("sliding_window_layers requires sliding_window")
        return self


class MoE(Strict):
    num_experts: int = Field(ge=2)
    experts_per_token: int = Field(ge=1)


class Variant(Strict):
    id: str
    format: Literal["safetensors", "gguf"]
    quantization: str = "none"
    dtype: Literal["bf16", "fp16", "fp32"] | None = None
    repository: str | None = None  # upstream repo id (org/name), reference only
    source: str | None = None  # corporate artifact URI (hdfs://, /path, https://)
    sha256: str | None = None  # of the artifact archive, when `source` is used
    files: list[str] | None = None
    size_gb: float = Field(gt=0)
    engines: list[str]
    provenance: Literal["official", "community"] = "official"

    @field_validator("repository")
    @classmethod
    def _repo(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not re.fullmatch(r"[A-Za-z0-9][\w.-]*/[\w.-]+", v):
            raise ValueError(f"invalid repository id {v!r}; expected org/name")
        return v

    @field_validator("quantization")
    @classmethod
    def _quant(cls, v: str) -> str:
        if v not in QUANTIZATIONS:
            raise ValueError(f"unknown quantization {v!r}; expected one of {QUANTIZATIONS}")
        return v

    @field_validator("engines")
    @classmethod
    def _engines(cls, v: list[str]) -> list[str]:
        for e in v:
            if e not in ENGINE_NAMES:
                raise ValueError(f"unknown engine {e!r}; expected one of {ENGINE_NAMES}")
        return v

    @model_validator(mode="after")
    def _format_rules(self) -> Variant:
        if self.format == "gguf":
            if not self.files:
                raise ValueError("gguf variants must list their files")
            if "vllm" in self.engines:
                raise ValueError("vLLM GGUF support is experimental; list gguf for llama.cpp only")
        if self.format == "safetensors" and "llama.cpp" in self.engines:
            raise ValueError("llama.cpp only loads GGUF files")
        if self.quantization == "none" and self.dtype is None:
            raise ValueError("unquantised variants must declare dtype")
        if not self.repository and not self.source:
            raise ValueError("a variant needs an upstream `repository` or an artifact `source`")
        if self.source:
            validate_artifact_uri(self.source)
        return self

    @property
    def origin(self) -> str:
        return self.repository or self.source or "?"

    @property
    def bytes(self) -> int:
        return int(self.size_gb * 1e9)


class ModelSpec(Strict):
    family: str
    display_name: str
    license: str
    architecture: Literal["dense", "moe"]
    parameters_b: float = Field(gt=0)
    active_parameters_b: float | None = Field(None, gt=0)
    workloads: list[str]
    quality: dict[str, float] = Field(default_factory=dict)
    context_length: int = Field(ge=512)
    attention: Attention
    moe: MoE | None = None
    fp16_safe: bool = True
    tags: list[str] = Field(default_factory=list)
    variants: list[Variant]

    @field_validator("workloads")
    @classmethod
    def _workloads(cls, v: list[str]) -> list[str]:
        for w in v:
            if w not in WORKLOADS:
                raise ValueError(f"unknown workload {w!r}; expected one of {WORKLOADS}")
        return v

    @model_validator(mode="after")
    def _moe(self) -> ModelSpec:
        if self.architecture == "moe":
            if self.active_parameters_b is None or self.moe is None:
                raise ValueError("MoE models must declare active_parameters_b and moe")
            if self.active_parameters_b >= self.parameters_b:
                raise ValueError("active_parameters_b must be smaller than parameters_b")
        ids = [v.id for v in self.variants]
        if len(ids) != len(set(ids)):
            raise ValueError("variant ids must be unique")
        if not self.variants:
            raise ValueError("a model needs at least one variant")
        return self

    @property
    def effective_active_b(self) -> float:
        return self.active_parameters_b or self.parameters_b

    def variant(self, variant_id: str) -> Variant | None:
        return next((v for v in self.variants if v.id == variant_id), None)


class ModelCatalog(Strict):
    models: dict[str, ModelSpec]

    @field_validator("models")
    @classmethod
    def _names(cls, v: dict[str, ModelSpec]) -> dict[str, ModelSpec]:
        for name in v:
            if not _NAME_RE.match(name):
                raise ValueError(f"invalid model id {name!r}")
        return v


# --------------------------------------------------------------------------- harnesses


class HarnessInstall(Strict):
    method: Literal["npm", "uv", "none"] | None = None
    python: str | None = None
    # One argv (["npm", "run", "build"]) or a list of argvs run in order in the checkout.
    build: list[str] | list[list[str]] | None = None

    @property
    def build_steps(self) -> list[list[str]]:
        if not self.build:
            return []
        if all(isinstance(x, str) for x in self.build):
            return [list(self.build)]  # type: ignore[arg-type]
        return [list(x) for x in self.build]  # type: ignore[union-attr]


class BinarySpec(Strict):
    """A binary to expose. `path` is relative to the checkout (git/local) or install dir."""

    name: str
    path: str | None = None
    interpreter: Literal["node", "python", "bun", "sh"] | None = None

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not re.fullmatch(r"[\w.-]+", v) or v.startswith("."):
            raise ValueError(f"invalid binary name {v!r}")
        return v

    @field_validator("path")
    @classmethod
    def _path(cls, v: str | None) -> str | None:
        if v is not None and (v.startswith("/") or ".." in Path(v).parts):
            raise ValueError(f"binary path {v!r} must be relative and stay inside the checkout")
        return v


class HarnessEnvironment(Strict):
    required: list[str] = Field(default_factory=list)
    optional: list[str] = Field(default_factory=list)

    @field_validator("required", "optional")
    @classmethod
    def _names(cls, v: list[str]) -> list[str]:
        for name in v:
            if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", name):
                raise ValueError(f"invalid environment variable name {name!r}")
        return v


_VERSION_RE = re.compile(r"^(latest|[0-9][0-9A-Za-z.+-]*)$")
_GIT_URL_RE = re.compile(
    r"^(https://[\w.-]+(:\d+)?/[\w./~-]+|ssh://[\w.@-]+(:\d+)?/[\w./~-]+|[\w.-]+@[\w.-]+:[\w./~-]+)$"
)
_REF_RE = re.compile(r"^[\w][\w./-]*$")


class HarnessSpec(Strict):
    description: str = ""
    source: Literal["npm", "pypi", "git", "local"]
    package: str | None = None
    version: str | None = None
    repository: str | None = None
    ref: str | None = None
    path: str | None = None
    install: HarnessInstall = Field(default_factory=HarnessInstall)
    binaries: list[str | BinarySpec] = Field(default_factory=list)
    requires: dict[str, str] = Field(default_factory=dict)
    requires_commands: list[str] = Field(default_factory=list)
    environment: HarnessEnvironment = Field(default_factory=HarnessEnvironment)
    verify: list[str] | None = None

    @model_validator(mode="after")
    def _source_rules(self) -> HarnessSpec:
        s = self.source
        if s in ("npm", "pypi"):
            if not self.package:
                raise ValueError(f"source {s} requires 'package'")
            if not self.version:
                raise ValueError(f"source {s} requires 'version' (use an exact version)")
            if not _VERSION_RE.match(self.version):
                raise ValueError(f"invalid version {self.version!r}")
            if s == "npm" and not re.fullmatch(
                r"(@[a-z0-9][\w.-]*/)?[a-z0-9][\w.-]*", self.package
            ):
                raise ValueError(f"invalid npm package name {self.package!r}")
            if s == "pypi" and not re.fullmatch(r"[A-Za-z0-9][\w.-]*", self.package):
                raise ValueError(f"invalid PyPI package name {self.package!r}")
        if s == "git":
            if not self.repository:
                raise ValueError("source git requires 'repository'")
            if not _GIT_URL_RE.match(self.repository):
                raise ValueError(
                    f"repository {self.repository!r} must be an https://, ssh:// or "
                    "user@host:path URL (no file://, ext:: or option-like values)"
                )
            if not self.ref:
                raise ValueError("source git requires 'ref' (branch, tag or commit) for pinning")
            if not _REF_RE.match(self.ref) or ".." in self.ref:
                raise ValueError(f"invalid git ref {self.ref!r}")
        if s == "local" and not self.path:
            raise ValueError("source local requires 'path'")
        if s in ("git", "local") and self.install.method is None:
            raise ValueError(f"source {s} requires install.method (npm, uv or none)")
        for b in self.binaries:
            name = b if isinstance(b, str) else b.name
            if not re.fullmatch(r"[\w.-]+", name) or name.startswith("."):
                raise ValueError(f"invalid binary name {name!r}")
        if self.install.build is not None and (
            not self.install.build or any(not argv for argv in self.install.build_steps)
        ):
            raise ValueError("install.build must be a non-empty argv list (or list of them)")
        for k in self.requires:
            if k not in ("node", "python"):
                raise ValueError(f"unsupported requirement {k!r} (node, python)")
        return self

    @property
    def binary_specs(self) -> list[BinarySpec]:
        return [BinarySpec(name=b) if isinstance(b, str) else b for b in self.binaries]

    @property
    def git_host(self) -> str | None:
        if not self.repository:
            return None
        m = re.match(r"^(?:https://|ssh://)?(?:[\w.-]+@)?([\w.-]+)", self.repository)
        return m.group(1).lower() if m else None

    @property
    def pinned(self) -> bool:
        if self.source in ("npm", "pypi"):
            return self.version not in (None, "latest")
        if self.source == "git":
            return bool(self.ref and re.fullmatch(r"[0-9a-f]{7,40}|v?\d[\w.-]*", self.ref))
        return True


class HarnessCatalog(Strict):
    harnesses: dict[str, HarnessSpec]

    @field_validator("harnesses")
    @classmethod
    def _names(cls, v: dict[str, HarnessSpec]) -> dict[str, HarnessSpec]:
        for name in v:
            if not _NAME_RE.match(name):
                raise ValueError(f"invalid harness name {name!r} (lowercase, digits, . _ -)")
        return v


# --------------------------------------------------------------------------- profiles


class ToolSet(Strict):
    required: list[str] = Field(default_factory=list)
    optional: list[str] = Field(default_factory=list)


class ProfileModels(Strict):
    preferences: dict[str, Any] = Field(default_factory=dict)
    pull: list[str] = Field(default_factory=list)


class ProfileSpec(Strict):
    description: str = ""
    extends: list[str] = Field(default_factory=list)
    tools: ToolSet = Field(default_factory=ToolSet)
    harnesses: ToolSet = Field(default_factory=ToolSet)
    engines: list[str] = Field(default_factory=list)
    models: ProfileModels = Field(default_factory=ProfileModels)
    inference: dict[str, Any] = Field(default_factory=dict)
    validation: list[str] = Field(default_factory=list)


class ProfileCatalog(Strict):
    profiles: dict[str, ProfileSpec]


# --------------------------------------------------------------------------- corporate

_ARTIFACT_URI_RE = re.compile(
    r"^(hdfs://[\w.-]*(:\d+)?/\S+|hdfs:/\S+|viewfs://\S+|file:///\S+|/\S+|~/\S+|https://\S+)$"
)
ARCHIVE_FORMATS = ("zip", "tar.gz", "tar", "dir", "file")


def validate_artifact_uri(uri: str) -> str:
    if not _ARTIFACT_URI_RE.match(uri) or any(c in uri for c in "\n\r\0"):
        raise ValueError(
            f"artifact URI {uri!r} must be hdfs://, viewfs://, https://, file:// "
            "or an absolute path"
        )
    return uri


class ArtifactRef(Strict):
    uri: str
    sha256: str | None = None
    format: Literal["zip", "tar.gz", "tar", "dir", "file"] | None = None
    keep_archive: bool | None = None
    description: str = ""

    @field_validator("uri")
    @classmethod
    def _uri(cls, v: str) -> str:
        return validate_artifact_uri(v)

    @field_validator("sha256")
    @classmethod
    def _sha(cls, v: str | None) -> str | None:
        if v is not None and not re.fullmatch(r"[0-9a-f]{64}", v):
            raise ValueError("sha256 must be a 64-char lowercase hex digest")
        return v

    @property
    def effective_format(self) -> str:
        if self.format:
            return self.format
        u = self.uri.lower()
        if u.endswith(".zip"):
            return "zip"
        if u.endswith((".tar.gz", ".tgz")):
            return "tar.gz"
        if u.endswith(".tar"):
            return "tar"
        if u.endswith((".gguf", ".xz", ".tar.xz")):
            return "file"
        return "dir"


class EngineArtifact(ArtifactRef):
    # Path of llama-server / the venv root inside the archive; discovered when omitted.
    path: str | None = None


class HdfsConfig(Strict):
    cli: str = "hdfs"
    kerberos: Literal["auto", "required", "off"] = "auto"
    extra_args: list[str] = Field(default_factory=list)
    timeout_s: int = Field(6 * 3600, ge=60)


class CorporateScript(Strict):
    name: str
    description: str = ""
    argv: list[str]
    creates: list[str] = Field(default_factory=list)  # skip when all of these exist

    @field_validator("argv")
    @classmethod
    def _argv(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("argv must not be empty")
        return v


class CorporateConfig(Strict):
    hdfs: HdfsConfig = Field(default_factory=HdfsConfig)
    scripts: list[CorporateScript] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    bridge_pip_conf: bool = True
    trusted_git_hosts: list[str] = Field(default_factory=list)
    models: dict[str, ArtifactRef] = Field(default_factory=dict)  # "model:variant" -> artifact
    engines: dict[str, EngineArtifact] = Field(default_factory=dict)  # llama.cpp | vllm
    tools: dict[str, ArtifactRef] = Field(default_factory=dict)  # uv | node | git-lfs archives

    @field_validator("env")
    @classmethod
    def _env(cls, v: dict[str, str]) -> dict[str, str]:
        for k in v:
            if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", k):
                raise ValueError(f"invalid environment variable name {k!r}")
            if re.search(r"(TOKEN|SECRET|PASSWORD|API_KEY)", k):
                raise ValueError(f"{k}: secrets must not be stored in configuration")
        return v

    @field_validator("engines")
    @classmethod
    def _engines(cls, v: dict[str, EngineArtifact]) -> dict[str, EngineArtifact]:
        for k in v:
            if k not in ENGINE_NAMES:
                raise ValueError(f"unknown engine {k!r}")
        return v

    @property
    def enabled(self) -> bool:
        return bool(self.scripts or self.env or self.models or self.engines or self.tools)


# --------------------------------------------------------------------------- loading


def deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        out = dict(base)
        for k, v in override.items():
            out[k] = deep_merge(base[k], v) if k in base else v
        return out
    return override


# Catalog files: an entry defined in a later layer replaces the earlier entry as a
# whole (overriding the npm `qwen-code` recipe with a git fork must not inherit
# `package:`). Settings files are deep-merged.
_CATALOG_KEYS = {
    "models.yaml": "models",
    "harnesses.yaml": "harnesses",
    "profiles.yaml": "profiles",
}


def _merge_layer(name: str, base: dict[str, Any], layer: dict[str, Any]) -> dict[str, Any]:
    key = _CATALOG_KEYS.get(name)
    if key is None:
        return deep_merge(base, layer)
    out = {k: v for k, v in base.items() if k != key}
    out.update({k: v for k, v in layer.items() if k != key})
    entries = dict(base.get(key) or {})
    entries.update(layer.get(key) or {})
    out[key] = entries
    return out


_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_path(value: str, env: dict[str, str] | None = None) -> Path:
    """Expand ``${VAR:-default}``, ``$VAR`` and ``~`` in a configured path."""
    env = os.environ if env is None else env

    def sub(m: re.Match[str]) -> str:
        val = env.get(m.group(1))
        return val if val else (m.group(2) or "")

    s = _VAR_RE.sub(sub, value)
    s = os.path.expandvars(s)
    return Path(s).expanduser()


def builtin_config_dir() -> Path:
    env = os.environ.get("SETUP_WORK_CONFIG_DIR")
    if env:
        return Path(env).expanduser()
    repo = Path(__file__).resolve().parents[2] / "config"
    if (repo / "defaults.yaml").is_file():
        return repo
    return Path(__file__).resolve().parent / "_config"


def team_config_dir() -> Path | None:
    """Shared team layer (e.g. a cloned config repo), between built-in and user."""
    env = os.environ.get("SETUP_WORK_TEAM_CONFIG_DIR")
    return Path(env).expanduser() if env else None


def user_config_dir() -> Path:
    env = os.environ.get("SETUP_WORK_USER_CONFIG_DIR")
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CONFIG_HOME")
    return (Path(xdg) if xdg else Path.home() / ".config") / "setup-work"


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError(f"{path}: invalid YAML: {e}") from e
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


def _fmt_validation(path: str, err: ValidationError) -> str:
    lines = [f"{path}: {err.error_count()} validation error(s)"]
    for e in err.errors():
        loc = ".".join(str(p) for p in e["loc"])
        lines.append(f"  - {loc}: {e['msg']}")
    return "\n".join(lines)


class Config:
    """Loaded, validated, layered configuration."""

    def __init__(self, builtin_dir: Path | None = None, user_dir: Path | None = None) -> None:
        self.builtin_dir = builtin_dir or builtin_config_dir()
        self.user_dir = user_dir or user_config_dir()
        self.team_dir = team_config_dir()
        if not (self.builtin_dir / "defaults.yaml").is_file():
            raise ConfigError(
                f"configuration directory {self.builtin_dir} has no defaults.yaml",
                hint="set SETUP_WORK_CONFIG_DIR to the repository's config/ directory",
            )
        self.sources: dict[str, list[Path]] = {}
        self.raw: dict[str, dict[str, Any]] = {}
        for name in CONFIG_FILES:
            merged: dict[str, Any] = {}
            used: list[Path] = []
            for d in (self.builtin_dir, self.team_dir, self.user_dir):
                if d is None:
                    continue
                p = d / name
                if p.is_file():
                    merged = _merge_layer(name, merged, _read_yaml(p))
                    used.append(p)
            self.raw[name] = merged
            self.sources[name] = used

    def _parse(self, name: str, model: type[BaseModel]) -> Any:
        try:
            return model.model_validate(self.raw[name])
        except ValidationError as e:
            where = " + ".join(str(p) for p in self.sources[name]) or name
            raise ConfigError(_fmt_validation(where, e)) from e

    @cached_property
    def defaults(self) -> Defaults:
        return self._parse("defaults.yaml", Defaults)

    @cached_property
    def models(self) -> ModelCatalog:
        return self._parse("models.yaml", ModelCatalog)

    @cached_property
    def harnesses(self) -> HarnessCatalog:
        return self._parse("harnesses.yaml", HarnessCatalog)

    @cached_property
    def profiles(self) -> ProfileCatalog:
        return self._parse("profiles.yaml", ProfileCatalog)

    @cached_property
    def corporate(self) -> CorporateConfig:
        return self._parse("corporate.yaml", CorporateConfig)

    def validate_all(self) -> list[str]:
        """Return a list of problems (empty when everything is valid)."""
        problems: list[str] = []
        for attr in ("defaults", "models", "harnesses", "profiles", "corporate"):
            try:
                getattr(self, attr)
            except ConfigError as e:
                problems.append(e.message)
        if problems:
            return problems
        names = set(self.profiles.profiles)
        harnesses = set(self.harnesses.harnesses)
        for pname, p in self.profiles.profiles.items():
            for parent in p.extends:
                if parent not in names:
                    problems.append(f"profile {pname}: extends unknown profile {parent!r}")
            for h in p.harnesses.required + p.harnesses.optional:
                if h not in harnesses:
                    problems.append(f"profile {pname}: unknown harness {h!r}")
            for e in p.engines:
                if e not in ENGINE_NAMES:
                    problems.append(f"profile {pname}: unknown engine {e!r}")
        for ref in self.corporate.models:
            mid, _, vid = ref.partition(":")
            spec = self.models.models.get(mid)
            if spec is None or not vid or spec.variant(vid) is None:
                problems.append(f"corporate.models: {ref!r} is not a catalog model:variant")
        return problems

    # Resolved paths -------------------------------------------------------------

    @property
    def data_dir(self) -> Path:
        return expand_path(self.defaults.paths.data_dir)

    @property
    def state_dir(self) -> Path:
        return expand_path(self.defaults.paths.state_dir)

    @property
    def bin_dir(self) -> Path:
        return expand_path(self.defaults.paths.bin_dir)

    @property
    def harness_dir(self) -> Path:
        return self.data_dir / "harnesses"

    @property
    def engines_dir(self) -> Path:
        return self.data_dir / "engines"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @property
    def models_dir(self) -> Path:
        v = os.environ.get("SETUP_WORK_MODEL_DIR")
        return expand_path(v) if v else self.data_dir / "models"

    @property
    def tools_dir(self) -> Path:
        return self.data_dir / "tools"

    def as_dict(self) -> dict[str, Any]:
        return {
            "builtin_dir": str(self.builtin_dir),
            "team_dir": str(self.team_dir) if self.team_dir else None,
            "user_dir": str(self.user_dir),
            "sources": {k: [str(p) for p in v] for k, v in self.sources.items()},
            "resolved_paths": {
                "data_dir": str(self.data_dir),
                "state_dir": str(self.state_dir),
                "bin_dir": str(self.bin_dir),
                "harness_dir": str(self.harness_dir),
                "engines_dir": str(self.engines_dir),
                "models_dir": str(self.models_dir),
            },
            "defaults": self.defaults.model_dump(by_alias=True),
        }
