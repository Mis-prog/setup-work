"""Схемы, загрузка и объединение конфигурации.

Встроенный слой: SETUP_WORK_CONFIG_DIR, config/ репозитория или файлы пакета.
Пользовательский слой: SETUP_WORK_USER_CONFIG_DIR, SETUP_WORK_HOME/config
или каталог XDG. Пользовательские значения имеют приоритет.
Настройки объединяются по ключам, записи каталогов моделей и харнессов
заменяются целиком. Секреты в YAML не хранятся."""

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
    "corporate.yaml",
)
ENGINE_NAMES = ("llama.cpp", "vllm")
WORKLOADS = ("general", "coding", "reasoning", "agents", "embedding")
QUANTIZATIONS = ("none", "fp8", "awq", "mxfp4", "q4_k_m", "q5_k_m", "q6_k", "q8_0", "f16")
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)


# --------------------------------------------------------------------------- значения по умолчанию


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
                raise ValueError(
                    f"SHA256 для {target!r} должен содержать 64 шестнадцатеричных символа"
                )
        return v


class Defaults(Strict):
    paths: PathsConfig
    inference: InferenceDefaults = Field(default_factory=InferenceDefaults)
    recommend: RecommendDefaults = Field(default_factory=RecommendDefaults)
    downloads: DownloadDefaults = Field(default_factory=DownloadDefaults)
    tools: dict[str, ToolPin] = Field(default_factory=dict)


# --------------------------------------------------------------------------- модели


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
            raise ValueError(
                "num_attention_heads должен быть кратен "
                "num_kv_heads; проверьте attention в config/models.yaml"
            )
        if self.sliding_window_layers > self.num_layers:
            raise ValueError("sliding_window_layers не должен превышать num_layers")
        if self.sliding_window_layers and not self.sliding_window:
            raise ValueError("для sliding_window_layers заполните sliding_window")
        return self


class MoE(Strict):
    num_experts: int = Field(ge=2)
    experts_per_token: int = Field(ge=1)


class Variant(Strict):
    id: str
    format: Literal["safetensors", "gguf"]
    quantization: str = "none"
    dtype: Literal["bf16", "fp16", "fp32"] | None = None
    repository: str | None = None  # идентификатор исходного репозитория org/name, только справочно
    source: str | None = None  # URI корпоративного артефакта: hdfs://, /путь, https://
    sha256: str | None = None  # контрольная сумма архива из source
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
            raise ValueError(f"неверный идентификатор репозитория {v!r}; требуется org/name")
        return v

    @field_validator("quantization")
    @classmethod
    def _quant(cls, v: str) -> str:
        if v not in QUANTIZATIONS:
            raise ValueError(f"неизвестное квантование {v!r}; допустимые значения: {QUANTIZATIONS}")
        return v

    @field_validator("engines")
    @classmethod
    def _engines(cls, v: list[str]) -> list[str]:
        for e in v:
            if e not in ENGINE_NAMES:
                raise ValueError(f"неизвестный движок {e!r}; допустимые значения: {ENGINE_NAMES}")
        return v

    @model_validator(mode="after")
    def _format_rules(self) -> Variant:
        if self.format == "gguf":
            if not self.files:
                raise ValueError("для GGUF заполните files в config/models.yaml")
            if "vllm" in self.engines:
                raise ValueError(
                    "для GGUF укажите engines: [llama.cpp]; "
                    "поддержка GGUF в vLLM здесь не используется"
                )
        if self.format == "safetensors" and "llama.cpp" in self.engines:
            raise ValueError("для llama.cpp требуется format: gguf")
        if self.quantization == "none" and self.dtype is None:
            raise ValueError("для неквантованного варианта заполните dtype в config/models.yaml")
        if not self.repository and not self.source:
            raise ValueError("для варианта заполните repository либо source в config/models.yaml")
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
                raise ValueError(f"неизвестная задача {w!r}; допустимые значения: {WORKLOADS}")
        return v

    @model_validator(mode="after")
    def _moe(self) -> ModelSpec:
        if self.architecture == "moe":
            if self.active_parameters_b is None or self.moe is None:
                raise ValueError("для MoE заполните active_parameters_b и moe в config/models.yaml")
            if self.active_parameters_b >= self.parameters_b:
                raise ValueError("active_parameters_b должно быть меньше parameters_b")
        ids = [v.id for v in self.variants]
        if len(ids) != len(set(ids)):
            raise ValueError("идентификаторы вариантов id должны быть уникальны")
        if not self.variants:
            raise ValueError("добавьте хотя бы один вариант в variants модели")
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
                raise ValueError(f"неверный идентификатор модели {name!r}")
        return v


# --------------------------------------------------------------------------- харнессы


class HarnessInstall(Strict):
    method: Literal["npm", "uv", "none"] | None = None
    python: str | None = None
    # Одна команда argv или список команд, выполняемых по порядку в каталоге исходников.
    build: list[str] | list[list[str]] | None = None

    @property
    def build_steps(self) -> list[list[str]]:
        if not self.build:
            return []
        if all(isinstance(x, str) for x in self.build):
            return [list(self.build)]  # type: ignore[arg-type]
        return [list(x) for x in self.build]  # type: ignore[union-attr]


class BinarySpec(Strict):
    """Исполняемый файл; path задаётся относительно исходников или каталога установки."""

    name: str
    path: str | None = None
    interpreter: Literal["node", "python", "bun", "sh"] | None = None

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not re.fullmatch(r"[\w.-]+", v) or v.startswith("."):
            raise ValueError(f"неверное имя исполняемого файла {v!r}")
        return v

    @field_validator("path")
    @classmethod
    def _path(cls, v: str | None) -> str | None:
        if v is not None and (v.startswith("/") or ".." in Path(v).parts):
            raise ValueError(f"путь {v!r} должен быть относительным и оставаться внутри исходников")
        return v


class HarnessEnvironment(Strict):
    required: list[str] = Field(default_factory=list)
    optional: list[str] = Field(default_factory=list)

    @field_validator("required", "optional")
    @classmethod
    def _names(cls, v: list[str]) -> list[str]:
        for name in v:
            if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", name):
                raise ValueError(f"неверное имя переменной окружения {name!r}")
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
                raise ValueError(f"source {s} требует поле package")
            if not self.version:
                raise ValueError(f"source {s} требует точную версию в поле version")
            if not _VERSION_RE.match(self.version):
                raise ValueError(f"неверная версия {self.version!r}")
            if s == "npm" and not re.fullmatch(
                r"(@[a-z0-9][\w.-]*/)?[a-z0-9][\w.-]*", self.package
            ):
                raise ValueError(f"неверное имя пакета npm {self.package!r}")
            if s == "pypi" and not re.fullmatch(r"[A-Za-z0-9][\w.-]*", self.package):
                raise ValueError(f"неверное имя пакета PyPI {self.package!r}")
        if s == "git":
            if not self.repository:
                raise ValueError("для source: git заполните repository в config/harnesses.yaml")
            if not _GIT_URL_RE.match(self.repository):
                raise ValueError(
                    f"repository {self.repository!r} должен быть URL https://, ssh:// либо "
                    "user@host:path; file://, ext:: и значения, похожие на опции, запрещены"
                )
            if not self.ref:
                raise ValueError("для source: git заполните ref: ветку, тег или коммит")
            if not _REF_RE.match(self.ref) or ".." in self.ref:
                raise ValueError(f"неверное значение ref для Git {self.ref!r}")
        if s == "local" and not self.path:
            raise ValueError("для source: local заполните path")
        if s in ("git", "local") and self.install.method is None:
            raise ValueError(f"source {s} требует install.method: npm, uv либо none")
        for b in self.binaries:
            name = b if isinstance(b, str) else b.name
            if not re.fullmatch(r"[\w.-]+", name) or name.startswith("."):
                raise ValueError(f"неверное имя исполняемого файла {name!r}")
        if self.install.build is not None and (
            not self.install.build or any(not argv for argv in self.install.build_steps)
        ):
            raise ValueError(
                "install.build должен содержать непустую команду argv либо список таких команд"
            )
        for k in self.requires:
            if k not in ("node", "python"):
                raise ValueError(f"неподдерживаемое требование {k!r} (node, python)")
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
                raise ValueError(
                    f"неверное имя харнесса {name!r}(строчные латинские буквы, цифры, . _ -)"
                )
        return v


# ------------------------------ корпоративные настройки

_ARTIFACT_URI_RE = re.compile(
    r"^(hdfs://[\w.-]*(:\d+)?/\S+|hdfs:/\S+|viewfs://\S+|file:///\S+|/\S+|~/\S+|https://\S+)$"
)
ARCHIVE_FORMATS = ("zip", "tar.gz", "tar", "dir", "file")


def validate_artifact_uri(uri: str) -> str:
    if not _ARTIFACT_URI_RE.match(uri) or any(c in uri for c in "\n\r\0"):
        raise ValueError(
            f"URI артефакта {uri!r} должен быть hdfs://, viewfs://, https://, file:// "
            "or an absolute path"
        )
    return uri


class ArtifactRef(Strict):
    uri: str
    sha256: str | None = None
    format: Literal["zip", "zip-parts", "tar.gz", "tar", "dir", "file"] | None = None
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
            raise ValueError(
                "sha256 должен содержать 64 шестнадцатеричных символа в нижнем регистре"
            )
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
    # Путь к llama-server или venv внутри архива; без настройки определяется автоматически.
    path: str | None = None


class HdfsConfig(Strict):
    cli: str = "hdfs"
    kerberos: Literal["auto", "required", "off"] = "auto"
    extra_args: list[str] = Field(default_factory=list)
    timeout_s: int = Field(6 * 3600, ge=60)


class CorporateConfig(Strict):
    hdfs: HdfsConfig = Field(default_factory=HdfsConfig)
    env: dict[str, str] = Field(default_factory=dict)
    trusted_git_hosts: list[str] = Field(default_factory=list)
    models: dict[str, ArtifactRef] = Field(default_factory=dict)  # модель:вариант → артефакт
    engines: dict[str, EngineArtifact] = Field(default_factory=dict)  # llama.cpp | vllm
    tools: dict[str, ArtifactRef] = Field(default_factory=dict)  # архивы uv или Node.js

    @field_validator("env")
    @classmethod
    def _env(cls, v: dict[str, str]) -> dict[str, str]:
        for k in v:
            if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", k):
                raise ValueError(f"неверное имя переменной окружения {k!r}")
            if re.search(r"(TOKEN|SECRET|PASSWORD|API_KEY)", k):
                raise ValueError(
                    f"{k}: уберите секрет из YAML; используйте окружение или файл токена"
                )
        return v

    @field_validator("engines")
    @classmethod
    def _engines(cls, v: dict[str, EngineArtifact]) -> dict[str, EngineArtifact]:
        for k in v:
            if k not in ENGINE_NAMES:
                raise ValueError(f"неизвестный движок {k!r}")
        return v

    @property
    def enabled(self) -> bool:
        return bool(self.env or self.models or self.engines or self.tools)


# --------------------------------------------------------------------------- загрузка


def deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        out = dict(base)
        for k, v in override.items():
            out[k] = deep_merge(base[k], v) if k in base else v
        return out
    return override


# В каталогах запись более позднего слоя заменяет раннюю запись целиком:
# Git-форк qwen-code при замене npm-рецепта не должен наследовать
# package. Обычные настройки объединяются рекурсивно.
_CATALOG_KEYS = {
    "models.yaml": "models",
    "harnesses.yaml": "harnesses",
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
    """Подставить ${VAR:-default}, $VAR и ~ в путь из настроек."""
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


def setup_work_home() -> Path | None:
    """Общий постоянный каталог настроек, данных, состояния, инструментов и кэшей."""
    env = os.environ.get("SETUP_WORK_HOME")
    return Path(env).expanduser() if env else None


def user_config_dir() -> Path:
    env = os.environ.get("SETUP_WORK_USER_CONFIG_DIR")
    if env:
        return Path(env).expanduser()
    home = setup_work_home()
    if home:
        return home / "config"
    xdg = os.environ.get("XDG_CONFIG_HOME")
    return (Path(xdg) if xdg else Path.home() / ".config") / "setup-work"


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError(f"{path}: неверный YAML: {e}") from e
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: верхний уровень YAML должен содержать пары ключ: значение")
    return data


def _fmt_validation(path: str, err: ValidationError) -> str:
    lines = [f"{path}: {err.error_count()} ошибок конфигурации"]
    for e in err.errors():
        loc = ".".join(str(p) for p in e["loc"])
        kind = e["type"]
        expected = {
            "int_parsing": "требуется целое число",
            "int_type": "требуется целое число",
            "float_parsing": "требуется число",
            "float_type": "требуется число",
            "string_type": "требуется строка",
            "missing": "заполните обязательное поле",
            "extra_forbidden": "неизвестное поле; проверьте имя ключа",
            "dict_type": "требуется набор пар ключ: значение",
            "list_type": "требуется список",
            "bool_parsing": "требуется true или false",
            "bool_type": "требуется true или false",
        }
        if isinstance(e.get("input"), str) and "CHANGE-ME" in e["input"]:
            detail = "замените CHANGE-ME реальным значением в этом поле"
        elif kind == "literal_error":
            detail = f"допустимые значения: {e.get('ctx', {}).get('expected', '')}"
        elif kind in ("greater_than", "greater_than_equal", "less_than", "less_than_equal"):
            ctx = e.get("ctx", {})
            relation, key = {
                "greater_than": ("больше", "gt"),
                "greater_than_equal": ("не меньше", "ge"),
                "less_than": ("меньше", "lt"),
                "less_than_equal": ("не больше", "le"),
            }[kind]
            detail = f"значение должно быть {relation} {ctx.get(key)}"
        elif kind == "value_error":
            detail = str(e.get("ctx", {}).get("error", "неверное значение"))
        else:
            detail = expected.get(kind, "неверный тип или значение; сверьте поле с образцом")
        lines.append(f"  - {loc}: {detail}")
    return "\n".join(lines)


class Config:
    """Загруженная и проверенная конфигурация с учётом слоёв."""

    def __init__(self, builtin_dir: Path | None = None, user_dir: Path | None = None) -> None:
        self.builtin_dir = builtin_dir or builtin_config_dir()
        self.user_dir = user_dir or user_config_dir()
        if not (self.builtin_dir / "defaults.yaml").is_file():
            raise ConfigError(
                f"каталог настроек {self.builtin_dir} не содержит defaults.yaml",
                hint="задайте SETUP_WORK_CONFIG_DIR с путём к config/ репозитория",
            )
        self.sources: dict[str, list[Path]] = {}
        self.raw: dict[str, dict[str, Any]] = {}
        for name in CONFIG_FILES:
            merged: dict[str, Any] = {}
            used: list[Path] = []
            for d in (self.builtin_dir, self.user_dir):
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
    def corporate(self) -> CorporateConfig:
        return self._parse("corporate.yaml", CorporateConfig)

    def validate_all(self) -> list[str]:
        """Вернуть список ошибок; пустой список означает успешную проверку."""
        problems: list[str] = []
        for attr in ("defaults", "models", "harnesses", "corporate"):
            try:
                getattr(self, attr)
            except ConfigError as e:
                problems.append(e.message)
        if problems:
            return problems
        for ref in self.corporate.models:
            mid, _, vid = ref.partition(":")
            spec = self.models.models.get(mid)
            if spec is None or not vid or spec.variant(vid) is None:
                problems.append(f"corporate.models: {ref!r} отсутствует в models.yaml")
        return problems

    # итоговые пути -------------------------------------------------------------

    @property
    def data_dir(self) -> Path:
        home = setup_work_home()
        return home / "data" if home else expand_path(self.defaults.paths.data_dir)

    @property
    def state_dir(self) -> Path:
        home = setup_work_home()
        return home / "state" if home else expand_path(self.defaults.paths.state_dir)

    @property
    def bin_dir(self) -> Path:
        home = setup_work_home()
        if home and not os.environ.get("SETUP_WORK_BIN_DIR"):
            return home / "bin"
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
