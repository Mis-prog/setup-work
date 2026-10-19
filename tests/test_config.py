from __future__ import annotations

import pytest
from pydantic import ValidationError

from setup_work.config import (
    Config,
    HarnessSpec,
    ModelSpec,
    deep_merge,
    expand_path,
)
from setup_work.errors import ConfigError


def test_builtin_config_is_valid(cfg):
    assert cfg.validate_all() == []
    assert len(cfg.models.models) >= 10
    assert "qwen-code" in cfg.harnesses.harnesses


def test_every_variant_has_known_engine_and_positive_size(cfg):
    for mid, m in cfg.models.models.items():
        for v in m.variants:
            assert v.size_gb > 0, mid
            assert v.engines, mid
            if v.format == "gguf":
                assert v.files, f"{mid}:{v.id}"


def test_moe_models_declare_active_params(cfg):
    for m in cfg.models.models.values():
        if m.architecture == "moe":
            assert m.active_parameters_b and m.active_parameters_b < m.parameters_b


def test_user_overrides_merge(tmp_path, monkeypatch):
    user = tmp_path / "userconf"
    user.mkdir(exist_ok=True)
    (user / "harnesses.yaml").write_text(
        "harnesses:\n"
        "  my-agent:\n"
        "    source: pypi\n"
        "    package: my-agent\n"
        "    version: 1.2.3\n"
        "    binaries: [my-agent]\n"
    )
    (user / "defaults.yaml").write_text("inference:\n  port: 9999\n")
    cfg = Config()
    assert "my-agent" in cfg.harnesses.harnesses
    assert "claude-code" in cfg.harnesses.harnesses  # встроенные записи сохранены
    assert cfg.defaults.inference.port == 9999
    assert cfg.defaults.inference.host == "127.0.0.1"  # остальные ключи сохранены


def test_invalid_yaml_reports_file(tmp_path):
    user = tmp_path / "userconf"
    user.mkdir(exist_ok=True)
    (user / "models.yaml").write_text("models: [unclosed\n")
    with pytest.raises(ConfigError, match="неверный YAML"):
        _ = Config().models


def test_deep_merge_replaces_lists():
    assert deep_merge({"a": {"b": 1, "c": [1]}}, {"a": {"c": [2]}}) == {"a": {"b": 1, "c": [2]}}


def test_expand_path_defaults():
    p = expand_path("${NOPE_NOT_SET:-/opt/x}/y", env={})
    assert str(p) == "/opt/x/y"
    p = expand_path("${HOMEISH:-/opt/x}/y", env={"HOMEISH": "/srv"})
    assert str(p) == "/srv/y"


@pytest.mark.parametrize(
    "spec,match",
    [
        (
            {
                "source": "git",
                "repository": "file:///etc",
                "ref": "main",
                "install": {"method": "uv"},
            },
            "должен быть URL https",
        ),
        (
            {
                "source": "git",
                "repository": "--upload-pack=touch /tmp/x",
                "ref": "main",
                "install": {"method": "uv"},
            },
            "должен быть URL https",
        ),
        (
            {
                "source": "git",
                "repository": "ext::sh -c id",
                "ref": "main",
                "install": {"method": "uv"},
            },
            "должен быть URL https",
        ),
        (
            {
                "source": "git",
                "repository": "https://bitbucket.corp.example/scm/a/b.git",
                "install": {"method": "uv"},
            },
            "заполните ref",
        ),
        (
            {
                "source": "git",
                "repository": "https://bitbucket.corp.example/scm/a/b.git",
                "ref": "-x",
                "install": {"method": "uv"},
            },
            "неверное значение ref для Git",
        ),
        (
            {
                "source": "git",
                "repository": "https://bitbucket.corp.example/scm/a/b.git",
                "ref": "main",
            },
            "install.method",
        ),
        ({"source": "npm", "package": "foo"}, "точную версию в поле version"),
        ({"source": "npm", "package": "Foo Bar", "version": "1.0.0"}, "неверное имя пакета npm"),
        ({"source": "pypi", "package": "x", "version": "1; rm -rf /"}, "неверная версия"),
        (
            {"source": "pypi", "package": "x", "version": "1.0", "binaries": ["../evil"]},
            "неверное имя исполняемого файла",
        ),
        (
            {
                "source": "npm",
                "package": "x",
                "version": "1.0",
                "environment": {"required": ["lower-case"]},
            },
            "неверное имя переменной окружения",
        ),
        ({"source": "local"}, "заполните path"),
        ({"source": "svn", "package": "x"}, "Input should be"),
    ],
)
def test_invalid_harness_definitions(spec, match):
    with pytest.raises(ValidationError, match=match):
        HarnessSpec.model_validate(spec)


def test_valid_private_git_harness_via_ssh():
    h = HarnessSpec.model_validate(
        {
            "source": "git",
            "repository": "ssh://git@bitbucket.corp.example:7999/team/agent.git",
            "ref": "v1.4.0",
            "install": {"method": "uv"},
            "environment": {"required": ["OPENAI_BASE_URL", "OPENAI_API_KEY"]},
        }
    )
    assert h.pinned


def test_branch_ref_is_not_considered_pinned():
    h = HarnessSpec.model_validate(
        {
            "source": "git",
            "repository": "https://bitbucket.corp.example/scm/a/b.git",
            "ref": "main",
            "install": {"method": "uv"},
        }
    )
    assert not h.pinned


def _model(**over):
    base = {
        "family": "x",
        "display_name": "X",
        "license": "mit",
        "architecture": "dense",
        "parameters_b": 1,
        "workloads": ["general"],
        "context_length": 4096,
        "attention": {"num_layers": 2, "num_attention_heads": 4, "num_kv_heads": 2, "head_dim": 8},
        "variants": [
            {
                "id": "q",
                "format": "gguf",
                "quantization": "q8_0",
                "repository": "a/b",
                "files": ["x.gguf"],
                "size_gb": 1,
                "engines": ["llama.cpp"],
            }
        ],
    }
    base.update(over)
    return base


def test_model_validation_rules():
    ModelSpec.model_validate(_model())
    with pytest.raises(ValidationError, match="для MoE заполните"):
        ModelSpec.model_validate(_model(architecture="moe"))
    with pytest.raises(ValidationError, match="для GGUF заполните files"):
        ModelSpec.model_validate(
            _model(
                variants=[
                    {
                        "id": "q",
                        "format": "gguf",
                        "quantization": "q8_0",
                        "repository": "a/b",
                        "size_gb": 1,
                        "engines": ["llama.cpp"],
                    }
                ]
            )
        )
    with pytest.raises(ValidationError, match="для llama.cpp требуется format: gguf"):
        ModelSpec.model_validate(
            _model(
                variants=[
                    {
                        "id": "b",
                        "format": "safetensors",
                        "dtype": "bf16",
                        "repository": "a/b",
                        "size_gb": 1,
                        "engines": ["llama.cpp"],
                    }
                ]
            )
        )
    with pytest.raises(ValidationError, match="неверный идентификатор репозитория"):
        ModelSpec.model_validate(
            _model(
                variants=[
                    {
                        "id": "b",
                        "format": "safetensors",
                        "dtype": "bf16",
                        "repository": "../etc",
                        "size_gb": 1,
                        "engines": ["vllm"],
                    }
                ]
            )
        )
    with pytest.raises(ValidationError, match="кратен num_kv_heads"):
        ModelSpec.model_validate(
            _model(
                attention={
                    "num_layers": 2,
                    "num_attention_heads": 5,
                    "num_kv_heads": 2,
                    "head_dim": 8,
                }
            )
        )
