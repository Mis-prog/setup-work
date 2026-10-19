from __future__ import annotations

import pytest

from setup_work import hub
from setup_work.errors import NotFoundError, SetupWorkError
from setup_work.models import resolve


def test_gguf_shard_entry(cfg, tmp_path):
    _, v = resolve(cfg, "qwen2.5-coder-32b:q4_k_m")
    for i in (1, 2, 3):
        (tmp_path / f"qwen2.5-coder-32b-instruct-q4_k_m-0000{i}-of-00003.gguf").write_bytes(b"")
    assert hub.gguf_entry_file(tmp_path, v).name.endswith("00001-of-00003.gguf")
    (tmp_path / "qwen2.5-coder-32b-instruct-q4_k_m-00003-of-00003.gguf").unlink()
    with pytest.raises(SetupWorkError, match="отсутствуют части"):
        hub.gguf_entry_file(tmp_path, v)


def test_require_artifact_without_source(cfg):
    _, v = resolve(cfg, "qwen3-8b:q4_k_m")
    with pytest.raises(NotFoundError, match="не имеет источника артефакта"):
        hub.require_artifact(cfg, "qwen3-8b", v)


def test_local_model_absent_without_artifact(cfg):
    _, v = resolve(cfg, "qwen3-8b:q4_k_m")
    assert hub.local_model(cfg, "qwen3-8b", v) is None


def test_list_artifact_models_empty(cfg):
    assert hub.list_artifact_models(cfg) == []
