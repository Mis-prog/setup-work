from __future__ import annotations

import io
import json
import zipfile

import pytest

import test_corporate
from setup_work import artifacts
from setup_work.config import ArtifactRef
from setup_work.errors import SetupWorkError
from setup_work.runner import Runner

fake_hdfs = test_corporate.fake_hdfs

RUN = Runner(assume_yes=True, interactive=False)


def zip_bytes(files):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
    return stream.getvalue()


def test_hdfs_independent_zip_parts(cfg, fake_hdfs):
    folder = fake_hdfs / "weights"
    folder.mkdir()
    # Проверяем числовой порядок: 10 следует после 9, а не после 1.
    for index in range(12, 0, -1):
        (folder / f"Qwen_Qwen3.8-27B-{index}-of-12.zip").write_bytes(
            zip_bytes({f"model/{index}.safetensors": bytes([index]), "model/config.json": b"{}"})
        )
    cfg.corporate.hdfs.kerberos = "off"
    ref = ArtifactRef(uri="hdfs:///weights", format="zip-parts")
    assert artifacts.zip_parts(cfg, ref)[1].endswith("-2-of-12.zip")
    dest = cfg.models_dir / "model"
    artifacts.install(cfg, ref, dest, RUN, name="qwen")
    assert len(list((dest / "model").glob("*.safetensors"))) == 12
    assert not list((cfg.artifacts_dir / "downloads").iterdir())
    marker = json.loads((dest / artifacts.MARKER).read_text())
    assert marker["format"] == "zip-parts"
    # Установленная модель доступна даже при недоступном источнике.
    for part in folder.iterdir():
        part.unlink()
    assert artifacts.install(cfg, ref, dest, RUN, name="qwen").reused


def test_raw_zip_parts_are_joined(cfg, tmp_path):
    folder = tmp_path / "parts"
    folder.mkdir()
    payload = zip_bytes({"model/config.json": b"{}", "model/weights.safetensors": b"w" * 1000})
    chunks = [payload[:100], payload[100:500], payload[500:]]
    for index, chunk in enumerate(chunks, 1):
        (folder / f"Model-{index}-of-3.zip").write_bytes(chunk)
    dest = cfg.models_dir / "joined"
    artifacts.install(cfg, ArtifactRef(uri=str(folder), format="zip-parts"), dest, RUN, name="raw")
    assert (dest / "model/weights.safetensors").read_bytes() == b"w" * 1000
    assert len(list(folder.iterdir())) == 3  # исходники не удаляются
    assert not list((cfg.artifacts_dir / "downloads").iterdir())


@pytest.mark.parametrize(
    "names,match",
    [
        (["M-1-of-3.zip", "M-3-of-3.zip"], "отсутствуют части: 2"),
        (["M-1-of-1.zip", "Other-1-of-1.zip"], "найдено комплектов: 2"),
        (["M-01-of-2.zip", "M-1-of-2.zip"], "повторяющийся"),
        (["M-0-of-2.zip"], "неверный"),
        (["unrelated.zip"], "найдено комплектов: 0"),
    ],
)
def test_incomplete_or_ambiguous_parts_fail_before_download(cfg, tmp_path, names, match):
    for name in names:
        (tmp_path / name).touch()
    with pytest.raises(SetupWorkError, match=match):
        artifacts.zip_parts(cfg, ArtifactRef(uri=str(tmp_path), format="zip-parts"))
    assert not cfg.artifacts_dir.exists()


def test_conflicting_parts_preserve_old_model(cfg, tmp_path):
    folder = tmp_path / "conflicts"
    folder.mkdir()
    for index in (1, 2):
        (folder / f"M-{index}-of-2.zip").write_bytes(zip_bytes({"weights": bytes([index])}))
    dest = cfg.models_dir / "existing"
    dest.mkdir(parents=True)
    (dest / "old").write_text("working")
    with pytest.raises(SetupWorkError, match="разные версии"):
        artifacts.install(
            cfg, ArtifactRef(uri=str(folder), format="zip-parts"), dest, RUN, name="bad"
        )
    assert (dest / "old").read_text() == "working"
    assert not (dest.parent / ".existing.unpacking").exists()


def test_direct_gguf_from_hdfs(cfg, fake_hdfs):
    (fake_hdfs / "weights.gguf").write_bytes(b"GGUF")
    cfg.corporate.hdfs.kerberos = "off"
    dest = cfg.models_dir / "gguf"
    artifacts.install(cfg, ArtifactRef(uri="hdfs:///weights.gguf"), dest, RUN, name="gguf")
    assert (dest / "weights.gguf").read_bytes() == b"GGUF"
