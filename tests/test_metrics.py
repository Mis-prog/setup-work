from __future__ import annotations

import concurrent.futures
import json
import sqlite3
import sys

import pytest
from pydantic import ValidationError

from setup_work import metrics
from setup_work.cli import main
from setup_work.config import MetricsConfig
from setup_work.errors import SetupWorkError, UsageError
from setup_work.runner import Runner


@pytest.fixture
def enabled(cfg):
    cfg.corporate.metrics = MetricsConfig(enabled=True, hdfs_uri="hdfs:///team/usage")
    cfg.corporate.hdfs.kerberos = "off"
    return cfg


def record(cfg, request_id="req-1", **kwargs):
    return metrics.record_usage(
        cfg,
        agent="qwen-code",
        model="GigaChat",
        request_id=request_id,
        usage={"prompt_tokens": 100, "completion_tokens": 20},
        **kwargs,
    )


def payloads(cfg):
    with sqlite3.connect(metrics.metrics_dir(cfg) / "usage.sqlite3") as conn:
        return [json.loads(r[0]) for r in conn.execute("SELECT payload FROM events")]


@pytest.mark.parametrize(
    "usage,expected",
    [
        (
            {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "precached_prompt_tokens": 30,
            },
            (100, 20, 120, 30, 0, 0),
        ),
        (
            {
                "input_tokens": 100,
                "output_tokens": 20,
                "input_tokens_details": {"cached_tokens": 40},
                "output_tokens_details": {"reasoning_tokens": 10},
            },
            (100, 20, 120, 40, 0, 10),
        ),
        (
            {
                "input_tokens": 10,
                "output_tokens": 20,
                "cache_read_input_tokens": 80,
                "cache_creation_input_tokens": 10,
            },
            (100, 20, 120, 80, 10, 0),
        ),
        ({"prompt_tokens": 0, "completion_tokens": 0}, (0, 0, 0, 0, 0, 0)),
    ],
)
def test_normalize_usage(usage, expected):
    assert tuple(metrics.normalize_usage(usage).values()) == expected


@pytest.mark.parametrize("usage", [None, {}, {"output_tokens": 10}, {"total_tokens": 50}])
def test_absent_or_partial_usage_is_unknown(usage):
    assert metrics.normalize_usage(usage) is None


@pytest.mark.parametrize(
    "usage",
    [
        {"input_tokens": -1, "output_tokens": 0},
        {"input_tokens": True, "output_tokens": 0},
        {"input_tokens": "10", "output_tokens": 0},
        {"input_tokens": 1.1, "output_tokens": 0},
        {"input_tokens": 1, "output_tokens": 0, "total_tokens": 20},
        {"input_tokens": 1, "output_tokens": 0, "input_tokens_details": {"cached_tokens": 2}},
    ],
)
def test_invalid_usage_rejected(usage):
    with pytest.raises(UsageError):
        metrics.normalize_usage(usage)


@pytest.mark.parametrize(
    "uri",
    ["/local/path", "https://host/a", "-bad", "hdfs:///a/*", "hdfs:///a/../b", "hdfs:///a\nb"],
)
def test_destination_validation(uri):
    with pytest.raises(ValidationError):
        MetricsConfig(hdfs_uri=uri)


def test_disabled_and_status_do_not_create_files(cfg):
    assert metrics.summary(cfg)["events"] == 0
    with pytest.raises(UsageError, match="отключён"):
        record(cfg)
    assert not metrics.metrics_dir(cfg).exists()


def test_record_is_durable_and_deduplicated(enabled):
    first = record(enabled, timestamp="2026-10-20T01:00:00+03:00")
    assert first["recorded"] is True
    assert record(enabled) == {"event_id": first["event_id"], "recorded": False}
    event = payloads(enabled)[0]
    assert event["timestamp"].startswith("2026-10-19T22:00:00")
    assert event["device_id"]
    assert event["schema_version"] == 1
    assert metrics.summary(enabled)["pending_events"] == 1
    assert metrics.summary(enabled)["total_tokens"] == 120
    assert (metrics.metrics_dir(enabled) / "usage.sqlite3").stat().st_mode & 0o077 == 0
    with pytest.raises(UsageError, match="другими метриками"):
        metrics.record_usage(
            enabled,
            agent="qwen-code",
            model="other",
            request_id="req-1",
            usage={"input_tokens": 1, "output_tokens": 1},
        )


def test_only_whitelisted_metadata_saved(enabled):
    metrics.record_usage(
        enabled,
        agent="pi",
        model="model",
        request_id="1",
        usage={
            "input_tokens": 3,
            "output_tokens": 2,
            "prompt": "SECRET PROMPT",
            "api_key": "SECRET KEY",
        },
    )
    text = json.dumps(payloads(enabled))
    assert "SECRET" not in text and "api_key" not in text
    metrics.record_usage(
        enabled,
        agent="pi",
        model="model",
        request_id="2",
        usage=None,
        http_status=502,
        outcome="error",
    )
    summary = metrics.summary(enabled)
    assert summary["total_tokens"] == 5 and summary["missing_usage"] == 1
    assert summary["failed_requests"] == 1
    assert payloads(enabled)[1]["total_tokens"] is None


def test_concurrent_recording_keeps_all_events(enabled):
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda n: record(enabled, str(n)), range(32)))
    assert all(r["recorded"] for r in results)
    assert metrics.summary(enabled)["events"] == 32
    assert len({e["device_id"] for e in payloads(enabled)}) == 1


def test_naive_timestamp_rejected(enabled):
    with pytest.raises(UsageError, match="ISO 8601"):
        record(enabled, timestamp="2026-10-20T01:00:00")
    assert not metrics.metrics_dir(enabled).exists()


@pytest.fixture
def hdfs(tmp_path, monkeypatch, enabled):
    root = tmp_path / "fake-hdfs"
    root.mkdir()
    script = tmp_path / "hdfs"
    script.write_text(
        f"#!{sys.executable}\n"
        + """
import os, pathlib, shutil, sys
from urllib.parse import urlsplit
root = pathlib.Path(os.environ['FAKE_HDFS_ROOT'])
args = sys.argv[2:]
op = args[0]
def path(uri):
    return root / urlsplit(uri).path.lstrip('/')
try:
    if os.environ.get('FAKE_HDFS_FAIL') == op:
        sys.exit(1)
    if op == '-test':
        sys.exit(0 if path(args[-1]).exists() else 1)
    elif op == '-mkdir':
        path(args[-1]).mkdir(parents=True, exist_ok=True)
    elif op == '-put':
        shutil.copyfile(args[-2], path(args[-1]))
    elif op == '-mv':
        target = path(args[-1])
        if target.exists():
            sys.exit(1)
        path(args[-2]).rename(target)
        if os.environ.get('FAKE_HDFS_LOST_ACK'):
            sys.exit(1)
    elif op == '-cat':
        sys.stdout.write(path(args[-1]).read_text())
    else:
        sys.exit(2)
except OSError:
    sys.exit(1)
"""
    )
    script.chmod(0o755)
    enabled.corporate.hdfs.cli = str(script)
    monkeypatch.setenv("FAKE_HDFS_ROOT", str(root))
    return root


def test_upload_groups_by_utc_date_and_is_repeatable(enabled, hdfs):
    enabled.corporate.metrics.batch_size = 2
    for n in range(5):
        record(enabled, str(n), timestamp=f"2026-10-{19 + n % 2}T01:00:00Z")
    result = metrics.upload(enabled, Runner(json_mode=True))
    assert result == {"uploaded_batches": 3, "uploaded_events": 5}
    files = list(hdfs.rglob("*.jsonl"))
    events = [json.loads(line) for file in files for line in file.read_text().splitlines()]
    assert len({e["event_id"] for e in events}) == 5
    assert {p.parent.parent.name for p in files} == {"dt=2026-10-19", "dt=2026-10-20"}
    assert metrics.summary(enabled)["pending_events"] == 0
    assert metrics.upload(enabled, Runner()) == {"uploaded_batches": 0, "uploaded_events": 0}
    assert list(hdfs.rglob("*.jsonl")) == files


@pytest.mark.parametrize("stage", ["-mkdir", "-put", "-mv"])
def test_failed_upload_keeps_queue_then_retries(enabled, hdfs, monkeypatch, stage):
    record(enabled)
    monkeypatch.setenv("FAKE_HDFS_FAIL", stage)
    with pytest.raises(SetupWorkError):
        metrics.upload(enabled, Runner(json_mode=True))
    assert metrics.summary(enabled)["pending_events"] == 1
    assert not list(hdfs.rglob("*.jsonl"))
    monkeypatch.delenv("FAKE_HDFS_FAIL")
    assert metrics.upload(enabled, Runner(json_mode=True))["uploaded_events"] == 1
    assert len(list(hdfs.rglob("*.jsonl"))) == 1
    assert not list(hdfs.rglob("*.pending"))


def test_lost_hdfs_ack_does_not_duplicate_and_preserves_original_destination(
    enabled, hdfs, monkeypatch
):
    record(enabled)
    monkeypatch.setenv("FAKE_HDFS_LOST_ACK", "1")
    with pytest.raises(SetupWorkError):
        metrics.upload(enabled, Runner(json_mode=True))
    assert len(list(hdfs.rglob("*.jsonl"))) == 1
    assert metrics.summary(enabled)["pending_events"] == 1
    monkeypatch.delenv("FAKE_HDFS_LOST_ACK")
    enabled.corporate.metrics.hdfs_uri = "hdfs:///different/destination"
    metrics.upload(enabled, Runner(json_mode=True))
    assert len(list(hdfs.rglob("*.jsonl"))) == 1
    assert not (hdfs / "different").exists()
    assert metrics.summary(enabled)["pending_events"] == 0


def test_existing_corrupt_batch_is_not_acknowledged(enabled, hdfs, monkeypatch):
    record(enabled)
    monkeypatch.setenv("FAKE_HDFS_LOST_ACK", "1")
    with pytest.raises(SetupWorkError):
        metrics.upload(enabled, Runner(json_mode=True))
    next(hdfs.rglob("*.jsonl")).write_text("corrupted")
    monkeypatch.delenv("FAKE_HDFS_LOST_ACK")
    with pytest.raises(SetupWorkError, match="не совпадает"):
        metrics.upload(enabled, Runner(json_mode=True))
    assert metrics.summary(enabled)["pending_events"] == 1


def test_upload_dry_run_does_not_create_state(enabled):
    result = metrics.upload(enabled, Runner(dry_run=True))
    assert result["dry_run"] is True
    assert not metrics.metrics_dir(enabled).exists()


def test_metrics_cli_json_and_record(tmp_path, cfg, capsys):
    cfg.user_dir.mkdir(parents=True, exist_ok=True)
    (cfg.user_dir / "corporate.yaml").write_text("metrics: {enabled: true}\n")
    args = [
        "metrics",
        "record",
        "--agent",
        "pi",
        "--model",
        "test",
        "--request-id",
        "r1",
        "--input-tokens",
        "10",
        "--output-tokens",
        "2",
        "--json",
    ]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["recorded"] is True
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["recorded"] is False
    assert main(["metrics", "status", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["total_tokens"] == 12 and data["pending_events"] == 1
    assert main(["metrics", "upload"]) == 2
    assert "metrics.hdfs_uri" in capsys.readouterr().err


def test_record_cli_dry_run_and_response_file(cfg, tmp_path, capsys):
    response = tmp_path / "response.json"
    response.write_text(
        json.dumps(
            {
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                "choices": [{"message": {"content": "PRIVATE"}}],
            }
        )
    )
    args = [
        "metrics",
        "record",
        "--agent",
        "pi",
        "--model",
        "test",
        "--request-id",
        "r1",
        "--response-file",
        str(response),
        "--json",
    ]
    assert main(["--dry-run", *args]) == 0
    assert json.loads(capsys.readouterr().out)["dry_run"] is True
    assert not metrics.metrics_dir(cfg).exists()
    cfg.user_dir.mkdir(parents=True, exist_ok=True)
    (cfg.user_dir / "corporate.yaml").write_text("metrics: {enabled: true}\n")
    assert main(args) == 0
    capsys.readouterr()
    assert metrics.summary(cfg)["total_tokens"] == 5
    assert "PRIVATE" not in json.dumps(payloads(cfg))
