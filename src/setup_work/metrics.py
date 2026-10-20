"""Учёт usage API, постоянная очередь SQLite и повторяемая выгрузка JSONL в HDFS."""

from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import getpass
import hashlib
import json
import os
import socket
import sqlite3
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .artifacts import hdfs_cli, require_kerberos
from .config import Config
from .errors import SetupWorkError, UsageError
from .runner import Runner

TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cached_input_tokens",
    "cache_creation_input_tokens",
    "reasoning_tokens",
)


def _count(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise UsageError("счётчик токенов должен быть неотрицательным целым числом")
    return value


def normalize_usage(usage: dict[str, Any] | None) -> dict[str, int] | None:
    """OpenAI/GigaChat, Responses и Anthropic; cache/reasoning не суммируются дважды.

    У Anthropic input_tokens исключает cache read/write, у OpenAI включает cache.
    Отсутствие usage или неполные счётчики означают неизвестный расход.
    """
    if not isinstance(usage, dict):
        return None
    chat = "prompt_tokens" in usage
    input_key, output_key = (
        ("prompt_tokens", "completion_tokens") if chat else ("input_tokens", "output_tokens")
    )
    if input_key not in usage or output_key not in usage:
        return None
    inp, out = _count(usage[input_key]), _count(usage[output_key])
    details = usage.get("prompt_tokens_details" if chat else "input_tokens_details") or {}
    output_details = (
        usage.get("completion_tokens_details" if chat else "output_tokens_details") or {}
    )
    if not isinstance(details, dict) or not isinstance(output_details, dict):
        raise UsageError("usage: details должен быть объектом JSON")
    cached = _count(usage.get("precached_prompt_tokens", details.get("cached_tokens", 0)))
    created = 0
    if "cache_read_input_tokens" in usage or "cache_creation_input_tokens" in usage:
        cached = _count(usage.get("cache_read_input_tokens", 0))
        created = _count(usage.get("cache_creation_input_tokens", 0))
        inp += cached + created
    reasoning = _count(output_details.get("reasoning_tokens", 0))
    if cached + created > inp or reasoning > out:
        raise UsageError("usage: вложенные счётчики превышают число input/output токенов")
    total = _count(usage.get("total_tokens", inp + out))
    if total != inp + out:
        raise UsageError("usage.total_tokens не равен сумме input_tokens и output_tokens")
    return dict(zip(TOKEN_FIELDS, (inp, out, total, cached, created, reasoning), strict=True))


def _label(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 200
        or any(ord(c) < 32 for c in value)
    ):
        raise UsageError(f"{name}: нужна непустая строка до 200 символов без управляющих символов")
    return value


def _enabled(cfg: Config) -> None:
    if not cfg.corporate.metrics.enabled:
        raise UsageError(
            "сбор метрик отключён", hint="задайте metrics.enabled: true в corporate.yaml"
        )


def metrics_dir(cfg: Config) -> Path:
    return cfg.state_dir / "metrics"


@contextlib.contextmanager
def _db(cfg: Config, *, create: bool = True) -> Iterator[sqlite3.Connection]:
    folder = metrics_dir(cfg)
    path = folder / "usage.sqlite3"
    try:
        if create:
            folder.mkdir(parents=True, exist_ok=True, mode=0o700)
            # DELETE journal работает и на постоянных томах без поддержки shared-memory WAL.
            fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
            os.close(fd)
        conn = sqlite3.connect(
            str(path) if create else path.as_uri() + "?mode=ro", timeout=30, uri=not create
        )
        conn.row_factory = sqlite3.Row
        try:
            if create:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS events (
                        event_id TEXT PRIMARY KEY, payload TEXT NOT NULL,
                        date TEXT NOT NULL, device_id TEXT NOT NULL, batch_id TEXT
                    );
                    CREATE TABLE IF NOT EXISTS batches (
                        batch_id TEXT PRIMARY KEY, uri TEXT NOT NULL,
                        content TEXT NOT NULL, uploaded INTEGER NOT NULL DEFAULT 0
                    );
                    CREATE INDEX IF NOT EXISTS events_batch ON events(batch_id);
                """)
            yield conn
        finally:
            conn.close()
    except (sqlite3.Error, OSError) as exc:
        raise SetupWorkError(f"не удалось открыть или сохранить очередь метрик: {exc}") from exc


def record_usage(
    cfg: Config,
    *,
    agent: str,
    model: str,
    request_id: str,
    usage: dict[str, Any] | None,
    session_id: str | None = None,
    timestamp: str | None = None,
    http_status: int = 200,
    outcome: str = "ok",
    duration_ms: int | None = None,
) -> dict[str, Any]:
    """Одна запись на реальный запрос API. request_id должен сохраняться при повторе hook.

    Передавайте только финальный usage запроса, а не накопительный итог сессии.
    В базу попадают исключительно перечисленные ниже поля.
    """
    _enabled(cfg)
    for name, value in (("agent", agent), ("model", model), ("request_id", request_id)):
        _label(value, name)
    if session_id is not None:
        _label(session_id, "session_id")
    if outcome not in ("ok", "error", "interrupted") or not 100 <= http_status <= 599:
        raise UsageError("неверные outcome или HTTP status")
    if duration_ms is not None:
        _count(duration_ms)
    counters = normalize_usage(usage)
    try:
        now = (
            dt.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if timestamp
            else (dt.datetime.now(dt.UTC))
        )
        if now.tzinfo is None:
            raise ValueError("нужен часовой пояс")
        now = now.astimezone(dt.UTC)
    except ValueError as exc:
        raise UsageError("timestamp: требуется ISO 8601 с часовым поясом") from exc
    settings = cfg.corporate.metrics
    with _db(cfg) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT OR IGNORE INTO meta VALUES ('installation_id', ?)", (uuid.uuid4().hex,)
        )
        installation = conn.execute(
            "SELECT value FROM meta WHERE key='installation_id'"
        ).fetchone()[0]
        device = (
            settings.device_id
            or hashlib.sha256(f"{installation}:{socket.gethostname()}".encode()).hexdigest()[:32]
        )
        event_id = hashlib.sha256(f"{device}\0{agent}\0{request_id}".encode()).hexdigest()
        payload = {
            "schema_version": 1,
            "event_id": event_id,
            "timestamp": now.isoformat(),
            "device_id": device,
            "user_id": settings.user_id or getpass.getuser(),
            "department": settings.department,
            "agent": agent,
            "model": model,
            "request_id": request_id,
            "session_id": session_id,
            "http_status": http_status,
            "outcome": outcome,
            "duration_ms": duration_ms,
            "usage_status": "reported" if counters is not None else "missing",
            **(counters or dict.fromkeys(TOKEN_FIELDS)),
        }
        existing = conn.execute(
            "SELECT payload FROM events WHERE event_id=?", (event_id,)
        ).fetchone()
        if existing:
            old = json.loads(existing[0])
            for key in (*TOKEN_FIELDS, "model", "session_id", "outcome", "http_status"):
                if old[key] != payload[key]:
                    raise UsageError(
                        "request_id уже записан с другими метриками; используйте новый ID"
                    )
            return {"event_id": event_id, "recorded": False}
        conn.execute(
            "INSERT INTO events VALUES (?, ?, ?, ?, NULL)",
            (
                event_id,
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                now.date().isoformat(),
                device,
            ),
        )
    return {"event_id": event_id, "recorded": True}


def summary(cfg: Config) -> dict[str, Any]:
    result: dict[str, Any] = {
        "enabled": cfg.corporate.metrics.enabled,
        "hdfs_uri": cfg.corporate.metrics.hdfs_uri,
        "queue_dir": str(metrics_dir(cfg)),
        "events": 0,
        "pending_events": 0,
        "missing_usage": 0,
        "failed_requests": 0,
        "groups": [],
        **dict.fromkeys(TOKEN_FIELDS, 0),
    }
    if not (metrics_dir(cfg) / "usage.sqlite3").is_file():
        return result
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    with _db(cfg, create=False) as conn:
        for row in conn.execute("""
            SELECT e.payload, COALESCE(b.uploaded, 0) AS uploaded FROM events e
            LEFT JOIN batches b ON e.batch_id=b.batch_id
        """):
            event = json.loads(row[0])
            result["events"] += 1
            result["pending_events"] += not row[1]
            result["missing_usage"] += event["usage_status"] == "missing"
            result["failed_requests"] += event["outcome"] != "ok"
            key = (event["timestamp"][:10], event["agent"], event["model"])
            group = groups.setdefault(
                key,
                {
                    "date": key[0],
                    "agent": key[1],
                    "model": key[2],
                    "requests": 0,
                    "missing_usage": 0,
                    **dict.fromkeys(TOKEN_FIELDS, 0),
                },
            )
            group["requests"] += 1
            group["missing_usage"] += event["usage_status"] == "missing"
            for field in TOKEN_FIELDS:
                count = event[field] or 0
                result[field] += count
                group[field] += count
    result["groups"] = [groups[key] for key in sorted(groups)]
    return result


def _prepare_batch(cfg: Config, conn: sqlite3.Connection) -> sqlite3.Row | None:
    with conn:
        conn.execute("BEGIN IMMEDIATE")
        pending = conn.execute(
            "SELECT * FROM batches WHERE uploaded=0 ORDER BY rowid LIMIT 1"
        ).fetchone()
        if pending:
            return pending  # URI и содержимое не меняются между повторами.
        first = conn.execute(
            "SELECT date, device_id FROM events WHERE batch_id IS NULL LIMIT 1"
        ).fetchone()
        if first is None:
            return None
        rows = conn.execute(
            """
            SELECT event_id, payload FROM events WHERE batch_id IS NULL AND date=? AND device_id=?
            ORDER BY event_id LIMIT ?
        """,
            (*first, cfg.corporate.metrics.batch_size),
        ).fetchall()
        batch = uuid.uuid4().hex
        uri = (
            f"{cfg.corporate.metrics.hdfs_uri.rstrip('/')}/dt={first[0]}/"
            f"device_id={first[1]}/{batch}.jsonl"
        )
        conn.execute(
            "INSERT INTO batches VALUES (?, ?, ?, 0)",
            (
                batch,
                uri,
                "".join(row[1] + "\n" for row in rows),
            ),
        )
        conn.executemany(
            "UPDATE events SET batch_id=? WHERE event_id=?", ((batch, row[0]) for row in rows)
        )
        return conn.execute("SELECT * FROM batches WHERE batch_id=?", (batch,)).fetchone()


def upload(cfg: Config, runner: Runner) -> dict[str, Any]:
    """Не удалять очередь при ошибке; immutable batch + rename исключают частичные JSONL."""
    _enabled(cfg)
    destination = cfg.corporate.metrics.hdfs_uri
    if not destination:
        raise UsageError("задайте metrics.hdfs_uri в corporate.yaml")
    if runner.dry_run:
        return {"dry_run": True, "destination": destination, **summary(cfg)}
    if not (metrics_dir(cfg) / "usage.sqlite3").is_file():
        return {"uploaded_batches": 0, "uploaded_events": 0}
    h = cfg.corporate.hdfs
    require_kerberos(h)
    cli = hdfs_cli(h)
    timeout = cfg.corporate.metrics.upload_timeout_s
    base = [cli, "dfs", *h.extra_args]
    result = {"uploaded_batches": 0, "uploaded_events": 0}
    with (metrics_dir(cfg) / "upload.lock").open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SetupWorkError("выгрузка метрик уже выполняется другим процессом") from exc
        with _db(cfg) as conn:
            # Снимок числа записей: постоянный приток новых запросов не задерживает завершение.
            remaining = conn.execute("""
                SELECT count(*) FROM events e LEFT JOIN batches b ON e.batch_id=b.batch_id
                WHERE COALESCE(b.uploaded, 0)=0
            """).fetchone()[0]
            while remaining > 0 and (batch := _prepare_batch(cfg, conn)) is not None:
                uri, content = batch["uri"], batch["content"]
                exists = runner.run(
                    [*base, "-test", "-e", uri],
                    timeout=timeout,
                    capture=True,
                    check=False,
                    mutating=False,
                )
                if exists.ok:
                    remote = runner.run(
                        [*base, "-cat", uri], timeout=timeout, capture=True, mutating=False
                    )
                    if remote.stdout != content:
                        raise SetupWorkError(
                            f"содержимое ранее выгруженного batch не совпадает: {uri}"
                        )
                else:
                    if exists.returncode != 1:
                        raise SetupWorkError("не удалось проверить batch в HDFS; очередь сохранена")
                    folder, filename = uri.rsplit("/", 1)
                    staging = f"{folder}/.{filename}.pending"
                    runner.run([*base, "-mkdir", "-p", folder], timeout=timeout, capture=True)
                    with tempfile.NamedTemporaryFile(
                        mode="w", encoding="utf-8", suffix=".jsonl", dir=metrics_dir(cfg)
                    ) as local:
                        local.write(content)
                        local.flush()
                        runner.run(
                            [*base, "-put", "-f", local.name, staging],
                            timeout=timeout,
                            capture=True,
                        )
                    runner.run([*base, "-mv", staging, uri], timeout=timeout, capture=True)
                with conn:
                    conn.execute(
                        "UPDATE batches SET uploaded=1 WHERE batch_id=?", (batch["batch_id"],)
                    )
                count = content.count("\n")
                result["uploaded_batches"] += 1
                result["uploaded_events"] += count
                remaining -= count
    return result
