"""Локальный HTTP-прокси: пересылка API и сбор финального usage без журналирования текстов."""

from __future__ import annotations

import contextlib
import http.client
import json
import ssl
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from .config import Config
from .errors import SetupWorkError, UsageError
from .metrics import _enabled, _label, normalize_usage, record_usage, upload
from .runner import Runner

MAX_BODY = 16 * 1024 * 1024
MAX_EVENT = 1024 * 1024
HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "content-length",
}
USAGE_KEYS = {
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "input_tokens",
    "output_tokens",
    "precached_prompt_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "prompt_tokens_details",
    "completion_tokens_details",
    "input_tokens_details",
    "output_tokens_details",
}


class UsageObserver:
    """SSE usage является накопительным: обновляем поля, не складываем chunks."""

    def __init__(self, streaming: bool, model: str = "unknown") -> None:
        self.streaming = streaming
        self.model = model
        self.usage: dict[str, Any] = {}
        self.buffer = b""
        self.data: list[bytes] = []
        self.data_size = 0
        self.overflow = False

    def _consume(self, raw: bytes) -> None:
        try:
            event = json.loads(raw)
        except (ValueError, UnicodeError):
            return
        if not isinstance(event, dict):
            return
        for value in (event.get("response"), event.get("message"), event):
            if not isinstance(value, dict):
                continue
            if isinstance(value.get("model"), str):
                self.model = value["model"][:200]
            usage = value.get("usage")
            if isinstance(usage, dict):
                for key in USAGE_KEYS & usage.keys():
                    if key.endswith("_details") and isinstance(usage[key], dict):
                        allowed = {"cached_tokens", "reasoning_tokens"}
                        self.usage[key] = {k: v for k, v in usage[key].items() if k in allowed}
                    elif type(usage[key]) is int:
                        self.usage[key] = usage[key]

    def feed(self, chunk: bytes) -> None:
        if self.overflow:
            return
        if not self.streaming:
            if len(self.buffer) + len(chunk) > MAX_BODY:
                self.buffer = b""
                self.overflow = True
            else:
                self.buffer += chunk
            return
        self.buffer += chunk
        while b"\n" in self.buffer:
            line, self.buffer = self.buffer.split(b"\n", 1)
            line = line.rstrip(b"\r")
            if line.startswith(b"data:"):
                part = line[5:].lstrip(b" ")
                self.data_size += len(part)
                if self.data_size > MAX_EVENT:
                    self.overflow = True
                    break
                self.data.append(part)
            elif not line:
                self._consume(b"\n".join(self.data))
                self.data = []
                self.data_size = 0
        if len(self.buffer) > MAX_EVENT or self.overflow:
            self.buffer = b""
            self.data = []
            self.overflow = True

    def finish(self) -> None:
        if self.streaming:
            if self.buffer:
                self.feed(b"\n\n")
            elif self.data:
                self._consume(b"\n".join(self.data))
        elif not self.overflow:
            self._consume(self.buffer)
        self.buffer = b""
        self.data = []


def create_proxy(
    cfg: Config,
    runner: Runner,
    *,
    agent: str,
    upstream: str,
    port: int = 8766,
    include_usage: bool = True,
) -> ThreadingHTTPServer:
    _enabled(cfg)
    _label(agent, "agent")
    target = urlsplit(upstream)
    if (
        target.scheme not in ("http", "https")
        or not target.hostname
        or target.username
        or target.password
        or target.query
        or target.fragment
    ):
        raise UsageError("upstream: нужен HTTP(S) URL без пароля, query и fragment")
    try:
        upstream_port = target.port
    except ValueError as exc:
        raise UsageError("upstream: неверный порт") from exc
    if target.hostname in ("127.0.0.1", "localhost") and (upstream_port or 80) == port and port:
        raise UsageError("upstream совпадает с адресом прокси")

    class Handler(BaseHTTPRequestHandler):
        # HTTP/1.0 с EOF framing: можно пересылать SSE сразу, не буферизуя весь ответ.
        protocol_version = "HTTP/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            pass  # Стандартный access log содержит пути и потенциальные секреты в URL.

        def do_GET(self) -> None:
            self._forward()

        def do_POST(self) -> None:
            self._forward()

        def _forward(self) -> None:
            if self.headers.get("Transfer-Encoding"):
                self.send_error(411, "Content-Length required")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.send_error(400, "Invalid Content-Length")
                return
            if length < 0 or length > MAX_BODY:
                self.send_error(413, "Request too large")
                return
            self.connection.settimeout(120)
            try:
                body = self.rfile.read(length) if length else None
            except OSError:
                return
            if body is not None and len(body) != length:
                self.send_error(400, "Incomplete request")
                return
            request_model = "unknown"
            if body:
                try:
                    request = json.loads(body)
                except (ValueError, UnicodeError):
                    request = None
                if isinstance(request, dict):
                    if isinstance(request.get("model"), str):
                        request_model = request["model"][:200]
                    if (
                        include_usage
                        and request.get("stream") is True
                        and urlsplit(self.path).path.rstrip("/").endswith("/chat/completions")
                    ):
                        opts = request.get("stream_options")
                        if opts is None or isinstance(opts, dict):
                            request["stream_options"] = {**(opts or {}), "include_usage": True}
                            body = json.dumps(request).encode()
            path = urlsplit(self.path)
            if path.scheme or path.netloc or not path.path.startswith("/"):
                self.send_error(400, "Expected relative API path")
                return
            # --upstream http://host:port/v1 и локальный /v1/chat/completions.
            prefix = target.path.rstrip("/")
            suffix = path.path
            if prefix and (suffix == prefix or suffix.startswith(prefix + "/")):
                suffix = suffix[len(prefix) :]
            upstream_path = prefix + suffix + ("?" + path.query if path.query else "")
            connection_headers = {
                h.strip().lower() for h in self.headers.get("Connection", "").split(",")
            }
            headers = {
                k: v
                for k, v in self.headers.items()
                if k.lower() not in HOP_HEADERS | connection_headers | {"host", "accept-encoding"}
            }
            headers["Accept-Encoding"] = "identity"
            if body is not None:
                headers["Content-Length"] = str(len(body))
            kwargs: dict[str, Any] = {"timeout": 120}
            connection_type = http.client.HTTPConnection
            if target.scheme == "https":
                connection_type = http.client.HTTPSConnection
                kwargs["context"] = ssl.create_default_context()
            conn = connection_type(target.hostname, upstream_port, **kwargs)
            observer = UsageObserver(False, request_model)
            status, outcome, sent_headers = 502, "error", False
            started = time.monotonic()
            request_id = uuid.uuid4().hex
            try:
                conn.request(self.command, upstream_path, body, headers)
                response = conn.getresponse()
                status = response.status
                outcome = "ok" if 200 <= status < 300 else "error"
                observer.streaming = "text/event-stream" in response.getheader("Content-Type", "")
                response_connection = {
                    h.strip().lower() for h in response.getheader("Connection", "").split(",")
                }
                self.send_response(status)
                for name, value in response.getheaders():
                    if name.lower() not in HOP_HEADERS | response_connection | {"server", "date"}:
                        self.send_header(name, value)
                self.send_header("Connection", "close")
                self.end_headers()
                sent_headers = True
                while chunk := response.read1(64 * 1024):
                    if response.getheader("Content-Encoding", "identity") == "identity":
                        observer.feed(chunk)
                    self.wfile.write(chunk)
                    self.wfile.flush()
                if response.length not in (None, 0):
                    outcome = "interrupted"
            except (OSError, http.client.HTTPException):
                outcome = "interrupted" if sent_headers else "error"
                if not sent_headers:
                    with contextlib.suppress(OSError):
                        self.send_error(502, "Upstream unavailable")
            finally:
                conn.close()
                observer.finish()
                # GET /models и health-check не являются обращениями к модели.
                if self.command == "POST" and path.path.rstrip("/").endswith(
                    ("/chat/completions", "/completions", "/responses", "/messages")
                ):
                    try:
                        normalize_usage(observer.usage)
                    except UsageError:
                        observer.usage = {}
                        runner.warn("API вернул неверный usage; запрос будет отмечен как missing")
                    try:
                        record_usage(
                            cfg,
                            agent=agent,
                            model=observer.model or "unknown",
                            request_id=request_id,
                            usage=observer.usage,
                            http_status=status,
                            outcome=outcome,
                            duration_ms=int((time.monotonic() - started) * 1000),
                        )
                    except SetupWorkError:
                        runner.warn("не удалось записать usage; проверьте очередь metrics status")
                self.close_connection = True

    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError as exc:
        raise SetupWorkError(
            "не удалось запустить прокси метрик; проверьте локальный порт"
        ) from exc
    server.daemon_threads = False
    return server


def serve_proxy(cfg: Config, runner: Runner, **kwargs: Any) -> None:
    if runner.dry_run:
        runner.info("(dry-run) запустить локальный прокси метрик и периодическую выгрузку")
        return
    server = create_proxy(cfg, runner, **kwargs)
    stop = threading.Event()

    def uploader() -> None:
        while not stop.is_set():
            if cfg.corporate.metrics.hdfs_uri:
                try:
                    upload(cfg, runner)
                except SetupWorkError:
                    runner.warn(
                        "HDFS недоступен: метрики остаются в очереди; выгрузка будет повторена"
                    )
            stop.wait(cfg.corporate.metrics.upload_interval_s)

    thread = threading.Thread(target=uploader, name="metrics-uploader", daemon=True)
    thread.start()
    runner.info(
        f"Метрики {kwargs['agent']}: http://127.0.0.1:{server.server_port}; Ctrl+C — остановить"
    )
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        stop.set()
        server.server_close()
        thread.join(timeout=1)
