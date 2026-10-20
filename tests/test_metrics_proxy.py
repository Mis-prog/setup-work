from __future__ import annotations

import contextlib
import http.client
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from setup_work import metrics
from setup_work.config import MetricsConfig
from setup_work.errors import UsageError
from setup_work.metrics_proxy import UsageObserver, create_proxy
from setup_work.runner import Runner


@pytest.fixture
def enabled(cfg):
    cfg.corporate.metrics = MetricsConfig(enabled=True)
    return cfg


@contextlib.contextmanager
def running(server):
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_sse_fragmentation_and_cumulative_usage():
    observer = UsageObserver(True)
    stream = (
        b'data: {"choices":[{"delta":{"content":"PRIVATE TEXT"}}]}\r\n\r\n'
        b'data: {"model":"GigaChat","usage":{"prompt_tokens":100,"completion_tokens":1}}\r\n\r\n'
        b'data: {"usage":{"prompt_tokens":100,"completion_tokens":20,"total_tokens":120}}\n\n'
        b"data: [DONE]\n\n"
    )
    for byte in stream:
        observer.feed(bytes([byte]))
    observer.finish()
    assert observer.model == "GigaChat"
    assert metrics.normalize_usage(observer.usage)["total_tokens"] == 120
    assert b"PRIVATE" not in observer.buffer
    assert "PRIVATE" not in json.dumps(observer.usage)


def test_anthropic_stream_merges_usage_without_double_counting():
    observer = UsageObserver(True)
    observer.feed(
        b'data: {"type":"message_start","message":{"model":"claude",'
        b'"usage":{"input_tokens":10,"output_tokens":1,"cache_read_input_tokens":90}}}\n\n'
    )
    observer.feed(b'data: {"type":"message_delta","usage":{"output_tokens":20}}\n\n')
    observer.finish()
    counters = metrics.normalize_usage(observer.usage)
    assert counters["input_tokens"] == 100 and counters["total_tokens"] == 120


def test_responses_stream_and_trailing_event():
    observer = UsageObserver(True)
    observer.feed(
        b'data: {"type":"response.completed","response":{"model":"test",'
        b'"usage":{"input_tokens":10,"output_tokens":20,"total_tokens":30}}}'
    )
    observer.finish()
    assert metrics.normalize_usage(observer.usage)["total_tokens"] == 30


@pytest.mark.parametrize("streaming", [False, True])
def test_observer_memory_is_bounded(streaming):
    observer = UsageObserver(streaming)
    observer.feed(b"x" * (17 * 1024 * 1024))
    assert observer.overflow and not observer.buffer
    assert observer.usage == {}


def send(proxy, path="/v1/chat/completions", body=None, method="POST"):
    conn = http.client.HTTPConnection("127.0.0.1", proxy.server_port, timeout=5)
    conn.request(
        method,
        path,
        body=json.dumps(body or {"model": "requested", "stream": True}),
        headers={"Content-Type": "application/json", "Authorization": "Bearer SECRET KEY"},
    )
    response = conn.getresponse()
    result = response.status, response.read(), dict(response.getheaders())
    conn.close()
    return result


def test_proxy_forwards_sse_and_auth_and_records_one_request(enabled):
    seen = {}
    stream = (
        b'data: {"model":"actual","choices":[{"delta":{"content":"SECRET CONTENT"}}]}\n\n'
        b'data: {"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}\n\n'
        b"data: [DONE]\n\n"
    )

    class Upstream(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"] = self.path
            seen["headers"] = dict(self.headers)
            seen["request"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(stream)))
            self.end_headers()
            # Фрагментация не должна менять forwarded bytes или создавать повторный usage.
            for offset in range(0, len(stream), 7):
                self.wfile.write(stream[offset : offset + 7])
                self.wfile.flush()

        def log_message(self, *args):
            pass

    with running(ThreadingHTTPServer(("127.0.0.1", 0), Upstream)) as upstream:
        proxy = create_proxy(
            enabled,
            Runner(),
            agent="qwen-code",
            upstream=f"http://127.0.0.1:{upstream.server_port}/v1",
            port=0,
        )
        with running(proxy):
            status, body, _ = send(proxy)
            assert status == 200 and body == stream
        assert seen["path"] == "/v1/chat/completions"
        assert seen["headers"]["Authorization"] == "Bearer SECRET KEY"
        assert seen["request"]["stream_options"] == {"include_usage": True}
    summary = metrics.summary(enabled)
    assert summary["events"] == 1 and summary["total_tokens"] == 12
    assert summary["groups"][0]["model"] == "actual"
    db = (metrics.metrics_dir(enabled) / "usage.sqlite3").read_bytes()
    assert b"SECRET" not in db


@pytest.mark.parametrize(
    "status,usage",
    [
        (200, {"prompt_tokens": 10, "completion_tokens": 2}),
        (200, None),
        (429, None),
        (200, {"prompt_tokens": -1, "completion_tokens": 2}),
    ],
)
def test_proxy_json_errors_missing_and_invalid_usage(enabled, status, usage):
    data = json.dumps(
        {"model": "actual", "usage": usage, "choices": [{"message": {"content": "PRIVATE"}}]}
    ).encode()

    class Upstream(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    with running(ThreadingHTTPServer(("127.0.0.1", 0), Upstream)) as upstream:
        proxy = create_proxy(
            enabled,
            Runner(),
            agent="pi",
            upstream=f"http://127.0.0.1:{upstream.server_port}",
            port=0,
        )
        with running(proxy):
            assert send(proxy, body={"model": "test"})[:2] == (status, data)
    summary = metrics.summary(enabled)
    assert summary["events"] == 1
    valid = usage is not None and usage["prompt_tokens"] >= 0
    assert summary["total_tokens"] == (12 if valid else 0)
    assert summary["missing_usage"] == (0 if valid else 1)
    assert summary["failed_requests"] == (1 if status == 429 else 0)


def test_proxy_does_not_count_models_endpoint(enabled):
    class Upstream(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"data":[]}')

        def log_message(self, *args):
            pass

    with (
        running(ThreadingHTTPServer(("127.0.0.1", 0), Upstream)) as upstream,
        running(
            create_proxy(
                enabled,
                Runner(),
                agent="pi",
                port=0,
                upstream=f"http://127.0.0.1:{upstream.server_port}",
            )
        ) as proxy,
    ):
        assert send(proxy, path="/models", method="GET")[0] == 200
    assert metrics.summary(enabled)["events"] == 0


@pytest.mark.parametrize(
    "upstream",
    [
        "file:///etc/passwd",
        "https://user:pass@host/v1",
        "http://host/v1?key=SECRET",
        "http://host:bad/v1",
    ],
)
def test_upstream_validation(enabled, upstream):
    with pytest.raises(UsageError):
        create_proxy(enabled, Runner(), agent="pi", upstream=upstream, port=0)
