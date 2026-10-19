"""Минимальный OpenAI-совместимый сервер для тестов; только стандартная библиотека."""

import json
import os
import signal
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok"})
        elif self.path == "/v1/models":
            self._send(200, {"object": "list", "data": [{"id": "fake-model"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/v1/chat/completions":
            self._send(
                200,
                {
                    "choices": [{"message": {"role": "assistant", "content": "pong"}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 1},
                    "model": req.get("model"),
                },
            )
        else:
            self._send(404, {"error": "not found"})

    def log_message(self, *args):
        sys.stdout.write("request " + (args[0] % args[1:]) + "\n")
        sys.stdout.flush()


def main():
    port = int(sys.argv[1])
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    signal.signal(signal.SIGTERM, lambda *_: os._exit(0))
    print(f"fake server listening on {port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
