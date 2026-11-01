import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from rag.config import ModelConfig
from rag.models import Message
from rag.providers.openai_compatible import OpenAICompatibleClient


def test_local_http_service(monkeypatch):
    # Local test traffic must bypass any proxy configured in the host environment.
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.headers.get("Authorization") != "Bearer test-key":
                self.send_error(401)
                return
            if self.path == "/v1/chat/completions":
                body = {"model": request["model"], "choices": [{"message": {"content": "ok"}}]}
            elif self.path == "/v1/embeddings":
                body = {"data": [{"index": i, "embedding": [float(i), 1.0]} for i, _ in enumerate(request["input"])]}
            else:
                self.send_error(404)
                return
            encoded = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    probe = socket.socket()
    try:
        result = probe.connect_ex(server.server_address)
    finally:
        probe.close()
    if result in (13, 10013):
        server.server_close()
        pytest.skip("Sandbox forbids local TCP connections")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = ModelConfig("openai_compatible", f"http://127.0.0.1:{server.server_port}/v1", "test-key", "local-model")
        adapter = OpenAICompatibleClient(config)
        assert adapter.generate([Message("user", "hello")]).text == "ok"
        assert adapter.embed(["a", "b"]) == [[0.0, 1.0], [1.0, 1.0]]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
