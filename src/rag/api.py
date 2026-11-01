"""Minimal authenticated loopback HTTP API for local development."""
import hmac
import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer

from .errors import RagError
from .pipeline import answer

logger = logging.getLogger(__name__)
MAX_REQUEST_BYTES = 64 * 1024


def make_handler(store, embedder, generator, embedding_id, token, *, backend="exact", ef_search=64, reranker=None, candidate_k=20, retrieval_mode="dense"):
    if not token:
        raise ValueError("RAG_API_TOKEN must be set")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # Avoid logging questions and credentials.
            pass

        def respond(self, status, body):
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def authorized(self):
            return hmac.compare_digest(self.headers.get("Authorization", "").encode("utf-8"), ("Bearer " + token).encode("utf-8"))

        def do_GET(self):
            if not self.authorized():
                return self.respond(401, {"error": "unauthorized"})
            if self.path != "/health":
                return self.respond(404, {"error": "not found"})
            return self.respond(200, {"status": "ok"})

        def do_POST(self):
            if not self.authorized():
                return self.respond(401, {"error": "unauthorized"})
            if self.path != "/v1/ask":
                return self.respond(404, {"error": "not found"})
            if self.headers.get_content_type() != "application/json":
                return self.respond(415, {"error": "application/json required"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_REQUEST_BYTES:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("expected JSON object")
                allowed = {"index_id", "question", "session_id", "strategy", "history_turns", "top_k", "min_score", "context_chars", "context_mode", "citation_policy"}
                if set(payload) - allowed or not isinstance(payload.get("index_id"), str) or not isinstance(payload.get("question"), str):
                    raise ValueError("invalid request fields")
                if payload.get("session_id") is not None and not isinstance(payload["session_id"], str):
                    raise ValueError("invalid session ID")
                for name, lower, upper in (("history_turns", 0, 20), ("top_k", 1, 100), ("context_chars", 1, 40000)):
                    if name in payload and (type(payload[name]) is not int or not lower <= payload[name] <= upper):
                        raise ValueError(f"invalid {name}")
                if "min_score" in payload and (type(payload["min_score"]) not in (int, float) or not -1 <= payload["min_score"] <= 1):
                    raise ValueError("invalid min_score")
                result = answer(store, embedder, generator, embedding_id=embedding_id, backend=backend, ef_search=ef_search, reranker=reranker, candidate_k=max(candidate_k, payload.get("top_k", 5)) if reranker else None, retrieval_mode=retrieval_mode, **payload)
                return self.respond(200, result)
            except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError):
                return self.respond(400, {"error": "invalid request"})
            except RagError:
                return self.respond(502, {"error": "index or provider error"})
            except Exception:
                logger.error("Unhandled API error")
                return self.respond(500, {"error": "internal error"})

    return Handler


def serve(store, embedder, generator, embedding_id, token, port=8000, *, backend="exact", ef_search=64, reranker=None, candidate_k=20, retrieval_mode="dense"):
    server = HTTPServer(("127.0.0.1", port), make_handler(store, embedder, generator, embedding_id, token, backend=backend, ef_search=ef_search, reranker=reranker, candidate_k=candidate_k, retrieval_mode=retrieval_mode))
    try:
        server.serve_forever()
    finally:
        server.server_close()
