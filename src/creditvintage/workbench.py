"""Loopback-only browser workbench for an existing local workspace."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from creditvintage.core import DataContractError
from creditvintage.reviews import record_review, review_workspace


def serve(workspace: Path, port: int) -> None:
    """Expose the existing workspace operations to the bundled browser page."""

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status: int, data: bytes, media: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", media)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()
            self.wfile.write(data)

        def send_json(self, status: int, value: dict[str, Any]) -> None:
            self.reply(status, json.dumps(value).encode("utf-8"), "application/json")

        def valid_host(self) -> bool:
            return self.headers.get("Host") == f"127.0.0.1:{http.server_port}"

        def do_GET(self) -> None:
            if not self.valid_host():
                self.send_json(403, {"error": "Open the exact loopback address."})
            elif self.path == "/":
                page = Path(__file__).with_name("workbench.html").read_bytes()
                self.reply(200, page, "text/html; charset=utf-8")
            elif self.path == "/api/state":
                try:
                    self.send_json(200, review_workspace(workspace))
                except (DataContractError, OSError) as exc:
                    self.send_json(422, {"error": str(exc)})
            else:
                self.send_json(404, {"error": "Unknown route"})

        def do_POST(self) -> None:
            if not self.valid_host() or self.headers.get("Origin") != (
                f"http://127.0.0.1:{http.server_port}"
            ):
                self.send_json(403, {"error": "Use the local workbench to submit changes."})
                return
            if self.path != "/api/change":
                self.send_json(404, {"error": "Unknown route"})
                return
            if self.headers.get_content_type() != "application/json":
                self.send_json(415, {"error": "Expected JSON"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 16_384:
                    self.send_json(413, {"error": "Request exceeds the workbench limit."})
                    return
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError("Expected a JSON object")
                result = record_review(workspace, **body)
                self.send_json(200, result)
            except (DataContractError, OSError, ValueError, TypeError) as exc:
                status = 409 if "Stale review version" in str(exc) else 422
                self.send_json(status, {"error": str(exc)})

    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as http:
        print(f"http://127.0.0.1:{http.server_port}/", flush=True)
        try:
            http.serve_forever()
        except KeyboardInterrupt:
            pass
