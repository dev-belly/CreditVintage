"""Loopback-only browser workbench for an existing local workspace."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from creditvintage.core import DataContractError
from creditvintage.review_bundle import compare_workspace_to_bundle, export_review_zip
from creditvintage.reviews import record_review, review_workspace

ASSETS = {
    "/": ("workbench.html", "text/html; charset=utf-8"),
    "/workbench.css": ("workbench.css", "text/css; charset=utf-8"),
    "/workbench.js": ("workbench.js", "text/javascript; charset=utf-8"),
    "/workbench-demo.js": ("workbench-demo.js", "text/javascript; charset=utf-8"),
}
MAX_BASELINE_BYTES = 16 * 1024 * 1024
MAX_BASELINE_UNPACKED_BYTES = 64 * 1024 * 1024


def create_workbench_server(workspace: Path, port: int) -> ThreadingHTTPServer:
    """Bind the bundled browser page and workspace operations to loopback."""

    class Handler(BaseHTTPRequestHandler):
        def reply(
            self, status: int, data: bytes, media: str, headers: dict[str, str] | None = None
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", media)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            try:
                self.end_headers()
                self.wfile.write(data)
            except ConnectionError:
                # The browser can abandon an outdated comparison while replay finishes.
                self.close_connection = True

        def send_json(self, status: int, value: dict[str, Any]) -> None:
            self.reply(status, json.dumps(value).encode("utf-8"), "application/json")

        def valid_host(self) -> bool:
            return self.headers.get("Host") == f"127.0.0.1:{http.server_port}"

        def do_GET(self) -> None:
            if not self.valid_host():
                self.send_json(403, {"error": "Open the exact loopback address."})
            elif self.path in ASSETS:
                asset, media = ASSETS[self.path]
                self.reply(200, Path(__file__).with_name(asset).read_bytes(), media)
            elif self.path == "/api/state":
                try:
                    self.send_json(200, review_workspace(workspace))
                except (DataContractError, OSError) as exc:
                    self.send_json(422, {"error": str(exc)})
            elif self.path == "/api/evidence":
                self.send_evidence()
            else:
                self.send_json(404, {"error": "Unknown route"})

        def send_evidence(self) -> None:
            if self.headers.get("Sec-Fetch-Site") not in (None, "none", "same-origin"):
                self.send_json(403, {"error": "Export evidence from the local workbench."})
                return
            try:
                data, receipt = export_review_zip(workspace)
            except DataContractError as exc:
                self.send_json(422, {"error": str(exc)})
                return
            digest = receipt["bundle_sha256"]
            filename = f"creditvintage-{digest[:12]}.zip"
            self.reply(
                200,
                data,
                "application/zip",
                {
                    "Content-Disposition": f'attachment; filename="{filename}"',
                    "X-CreditVintage-Checkpoint": digest,
                },
            )

        def do_POST(self) -> None:
            if not self.valid_host() or self.headers.get("Origin") != (
                f"http://127.0.0.1:{http.server_port}"
            ):
                self.send_json(403, {"error": "Use the local workbench for this request."})
                return
            if self.path == "/api/compare":
                self.compare_evidence()
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

        def compare_evidence(self) -> None:
            if self.headers.get_content_type() != "application/zip":
                self.send_json(415, {"error": "Choose an earlier review ZIP."})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_BASELINE_BYTES:
                    limit = MAX_BASELINE_BYTES // (1024 * 1024)
                    self.send_json(413, {"error": f"Choose a review ZIP up to {limit} MiB."})
                    return
                baseline = self.rfile.read(size)
                if len(baseline) != size:
                    raise DataContractError("The review ZIP was not received in full")
                result = compare_workspace_to_bundle(
                    workspace,
                    baseline,
                    baseline_sha256=self.headers.get("X-CreditVintage-Baseline-Checkpoint"),
                    max_unpacked_bytes=MAX_BASELINE_UNPACKED_BYTES,
                )
                self.send_json(200, result)
            except (OSError, ValueError) as exc:
                self.send_json(422, {"error": str(exc)})

    http = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    return http


def serve(workspace: Path, port: int) -> None:
    """Run the local workbench until interrupted."""
    with create_workbench_server(workspace, port) as http:
        print(f"http://127.0.0.1:{http.server_port}/", flush=True)
        try:
            http.serve_forever()
        except KeyboardInterrupt:
            pass
