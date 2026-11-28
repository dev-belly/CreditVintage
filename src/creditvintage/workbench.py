"""Loopback-only browser workbench and review evidence reader."""

from __future__ import annotations

import json
import re
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl

from creditvintage.core import DataContractError
from creditvintage.review_bundle import (
    compare_review_bundles,
    compare_workspace_to_bundle,
    export_review_zip,
    replay_review_zip,
)
from creditvintage.reviews import record_review, review_workspace

ASSETS = {
    "/": ("workbench.html", "text/html; charset=utf-8"),
    "/replay": ("workbench.html", "text/html; charset=utf-8"),
    "/workbench.css": ("workbench.css", "text/css; charset=utf-8"),
    "/workbench.js": ("workbench.js", "text/javascript; charset=utf-8"),
    "/workbench-demo.js": ("workbench-demo.js", "text/javascript; charset=utf-8"),
}
MAX_REVIEW_ZIP_BYTES = 16 * 1024 * 1024
MAX_REVIEW_UNPACKED_BYTES = 64 * 1024 * 1024
MAX_MULTIPART_OVERHEAD = 8 * 1024


def _uploaded_zip_pair(body: bytes, boundary: str) -> tuple[bytes, bytes]:
    header = f'Content-Type: multipart/form-data; boundary="{boundary}"\r\n\r\n'
    message = BytesParser(policy=policy.default).parsebytes(header.encode("ascii") + body)
    error = "Choose one earlier and one later review ZIP."
    if not message.is_multipart() or message.defects:
        raise DataContractError(error)
    files: dict[str, bytes] = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if (
            part.get_content_disposition() != "form-data"
            or part.is_multipart()
            or part.defects
            or part.get("Content-Transfer-Encoding") is not None
            or not isinstance(name, str)
            or name not in ("baseline", "current")
            or name in files
        ):
            raise DataContractError(error)
        payload = part.get_payload(decode=True)
        if not isinstance(payload, bytes) or not payload:
            raise DataContractError(error)
        files[name] = payload
    if len(files) != 2:
        raise DataContractError(error)
    return files["baseline"], files["current"]


def create_workbench_server(workspace: Path | None, port: int) -> ThreadingHTTPServer:
    """Bind the browser reader or workspace to loopback."""

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
                data = Path(__file__).with_name(asset).read_bytes()
                if asset == "workbench.html" and (workspace is None or self.path == "/replay"):
                    data = data.replace(b'data-mode="local"', b'data-mode="reader"', 1)
                self.reply(200, data, media)
            elif workspace is None:
                self.send_json(404, {"error": "This route requires an open review workspace."})
            elif self.path == "/api/state":
                try:
                    self.send_json(200, review_workspace(workspace))
                except (DataContractError, OSError) as exc:
                    self.send_json(422, {"error": str(exc)})
            elif self.path == "/api/evidence" or self.path.startswith("/api/evidence?"):
                self.send_evidence(workspace)
            else:
                self.send_json(404, {"error": "Unknown route"})

        def send_evidence(self, database: Path) -> None:
            if self.headers.get("Sec-Fetch-Site") not in (None, "none", "same-origin"):
                self.send_json(403, {"error": "Export evidence from the local workbench."})
                return
            through_event, expected_sha256 = None, None
            query = self.path.partition("?")[2]
            if query:
                try:
                    parameters = dict(parse_qsl(query, strict_parsing=True, max_num_fields=2))
                    if parameters.keys() != {"through_event", "expected_sha256"}:
                        raise ValueError
                    through_event = int(parameters["through_event"])
                    expected_sha256 = parameters["expected_sha256"]
                except ValueError:
                    self.send_json(400, {"error": "Supply one review event number and checkpoint."})
                    return
            try:
                data, receipt = export_review_zip(
                    database, through_event=through_event, expected_sha256=expected_sha256
                )
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
            if self.path == "/api/replay":
                self.inspect_zip(None)
                return
            if self.path == "/api/compare-deliveries":
                self.compare_deliveries()
                return
            if workspace is None:
                self.send_json(404, {"error": "This route requires an open review workspace."})
                return
            if self.path == "/api/compare":
                self.inspect_zip(workspace)
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

        def inspect_zip(self, compare_with: Path | None) -> None:
            if self.headers.get_content_type() != "application/zip":
                self.send_json(415, {"error": "Choose a review ZIP."})
                return
            try:
                archive = self.read_upload(MAX_REVIEW_ZIP_BYTES)
                if archive is None:
                    return
                if compare_with is None:
                    result = replay_review_zip(
                        archive,
                        expected_sha256=self.headers.get("X-CreditVintage-Checkpoint"),
                        max_unpacked_bytes=MAX_REVIEW_UNPACKED_BYTES,
                    )
                else:
                    result = compare_workspace_to_bundle(
                        compare_with,
                        archive,
                        baseline_sha256=self.headers.get("X-CreditVintage-Baseline-Checkpoint"),
                        max_unpacked_bytes=MAX_REVIEW_UNPACKED_BYTES,
                        include_actions=True,
                    )
                self.send_json(200, result)
            except (OSError, ValueError) as exc:
                self.send_json(422, {"error": str(exc)})

        def read_upload(self, maximum: int) -> bytes | None:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= maximum:
                self.reject_upload_size()
                return None
            body = self.rfile.read(size)
            if len(body) != size:
                raise DataContractError("The review upload was not received in full")
            return body

        def reject_upload_size(self) -> None:
            limit = MAX_REVIEW_ZIP_BYTES // (1024 * 1024)
            self.send_json(413, {"error": f"Choose review ZIPs of up to {limit} MiB each."})

        def compare_deliveries(self) -> None:
            boundary = self.headers.get_boundary()
            if (
                self.headers.get_content_type() != "multipart/form-data"
                or not boundary
                or not re.fullmatch(r"[A-Za-z0-9'()+_,./:=?-]{1,70}", boundary)
            ):
                self.send_json(415, {"error": "Choose two review ZIPs."})
                return
            try:
                body = self.read_upload(2 * MAX_REVIEW_ZIP_BYTES + MAX_MULTIPART_OVERHEAD)
                if body is None:
                    return
                baseline, current = _uploaded_zip_pair(body, boundary)
                if max(len(baseline), len(current)) > MAX_REVIEW_ZIP_BYTES:
                    self.reject_upload_size()
                    return
                result = compare_review_bundles(
                    BytesIO(baseline),
                    BytesIO(current),
                    baseline_sha256=self.headers.get("X-CreditVintage-Baseline-Checkpoint"),
                    current_sha256=self.headers.get("X-CreditVintage-Current-Checkpoint"),
                    max_unpacked_bytes=MAX_REVIEW_UNPACKED_BYTES,
                    include_actions=True,
                )
                self.send_json(200, result)
            except (OSError, ValueError) as exc:
                self.send_json(422, {"error": str(exc)})

    http = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    return http


def serve(workspace: Path | None, port: int) -> None:
    """Run the local workbench until interrupted."""
    with create_workbench_server(workspace, port) as http:
        print(f"http://127.0.0.1:{http.server_port}/", flush=True)
        try:
            http.serve_forever()
        except KeyboardInterrupt:
            pass
