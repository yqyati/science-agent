"""Minimal outbound HTTP worker for one Agent Sandbox."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from threading import Lock
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen


PAGE_LOCK = Lock()


def save_page(payload, text):
    from tool_url_guard import evaluate_guard_value, guard_context_from_snapshot
    context = guard_context_from_snapshot(payload["guard_snapshot"])
    if evaluate_guard_value(text, context).blocked:
        return {"blocked": True}
    if not text.strip() or text.startswith(("[read_page] Failed", "[document_parser]")):
        raise ValueError("Reader returned no usable page content")
    page_id = hashlib.sha256(payload["source_url"].encode()).hexdigest()[:20]
    folder = Path("pages") / page_id
    folder.mkdir(parents=True, exist_ok=True)
    metadata = {
        "page_id": page_id, "path": str(folder / "content.md"),
        "source_url": payload["source_url"], "reader_url": payload["url"],
        "format": "markdown", "characters": len(text), "lines": len(text.splitlines()),
        "fetched_at": time.time(), "preview": text[:1000],
        "guard_fingerprint": payload["guard_fingerprint"],
    }
    (folder / "content.md.tmp").write_text(text, encoding="utf-8")
    (folder / "content.md.tmp").replace(folder / "content.md")
    (folder / "source.json.tmp").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
    (folder / "source.json.tmp").replace(folder / "source.json")
    return metadata


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        return

    def do_GET(self):  # noqa: N802
        if self.path != "/health":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def do_POST(self):  # noqa: N802
        if self.path != "/request":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            method = str(payload.get("method") or "GET").upper()
            url = str(payload.get("url") or "")
            headers = {str(k): str(v) for k, v in (payload.get("headers") or {}).items()}
            body = None
            if payload.get("json") is not None:
                body = json.dumps(payload["json"], ensure_ascii=False).encode("utf-8")
                headers.setdefault("Content-Type", "application/json")
            if payload.get("save_page"):
                # Serialize same-run page writes, and reuse only a complete,
                # nonempty page cached under the same evaluation guard policy.
                with PAGE_LOCK:
                    page_id = hashlib.sha256(payload["source_url"].encode()).hexdigest()[:20]
                    folder = Path("pages") / page_id
                    cached = None
                    if (folder / "source.json").is_file() and (folder / "content.md").is_file() and (folder / "content.md").stat().st_size:
                        cached = json.loads((folder / "source.json").read_text(encoding="utf-8"))
                    if cached and cached["guard_fingerprint"] == payload["guard_fingerprint"]:
                        cached.pop("absolute_path", None)
                        result = {"status_code": 200, "page": dict(cached, cached=True)}
                    else:
                        request = Request(url, headers=headers, method="GET")
                        try:
                            with urlopen(request, timeout=float(payload.get("timeout", 180))) as response:
                                text = response.read().decode("utf-8", errors="replace")
                                result = {"status_code": response.status}
                                if response.status == 200:
                                    result["page"] = save_page(payload, text)
                        except HTTPError as exc:
                            result = {"status_code": exc.code, "text": exc.read().decode("utf-8", errors="replace")[:1000]}
                        except Exception as exc:
                            result = {"status_code": 599, "text": f"{type(exc).__name__}: {exc}"}
                body = json.dumps(result, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            started = time.time()
            request = Request(url, data=body, headers=headers, method=method)
            try:
                with urlopen(request, timeout=float(payload.get("timeout", 180))) as response:
                    raw = response.read()
                    result = {
                        "status_code": response.status,
                        "text": raw.decode("utf-8", errors="replace"),
                        "headers": dict(response.headers),
                        "duration_s": time.time() - started,
                    }
            except HTTPError as exc:
                result = {
                    "status_code": exc.code,
                    "text": exc.read().decode("utf-8", errors="replace"),
                    "headers": dict(exc.headers),
                    "duration_s": time.time() - started,
                }
            except Exception as exc:
                result = {"status_code": 599, "text": f"{type(exc).__name__}: {exc}", "headers": {}, "duration_s": time.time() - started}
            body = json.dumps(result, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            self.send_error(500, str(exc))


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 18081), Handler).serve_forever()
