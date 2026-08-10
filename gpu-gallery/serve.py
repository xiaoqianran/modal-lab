#!/usr/bin/env python3
"""Static server with Accept-Ranges for large GLBs (model-viewer friendly)."""
from __future__ import annotations

import argparse
import os
import re
import socket
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parent


class _LimitedReader:
    def __init__(self, f, remaining: int):
        self._f = f
        self._remaining = remaining

    def read(self, n: int = -1) -> bytes:
        if self._remaining <= 0:
            return b""
        if n is None or n < 0:
            n = self._remaining
        n = min(n, self._remaining)
        data = self._f.read(n)
        self._remaining -= len(data)
        return data

    def close(self) -> None:
        self._f.close()


class RangeHandler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def end_headers(self) -> None:
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "public, max-age=3600")
        super().end_headers()

    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        if not os.path.isfile(path):
            self.send_error(404, "File not found")
            return None

        ctype = self.guess_type(path)
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None

        fs = os.fstat(f.fileno())
        size = fs.st_size
        range_header = self.headers.get("Range")
        if range_header:
            m = re.match(r"bytes=(\d*)-(\d*)", range_header.strip())
            if not m:
                f.close()
                self.send_error(400, "Invalid Range")
                return None
            start_s, end_s = m.group(1), m.group(2)
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else size - 1
            if start >= size or end >= size or start > end:
                f.close()
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return None
            length = end - start + 1
            f.seek(start)
            self.send_response(206)
            self.send_header("Content-type", ctype)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(length))
            self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
            self.end_headers()
            return _LimitedReader(f, length)

        self.send_response(200)
        self.send_header("Content-type", ctype)
        self.send_header("Content-Length", str(size))
        self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
        self.end_headers()
        return f

    def log_message(self, fmt: str, *args) -> None:
        if args and str(args[0]).startswith('"GET /assets/'):
            return
        super().log_message(fmt, *args)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()
    os.chdir(ROOT)
    handler = partial(RangeHandler, directory=str(ROOT))
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    httpd.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    print(f"gallery on http://{args.host}:{args.port}/  root={ROOT}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
