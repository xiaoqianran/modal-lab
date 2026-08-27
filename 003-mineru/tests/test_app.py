"""003 app.py 的本地 CLI / planning 测试；不连接 Modal。"""
from __future__ import annotations

import hashlib
import sys
import tempfile
import types
import unittest
from pathlib import Path


class _Chain:
    @classmethod
    def from_name(cls, *args, **kwargs):
        return cls()

    @classmethod
    def debian_slim(cls, *args, **kwargs):
        return cls()

    @classmethod
    def from_registry(cls, *args, **kwargs):
        return cls()

    def __getattr__(self, name):
        return lambda *args, **kwargs: self


class _DummyApp:
    def __init__(self, *args, **kwargs):
        pass

    def function(self, *args, **kwargs):
        return lambda fn: fn

    def local_entrypoint(self, *args, **kwargs):
        return lambda fn: fn


sys.modules["modal"] = types.SimpleNamespace(App=_DummyApp, Image=_Chain, Volume=_Chain)
EXP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP_DIR))
import app  # noqa: E402


class TestCli(unittest.TestCase):
    def test_status(self):
        status = app.local_status()
        self.assertEqual(status["mineru_version"], "3.4.4")
        self.assertEqual(status["default_gpu"], "H100!")
        self.assertEqual(status["default_backend"], "hybrid-engine")

    def test_benchmark_pages_become_explicit_range(self):
        plan = app.parse_plan(
            app.parse_cli(
                [
                    "benchmark",
                    "--pages",
                    "100",
                    "--start-page",
                    "11",
                    "--gpu",
                    "RTX-PRO-6000",
                ]
            )
        )
        self.assertEqual(plan["start_page"], 11)
        self.assertEqual(plan["end_page"], 110)
        self.assertEqual(plan["gpu"], "RTX-PRO-6000")

    def test_parse_preserves_backend_effort_resume(self):
        plan = app.parse_plan(
            app.parse_cli(
                [
                    "parse",
                    "--backend",
                    "pipeline",
                    "--effort",
                    "high",
                    "--start-page",
                    "10",
                    "--end-page",
                    "20",
                    "--no-resume",
                ]
            )
        )
        self.assertEqual(plan["backend"], "pipeline")
        self.assertEqual(plan["effort"], "high")
        self.assertEqual((plan["start_page"], plan["end_page"]), (10, 20))
        self.assertFalse(plan["resume"])

    def test_invalid_page_ranges_fail_locally(self):
        with self.assertRaisesRegex(ValueError, "--pages"):
            app.parse_plan(app.parse_cli(["benchmark", "--pages", "0"]))
        with self.assertRaisesRegex(ValueError, "--start-page"):
            app.parse_plan(app.parse_cli(["parse", "--start-page", "0"]))
        with self.assertRaisesRegex(ValueError, "--end-page"):
            app.parse_plan(
                app.parse_cli(["parse", "--start-page", "10", "--end-page", "5"])
            )

    def test_upload_is_explicit_local_input_boundary(self):
        args = app.parse_cli(
            ["upload", "--pdf", "/tmp/book.pdf", "--remote-pdf", "/books/book.pdf"]
        )
        self.assertEqual(args.pdf, Path("/tmp/book.pdf"))
        self.assertEqual(args.remote_pdf, "/books/book.pdf")

    def test_sha256(self):
        with tempfile.NamedTemporaryFile("wb", delete=False) as handle:
            handle.write(b"mineru")
            path = Path(handle.name)
        try:
            self.assertEqual(app._sha256(path), hashlib.sha256(b"mineru").hexdigest())
        finally:
            path.unlink(missing_ok=True)

    def test_dry_run(self):
        self.assertTrue(app.parse_cli(["download", "--dry-run"]).dry_run)
        self.assertTrue(app.parse_cli(["upload", "--dry-run"]).dry_run)
        self.assertTrue(app.parse_cli(["parse", "--dry-run"]).dry_run)


if __name__ == "__main__":
    unittest.main()
