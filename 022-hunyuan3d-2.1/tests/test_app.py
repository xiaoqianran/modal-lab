"""022 app.py 的本地 CLI / planning 测试；不连接 Modal。"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path


class _DummyImage:
    @classmethod
    def from_registry(cls, *args, **kwargs):
        return cls()

    def __getattr__(self, name):
        return lambda *args, **kwargs: self


class _DummyVolume:
    @classmethod
    def from_name(cls, *args, **kwargs):
        return cls()


class _DummyApp:
    def __init__(self, *args, **kwargs):
        pass

    def cls(self, *args, **kwargs):
        return lambda cls: cls

    def local_entrypoint(self, *args, **kwargs):
        return lambda fn: fn


def _decorator(*args, **kwargs):
    return lambda obj: obj


sys.modules["modal"] = types.SimpleNamespace(
    App=_DummyApp,
    Image=_DummyImage,
    Volume=_DummyVolume,
    enter=_decorator,
    method=_decorator,
)

EXP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP_DIR))

import app  # noqa: E402


class TestCli(unittest.TestCase):
    def test_status_is_local_metadata(self):
        status = app.local_status()
        self.assertEqual(status["gpu"], "L40S")
        self.assertEqual(status["cuda_arch"], "8.9")
        self.assertEqual(status["upstream_commit"], app.UPSTREAM_COMMIT)

    def test_smoke_defaults_match_previous_cli(self):
        args = app.parse_cli(["smoke"])
        plan = app.smoke_plan(args)
        self.assertEqual(plan["image_url"], app.SAMPLE_URL)
        self.assertEqual(plan["output_name"], "smoke_l40s")
        self.assertEqual(plan["mode"], "full")
        self.assertEqual(plan["seed"], 42)
        self.assertEqual(plan["max_num_view"], 6)
        self.assertEqual(plan["paint_resolution"], 512)

    def test_smoke_overrides(self):
        args = app.parse_cli(
            [
                "smoke",
                "--mode",
                "shape",
                "--seed",
                "7",
                "--paint-resolution",
                "256",
            ]
        )
        plan = app.smoke_plan(args)
        self.assertEqual(plan["mode"], "shape")
        self.assertEqual(plan["seed"], 7)
        self.assertEqual(plan["paint_resolution"], 256)

    def test_dry_run_is_explicit(self):
        args = app.parse_cli(["smoke", "--dry-run"])
        self.assertTrue(args.dry_run)
        self.assertFalse(args.i_know_this_costs_money)


if __name__ == "__main__":
    unittest.main()
