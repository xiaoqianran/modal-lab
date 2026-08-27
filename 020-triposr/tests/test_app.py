"""020 app.py 的本地 CLI / planning 测试；不连接 Modal。"""

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
        self.assertEqual(status["model"], "stabilityai/TripoSR")
        self.assertEqual(status["default_gpu"], "L40S")
        self.assertEqual(status["gpus"], ["L40S", "RTX-PRO-6000"])

    def test_smoke_l40s_default_name(self):
        plan = app.smoke_plan(app.parse_cli(["smoke"]))
        self.assertEqual(plan["gpu"], "L40S")
        self.assertEqual(plan["output_name"], "smoke_l40s")

    def test_smoke_pro6000_default_name(self):
        plan = app.smoke_plan(
            app.parse_cli(["smoke", "--gpu", "RTX-PRO-6000"])
        )
        self.assertEqual(plan["gpu"], "RTX-PRO-6000")
        self.assertEqual(plan["output_name"], "smoke_pro6000")

    def test_output_name_override(self):
        plan = app.smoke_plan(app.parse_cli(["smoke", "--output-name", "chair"]))
        self.assertEqual(plan["output_name"], "chair")

    def test_dry_run_does_not_require_cost_ack(self):
        args = app.parse_cli(["smoke", "--dry-run"])
        self.assertTrue(args.dry_run)
        self.assertFalse(args.i_know_this_costs_money)


if __name__ == "__main__":
    unittest.main()
