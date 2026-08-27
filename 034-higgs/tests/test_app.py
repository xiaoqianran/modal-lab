"""034 app.py 的本地 CLI / planning 测试；不连接 Modal。"""

from __future__ import annotations

import sys
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

    def __getattr__(self, name):
        return lambda *args, **kwargs: self


class _DummyApp:
    def __init__(self, *args, **kwargs):
        pass

    def function(self, *args, **kwargs):
        return lambda fn: fn

    def local_entrypoint(self, *args, **kwargs):
        return lambda fn: fn


sys.modules["modal"] = types.SimpleNamespace(
    App=_DummyApp,
    Image=_Chain,
    Volume=_Chain,
)

EXP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP_DIR))

import app  # noqa: E402


class TestCli(unittest.TestCase):
    def test_status_is_local_metadata(self):
        status = app.local_status()
        self.assertEqual(status["default_gpu"], "L40S")
        self.assertEqual(status["model_revision"], app.MODEL_REVISION)
        self.assertEqual(status["outputs_volume"], app.VOLUME_OUTPUTS)

    def test_smoke_en_defaults(self):
        plan = app.smoke_plan(app.parse_cli(["smoke"]))
        self.assertEqual(plan["run_name"], "smoke_en")
        self.assertIn("quiet room", plan["scene"])
        self.assertEqual(plan["text"], app.SMOKE_EN)

    def test_smoke_expressive_defaults(self):
        plan = app.smoke_plan(app.parse_cli(["smoke", "--kind", "expressive"]))
        self.assertEqual(plan["run_name"], "smoke_expressive")
        self.assertIn("excited", plan["scene"])
        self.assertEqual(plan["text"], app.SMOKE_EXPRESSIVE)

    def test_t2s_plan_strips_text(self):
        plan = app.t2s_plan(
            app.parse_cli(["t2s", "--text", "  hello  ", "--temperature", "0.5"])
        )
        self.assertEqual(plan["text"], "hello")
        self.assertEqual(plan["temperature"], 0.5)

    def test_dry_run_is_explicit(self):
        self.assertTrue(app.parse_cli(["smoke", "--dry-run"]).dry_run)
        self.assertTrue(app.parse_cli(["t2s", "--text", "hi", "--dry-run"]).dry_run)


if __name__ == "__main__":
    unittest.main()
