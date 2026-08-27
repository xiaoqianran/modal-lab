"""033 app.py 的本地 CLI / planning 测试；不连接 Modal。"""

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


sys.modules["modal"] = types.SimpleNamespace(App=_DummyApp, Image=_Chain, Volume=_Chain)
EXP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP_DIR))
import app  # noqa: E402


class TestCli(unittest.TestCase):
    def test_status(self):
        status = app.local_status()
        self.assertEqual(status["default_gpu"], "L4")
        self.assertEqual(status["default_model"], "F5TTS_v1_Base")
        self.assertEqual(status["refs"]["en"], app.REF_EN)

    def test_smoke_defaults(self):
        plan = app.smoke_plan(app.parse_cli(["smoke"]))
        self.assertEqual(plan["run_name"], "smoke_en")
        self.assertEqual(plan["lang"], "en")
        self.assertEqual(plan["nfe_step"], 32)

    def test_zh_smoke(self):
        plan = app.smoke_plan(app.parse_cli(["smoke", "--kind", "zh", "--nfe-step", "24"]))
        self.assertEqual(plan["run_name"], "smoke_zh")
        self.assertEqual(plan["lang"], "zh")
        self.assertEqual(plan["text"], app.SMOKE_ZH)
        self.assertEqual(plan["nfe_step"], 24)

    def test_t2s_exposes_real_generate_options(self):
        plan = app.t2s_plan(
            app.parse_cli(
                [
                    "t2s",
                    "--text",
                    "  hello  ",
                    "--lang",
                    "en",
                    "--ref-audio",
                    "/prompts/custom.wav",
                    "--ref-text",
                    "reference",
                    "--nfe-step",
                    "20",
                    "--seed",
                    "7",
                ]
            )
        )
        self.assertEqual(plan["text"], "hello")
        self.assertEqual(plan["ref_audio"], "/prompts/custom.wav")
        self.assertEqual(plan["ref_text"], "reference")
        self.assertEqual(plan["nfe_step"], 20)
        self.assertEqual(plan["seed"], 7)

    def test_dry_run(self):
        self.assertTrue(app.parse_cli(["smoke", "--dry-run"]).dry_run)
        self.assertTrue(app.parse_cli(["t2s", "--text", "hi", "--dry-run"]).dry_run)


if __name__ == "__main__":
    unittest.main()
