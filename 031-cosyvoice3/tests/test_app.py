"""031 app.py 的本地 CLI / planning 测试；不连接 Modal。"""

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
        self.assertEqual(status["hf_repo"], app.HF_REPO)
        self.assertEqual(status["outputs_volume"], app.VOLUME_OUTPUTS)

    def test_zh_smoke_defaults(self):
        plan = app.smoke_plan(app.parse_cli(["smoke"]))
        self.assertEqual(plan["run_name"], "smoke_zh")
        self.assertEqual(plan["mode"], "zero_shot")
        self.assertEqual(plan["text"], app.SMOKE_ZH)

    def test_dialect_smoke_owns_instruction(self):
        plan = app.smoke_plan(app.parse_cli(["smoke", "--kind", "dialect"]))
        self.assertEqual(plan["run_name"], "smoke_dialect")
        self.assertEqual(plan["mode"], "instruct")
        self.assertIn("四川话", plan["instruct"])

    def test_tongue_and_english(self):
        tongue = app.smoke_plan(app.parse_cli(["smoke", "--kind", "tongue"]))
        english = app.smoke_plan(app.parse_cli(["smoke", "--kind", "en"]))
        self.assertEqual(tongue["text"], app.SMOKE_TONGUE)
        self.assertEqual(english["text"], app.SMOKE_EN)
        self.assertEqual(english["mode"], "en")

    def test_t2s_plan_preserves_domain_args(self):
        plan = app.t2s_plan(
            app.parse_cli(
                [
                    "t2s",
                    "--text",
                    "  你好  ",
                    "--mode",
                    "instruct",
                    "--instruct",
                    "开心地说",
                    "--prompt-text",
                    "参考文本",
                ]
            )
        )
        self.assertEqual(plan["text"], "你好")
        self.assertEqual(plan["mode"], "instruct")
        self.assertEqual(plan["instruct"], "开心地说")
        self.assertEqual(plan["prompt_text"], "参考文本")

    def test_dry_run_is_explicit(self):
        self.assertTrue(app.parse_cli(["smoke", "--dry-run"]).dry_run)
        self.assertTrue(app.parse_cli(["t2s", "--text", "hi", "--dry-run"]).dry_run)


if __name__ == "__main__":
    unittest.main()
