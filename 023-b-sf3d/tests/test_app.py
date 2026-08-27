"""023-b app.py 的本地 CLI / planning 测试；不连接 Modal。"""

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
    def from_registry(cls, *args, **kwargs):
        return cls()

    def __getattr__(self, name):
        return lambda *args, **kwargs: self


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
    Image=_Chain,
    Volume=_Chain,
    Secret=_Chain,
    enter=_decorator,
    method=_decorator,
)

EXP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP_DIR))

import app  # noqa: E402


class TestCli(unittest.TestCase):
    def test_status(self):
        status = app.local_status()
        self.assertEqual(status["default_gpu"], "L40S")
        self.assertEqual(status["official_model"], "stabilityai/stable-fast-3d")

    def test_default_smoke(self):
        plan = app.smoke_plan(app.parse_cli(["smoke"]))
        self.assertEqual(plan["output_name"], "smoke_l40s")
        self.assertEqual(plan["texture_resolution"], 1024)

    def test_pro6000(self):
        plan = app.smoke_plan(
            app.parse_cli(
                ["smoke", "--gpu", "RTX-PRO-6000", "--texture-resolution", "512"]
            )
        )
        self.assertEqual(plan["gpu"], "RTX-PRO-6000")
        self.assertEqual(plan["output_name"], "smoke_pro6000")
        self.assertEqual(plan["texture_resolution"], 512)

    def test_official_model_override(self):
        plan = app.smoke_plan(
            app.parse_cli(["smoke", "--hf-model", app.HF_MODEL_OFFICIAL])
        )
        self.assertEqual(plan["hf_model"], app.HF_MODEL_OFFICIAL)


if __name__ == "__main__":
    unittest.main()
