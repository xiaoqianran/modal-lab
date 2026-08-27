"""005-v2 app.py 的本地 CLI / planning 测试；不连接 Modal。"""

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
    enter=_decorator,
    method=_decorator,
)
EXP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP_DIR))
import app  # noqa: E402


class TestCli(unittest.TestCase):
    def test_status(self):
        status = app.local_status()
        self.assertEqual(status["gpu"], "L40S")
        self.assertEqual(status["cuda_arch"], "8.9")
        self.assertEqual(status["model"], "TencentARC/Pixal3D")

    def test_build_plan(self):
        plan = app.paid_plan(app.parse_cli(["build-sm89", "--dry-run"]))
        self.assertEqual(plan, {"action": "build-sm89", "gpu": "L40S"})

    def test_smoke_honors_output_name(self):
        plan = app.paid_plan(
            app.parse_cli(
                [
                    "smoke",
                    "--dry-run",
                    "--output-name",
                    "custom_l40s",
                    "--resolution",
                    "512",
                ]
            )
        )
        self.assertEqual(plan["output_name"], "custom_l40s")
        self.assertEqual(plan["resolution"], 512)
        self.assertEqual(plan["image_url"], app.SAMPLE_IMAGE_URL)
        self.assertTrue(plan["low_vram"])

    def test_i2v_is_distinct_from_smoke(self):
        plan = app.paid_plan(
            app.parse_cli(
                [
                    "i2v",
                    "--dry-run",
                    "--image-url",
                    "https://example.com/chair.png",
                    "--no-low-vram",
                    "--fov",
                    "35",
                ]
            )
        )
        self.assertEqual(plan["image_url"], "https://example.com/chair.png")
        self.assertEqual(plan["output_name"], "demo_l40s")
        self.assertFalse(plan["low_vram"])
        self.assertEqual(plan["fov"], 35.0)

    def test_cost_ack_required_only_for_real_paid_execution(self):
        args = app.parse_cli(["verify"])
        with self.assertRaises(SystemExit):
            app.require_cost_ack(args)
        args = app.parse_cli(["verify", "--i-know-this-costs-money"])
        app.require_cost_ack(args)


if __name__ == "__main__":
    unittest.main()
