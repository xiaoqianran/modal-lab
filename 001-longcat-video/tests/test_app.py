# -*- coding: utf-8 -*-
"""001 app.py 的纯规划测试；不连接 Modal，不启动 GPU。"""

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

    def commit(self):
        return None


class _DummyApp:
    def __init__(self, *args, **kwargs):
        pass

    def function(self, *args, **kwargs):
        return lambda fn: fn

    def local_entrypoint(self, *args, **kwargs):
        return lambda fn: fn


sys.modules["modal"] = types.SimpleNamespace(
    App=_DummyApp,
    Image=_DummyImage,
    Volume=_DummyVolume,
)

EXP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP_DIR))

import app  # noqa: E402


class TestProfiles(unittest.TestCase):
    def test_two_gpu_profile_is_atomic(self):
        profile = app.profile_for("pro6000-2")
        self.assertEqual(profile.gpu, "RTX-PRO-6000:2")
        self.assertEqual(profile.nproc, 2)
        self.assertEqual(profile.cpu, 8.0)
        self.assertEqual(profile.memory_mb, 65536)

    def test_storyboard_uses_long_profile_by_default(self):
        self.assertEqual(app.default_profile_for("storyboard"), "pro6000-long")
        self.assertEqual(app.default_profile_for("t2v"), app.DEFAULT_PROFILE)

    def test_unknown_profile_fails(self):
        with self.assertRaisesRegex(ValueError, "未知 profile"):
            app.profile_for("future-gpu")


class TestCommandPlanning(unittest.TestCase):
    def test_infra_args_are_injected_once(self):
        profile = app.profile_for("pro6000-2")
        argv = app.build_script_argv(
            "t2v",
            profile=profile,
            enable_compile=True,
            upstream_args=["--", "--seed", "7"],
        )
        self.assertEqual(argv[0], f"--checkpoint_dir={app.CHECKPOINT_DIR}")
        self.assertIn("--context_parallel_size=2", argv)
        self.assertIn("--enable_compile", argv)
        self.assertEqual(argv[-2:], ["--seed", "7"])

    def test_no_compile(self):
        argv = app.build_script_argv(
            "t2v",
            profile=app.profile_for("pro6000-1"),
            enable_compile=False,
        )
        self.assertNotIn("--enable_compile", argv)

    def test_upstream_args_are_opaque(self):
        argv = app.build_script_argv(
            "storyboard",
            profile=app.profile_for("pro6000-long"),
            upstream_args=["--", "--future-option", "value"],
        )
        self.assertEqual(argv[-2:], ["--future-option", "value"])

    def test_reserved_upstream_args_are_rejected(self):
        for token in (
            "--checkpoint_dir=/tmp/model",
            "--context-parallel-size=2",
            "--enable_compile",
        ):
            with self.subTest(token=token):
                with self.assertRaisesRegex(ValueError, "modal-lab 管理"):
                    app.normalize_upstream_args(["--", token])

    def test_summary_has_one_resource_source(self):
        summary = app.run_summary(
            "t2v",
            profile_name="a100-80-2",
            checkpoint_dir=app.CHECKPOINT_DIR,
            enable_compile=False,
            output_subdir="demo",
            upstream_args=[],
        )
        self.assertEqual(summary["resources"]["gpu"], "A100-80GB:2")
        self.assertEqual(summary["resources"]["nproc"], 2)
        self.assertIn("--context_parallel_size=2", summary["script_argv"])


class TestCli(unittest.TestCase):
    def test_passthrough_after_separator(self):
        args = app.parse_cli(
            ["t2v", "--profile", "pro6000-2", "--", "--seed", "42"]
        )
        self.assertEqual(args.command, "t2v")
        self.assertEqual(args.profile, "pro6000-2")
        self.assertEqual(args.upstream_args, ["--", "--seed", "42"])

    def test_dry_run_flag(self):
        args = app.parse_cli(["storyboard", "--dry-run"])
        self.assertTrue(args.dry_run)


if __name__ == "__main__":
    unittest.main()
