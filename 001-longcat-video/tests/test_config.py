# -*- coding: utf-8 -*-
"""本地配置合并 / 透传 / 校验单测（不连 Modal）。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))

from lib.config import (  # noqa: E402
    ConfigError,
    build_script_argv,
    merge_run_config,
    parse_script_passthrough,
    validate_local_prereqs,
)
from lib.demo_specs import get_demo  # noqa: E402


class TestPassthrough(unittest.TestCase):
    def test_parse_value_and_bool(self):
        d = parse_script_passthrough(
            ["--context_parallel_size", "2", "--enable_compile"]
        )
        self.assertEqual(d["context_parallel_size"], 2)
        self.assertTrue(d["enable_compile"])

    def test_parse_eq(self):
        d = parse_script_passthrough(["--checkpoint_dir=/weights/LongCat-Video"])
        self.assertEqual(d["checkpoint_dir"], "/weights/LongCat-Video")

    def test_unknown_arg(self):
        with self.assertRaises(ConfigError):
            parse_script_passthrough(["--prompt", "hi"])


class TestBuildArgv(unittest.TestCase):
    def test_build(self):
        argv = build_script_argv(
            {
                "checkpoint_dir": "/weights/LongCat-Video",
                "context_parallel_size": 1,
                "enable_compile": True,
            },
            demo="t2v",
            default_checkpoint="/weights/LongCat-Video",
        )
        self.assertIn("--checkpoint_dir=/weights/LongCat-Video", argv)
        self.assertIn("--context_parallel_size=1", argv)
        self.assertIn("--enable_compile", argv)

    def test_compile_off(self):
        argv = build_script_argv(
            {
                "checkpoint_dir": "/weights/LongCat-Video",
                "context_parallel_size": 1,
                "enable_compile": False,
            },
            demo="t2v",
            default_checkpoint="/weights/LongCat-Video",
        )
        self.assertNotIn("--enable_compile", argv)


class TestMerge(unittest.TestCase):
    def test_default_t2v(self):
        cfg = merge_run_config(command="demo", demo="t2v")
        self.assertEqual(cfg.demo, "t2v")
        self.assertEqual(cfg.resolved_nproc, 1)
        self.assertIn("RTX-PRO-6000", cfg.resolved_gpu)
        self.assertTrue(cfg.resolved_enable_compile)
        self.assertEqual(cfg.resolved_context_parallel_size, 1)

    def test_two_gpu(self):
        cfg = merge_run_config(command="demo", demo="t2v", two_gpu=True)
        self.assertEqual(cfg.resolved_nproc, 2)
        self.assertEqual(cfg.resolved_context_parallel_size, 2)
        self.assertIn(":2", cfg.resolved_gpu)

    def test_no_compile(self):
        cfg = merge_run_config(command="demo", demo="t2v", no_compile=True)
        self.assertFalse(cfg.resolved_enable_compile)
        self.assertNotIn("--enable_compile", cfg.script_argv)

    def test_cli_passthrough_wins(self):
        cfg = merge_run_config(
            command="demo",
            demo="t2v",
            cli_script_args={"enable_compile": False, "context_parallel_size": 1},
        )
        self.assertFalse(cfg.resolved_enable_compile)

    def test_mismatch_cp_nproc(self):
        with self.assertRaises(ConfigError):
            merge_run_config(
                command="demo",
                demo="t2v",
                two_gpu=True,
                cli_script_args={"context_parallel_size": 1},
            )

    def test_unknown_demo(self):
        with self.assertRaises(ConfigError):
            merge_run_config(command="demo", demo="not-a-demo")

    def test_file_override(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write('{"infra": {"profile": "a100-80-1", "enable_compile": false}}')
            path = Path(f.name)
        try:
            cfg = merge_run_config(
                command="demo", demo="t2v", config_path=path
            )
            self.assertEqual(cfg.infra.profile, "a100-80-1")
            self.assertIn("A100-80GB", cfg.resolved_gpu)
            self.assertFalse(cfg.resolved_enable_compile)
        finally:
            path.unlink(missing_ok=True)


class TestLocalPrereqs(unittest.TestCase):
    def test_upstream_and_assets(self):
        cfg = merge_run_config(command="demo", demo="i2v")
        validate_local_prereqs(cfg, upstream=EXP / "LongCat-Video")
        cfg2 = merge_run_config(command="demo", demo="continuation")
        validate_local_prereqs(cfg2, upstream=EXP / "LongCat-Video")

    def test_demo_specs(self):
        self.assertEqual(get_demo("t2v").script, "run_demo_text_to_video.py")


if __name__ == "__main__":
    unittest.main()
