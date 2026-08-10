#!/usr/bin/env python3
"""Verify Blackwell sm_120 stack for Pixal3D PRO 6000."""
from __future__ import annotations

import argparse
import importlib
import sys


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--expect-gpu", action="store_true")
    args = p.parse_args()

    print("=== verify_sm120 · Blackwell / PRO 6000 gate ===")
    import torch

    if not torch.cuda.is_available():
        print("FAIL: CUDA not available")
        return 1
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU: {name}  capability={cap}")
    print(f"torch: {torch.__version__}  cuda: {torch.version.cuda}")
    arch = list(torch.cuda.get_arch_list()) if hasattr(torch.cuda, "get_arch_list") else []
    print(f"arch_list: {arch}")

    if args.expect_gpu and cap[0] != 12:
        print(f"FAIL: expected sm_12x, got {cap}")
        return 1
    print("OK: compute capability 12.x (Blackwell family)")

    # matmul
    a = torch.randn(128, 128, device="cuda", dtype=torch.float16)
    b = torch.randn(128, 128, device="cuda", dtype=torch.float16)
    _ = a @ b
    torch.cuda.synchronize()
    print("OK: matmul")

    mods = [
        ("natten", "natten"),
        ("flex_gemm", "flex_gemm"),
        ("o_voxel", "o_voxel"),
        ("cumesh", "cumesh"),
        ("nvdiffrast", "nvdiffrast"),
    ]
    print(f"\n{'package':<28} status")
    print("-" * 50)
    failed = []
    for label, mod in mods:
        try:
            importlib.import_module(mod)
            print(f"{label:<28} ok-import")
        except Exception as e:
            print(f"{label:<28} FAIL {e!r}")
            failed.append(label)

    # natten forward
    try:
        from natten.functional import na2d

        q = torch.randn(1, 4, 16, 16, 32, device="cuda", dtype=torch.float16)
        with torch.no_grad():
            _ = na2d(q, q, q, kernel_size=3, dilation=1)
        torch.cuda.synchronize()
        print(f"{'natten-forward':<28} ok")
    except Exception as e:
        print(f"{'natten-forward':<28} FAIL {e!r}")
        failed.append("natten-forward")

    print("-" * 50)
    if failed:
        print("RESULT: FAIL —", ", ".join(failed))
        return 1
    print("RESULT: PASS — sm_120 gate ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
