#!/usr/bin/env python3
"""Gate: ensure critical CUDA extensions contain Ada sm_89 (or sm_86) cubins.

Run inside a container that already has the extensions installed, e.g.:

  python scripts/verify_sm89.py
  python scripts/verify_sm89.py --packages natten,flex_gemm,o_voxel

Exit codes:
  0 = all present packages pass
  1 = at least one failure
  2 = nothing to check / import errors for required packages
"""
from __future__ import annotations

import argparse
import glob
import importlib
import os
import re
import struct
import sys
import zipfile
from pathlib import Path
from typing import Iterable

# Packages that must be sm_89 for Pixal3D/TRELLIS.2 on L40S
DEFAULT_PACKAGES = [
    "flex_gemm",
    "o_voxel",
    "cumesh",
    "nvdiffrast",
    "renderutils",
]

# Cubins that are OK on L40S (Ada 8.x family). sm_86 runs on sm_89.
OK_SMS = {"sm_86", "sm_89"}
# Definitely wrong for L40S if these are the *only* architectures present
BAD_ONLY = {"sm_90", "sm_100", "sm_120", "sm_121"}

SM_RE = re.compile(rb"sm_(\d+)")


def _module_search_roots(mod) -> list[Path]:
    roots: list[Path] = []
    f = getattr(mod, "__file__", None)
    if f:
        roots.append(Path(f).resolve().parent)
    for p in getattr(mod, "__path__", []) or []:
        roots.append(Path(p).resolve())
    return roots


def _iter_so_files(roots: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        out.extend(Path(p) for p in glob.glob(str(root / "**" / "*.so"), recursive=True))
    # de-dupe
    seen = set()
    uniq = []
    for p in out:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def _sms_from_bytes(data: bytes) -> set[str]:
    found = {f"sm_{m.group(1).decode()}" for m in SM_RE.finditer(data)}
    return found


def _sms_from_so(path: Path) -> set[str]:
    try:
        data = path.read_bytes()
    except OSError as e:
        print(f"  ! cannot read {path}: {e}")
        return set()
    return _sms_from_bytes(data)


def _sms_from_wheel(path: Path) -> set[str]:
    sms: set[str] = set()
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if name.endswith(".so"):
                sms |= _sms_from_bytes(zf.read(name))
    return sms


def _classify(sms: set[str]) -> str:
    if not sms:
        return "unknown"  # nvdiffrast may JIT; treat as soft
    if sms & OK_SMS:
        return "ok"
    if sms <= BAD_ONLY or (sms & BAD_ONLY and not (sms & OK_SMS)):
        # has only bad, or has bad without any ok
        if not (sms & OK_SMS):
            return "bad"
    # mixed or other sm (e.g. sm_80 only) — not ok for L40S
    if sms & OK_SMS:
        return "ok"
    return "bad"


def check_package(name: str, required: bool) -> dict:
    result = {
        "name": name,
        "import_ok": False,
        "so_count": 0,
        "sms": set(),
        "status": "missing",
        "detail": "",
    }
    try:
        mod = importlib.import_module(name)
    except Exception as e:
        result["detail"] = f"import failed: {e}"
        result["status"] = "missing" if required else "optional-missing"
        return result

    result["import_ok"] = True
    sos = _iter_so_files(_module_search_roots(mod))
    result["so_count"] = len(sos)
    sms: set[str] = set()
    for so in sos:
        sms |= _sms_from_so(so)
    result["sms"] = sms

    # natten special: prefer HAS_LIBNATTEN
    if name == "natten":
        has_lib = bool(getattr(mod, "HAS_LIBNATTEN", False))
        result["detail"] = f"HAS_LIBNATTEN={has_lib}"
        if not has_lib:
            result["status"] = "bad"
            result["detail"] += " (need real libnatten CUDA build)"
            return result

    cls = _classify(sms)
    if cls == "ok":
        result["status"] = "ok"
    elif cls == "unknown":
        # no sm string in .so — JIT packages: require env arch hint
        arch = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
        if "8.9" in arch or arch.strip() in {"8.9", "89"}:
            result["status"] = "ok-jit"
            result["detail"] = (result["detail"] + " | ").lstrip(" |")
            result["detail"] += f"no cubin tag; TORCH_CUDA_ARCH_LIST={arch!r} (assume JIT sm_89)"
        else:
            result["status"] = "warn"
            result["detail"] = (
                result["detail"] + " | " if result["detail"] else ""
            ) + "no sm_* tag in .so; set TORCH_CUDA_ARCH_LIST=8.9 for JIT"
    else:
        result["status"] = "bad"
        result["detail"] = f"architectures {sorted(sms)} not usable on L40S (need sm_89/sm_86)"

    return result


def check_wheel_dir(wheel_dir: Path) -> list[dict]:
    rows = []
    for whl in sorted(wheel_dir.glob("*.whl")):
        sms = _sms_from_wheel(whl)
        cls = _classify(sms)
        rows.append(
            {
                "name": whl.name,
                "import_ok": True,
                "so_count": -1,
                "sms": sms,
                "status": "ok" if cls == "ok" else ("unknown" if cls == "unknown" else "bad"),
                "detail": "wheel scan",
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--packages",
        default=",".join(DEFAULT_PACKAGES),
        help="comma-separated module names",
    )
    p.add_argument(
        "--required",
        default="flex_gemm,o_voxel,cumesh",
        help="comma-separated required modules (fail hard if missing/bad)",
    )
    p.add_argument(
        "--wheel-dir",
        type=Path,
        default=None,
        help="optional directory of .whl to scan without importing",
    )
    p.add_argument(
        "--expect-gpu",
        action="store_true",
        help="also require torch cuda device capability (8,9)",
    )
    args = p.parse_args(argv)

    print("=== verify_sm89 · Ada / L40S gate ===")
    failures = 0
    required = {x.strip() for x in args.required.split(",") if x.strip()}
    packages = [x.strip() for x in args.packages.split(",") if x.strip()]

    if args.expect_gpu:
        try:
            import torch

            if not torch.cuda.is_available():
                print("FAIL: CUDA not available")
                return 2
            cap = torch.cuda.get_device_capability()
            name = torch.cuda.get_device_name(0)
            print(f"GPU: {name}  capability={cap}")
            if cap != (8, 9):
                print(f"FAIL: expected compute capability (8, 9), got {cap}")
                failures += 1
            else:
                print("OK: compute capability 8.9 (Ada / L40S family)")
        except Exception as e:
            print(f"FAIL: torch gpu check: {e}")
            failures += 1

    results = []
    if args.wheel_dir:
        if not args.wheel_dir.is_dir():
            print(f"FAIL: wheel dir not found: {args.wheel_dir}")
            return 2
        results.extend(check_wheel_dir(args.wheel_dir))
    else:
        for name in packages:
            results.append(check_package(name, required=name in required))

    print()
    print(f"{'package':<28} {'status':<14} sms / detail")
    print("-" * 72)
    hard_missing = 0
    for r in results:
        sms = ",".join(sorted(r["sms"])) if r["sms"] else "-"
        line = f"{r['name']:<28} {r['status']:<14} {sms}"
        if r["detail"]:
            line += f"  ({r['detail']})"
        print(line)
        st = r["status"]
        is_req = r["name"] in required or (
            args.wheel_dir and st == "bad"
        )
        if st in {"bad", "missing"} and (r["name"] in required or args.wheel_dir):
            failures += 1
            if st == "missing":
                hard_missing += 1
        elif st == "missing" and r["name"] not in required:
            pass  # optional
        elif st == "warn" and r["name"] in required:
            failures += 1

    print("-" * 72)
    if failures:
        print(f"RESULT: FAIL ({failures} issue(s)) — do NOT run inference")
        return 1 if hard_missing == 0 or failures else 2
    print("RESULT: PASS — sm_89 gate ok (still need end-to-end smoke later)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
