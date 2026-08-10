#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""005-pixal3d CLI — GLB 写入远程 Modal Volume。"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
MODAL_APP = EXP_DIR / "modal_app.py"
VOL_OUT = "modal-lab-pixal3d-outputs"
DEFAULT_GPU = "H100"  # 与 HF demo 轮子对齐
DEFAULT_IMAGE = EXP_DIR / "inputs" / "sample.webp"
SAMPLE_URL = (
    "https://raw.githubusercontent.com/TencentARC/Pixal3D/master/"
    "assets/images/5_img.webp"
)


def _modal() -> str:
    m = shutil.which("modal")
    if not m:
        raise SystemExit("未找到 modal CLI，请先 pip install modal && modal token new")
    return m


def _run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def cmd_status(_: argparse.Namespace) -> int:
    print("GLB 位置（远程 Volume）:")
    print(f"  volume : {VOL_OUT}")
    print("  path   : meshes/<name>.glb  +  meshes/latest.glb")
    print("  列表页 : https://seachenxyt--modal-lab-pixal3d-index.modal.run")
    print("  下载   : https://seachenxyt--modal-lab-pixal3d-download.modal.run?name=latest")
    print(f"  CLI    : modal volume ls {VOL_OUT} meshes")
    print(f"  默认GPU: {DEFAULT_GPU} · low_vram · 1024")
    print("  选卡   : H100 推荐；A100-40GB 需先 build-natten；PRO6000/L40S 当前不可用")
    return _run([_modal(), "run", str(MODAL_APP), "--action", "status"])


def cmd_download(ns: argparse.Namespace) -> int:
    cmd = [_modal(), "run", str(MODAL_APP), "--action", "download"]
    if ns.force:
        cmd.append("--force-download")
    if ns.no_aux:
        cmd.append("--no-with-aux")
    return _run(cmd)


def cmd_build_natten(ns: argparse.Namespace) -> int:
    print(
        f"[build-natten] 在 {ns.gpu} 上编译 natten 并缓存到 Volume "
        f"（A100 首次约 20–30min，之后复用）",
        flush=True,
    )
    return _run(
        [_modal(), "run", str(MODAL_APP), "--action", "build-natten", "--gpu", ns.gpu]
    )


def cmd_smoke(ns: argparse.Namespace) -> int:
    return _run(
        [_modal(), "run", str(MODAL_APP), "--action", "smoke", "--gpu", ns.gpu]
    )


def cmd_list_outputs(_: argparse.Namespace) -> int:
    print(f"=== remote volume {VOL_OUT}/meshes ===", flush=True)
    _run([_modal(), "volume", "ls", VOL_OUT, "meshes"])
    return _run([_modal(), "run", str(MODAL_APP), "--action", "list-outputs"])


def cmd_pull(ns: argparse.Namespace) -> int:
    """从 Volume 拉 GLB 到本地（可选）。"""
    dest = Path(ns.dest).expanduser()
    dest.parent.mkdir(parents=True, exist_ok=True)
    name = ns.name
    remote = f"meshes/{name if name.endswith('.glb') else name + '.glb'}"
    return _run([_modal(), "volume", "get", VOL_OUT, remote, str(dest)])


def cmd_i2v(ns: argparse.Namespace) -> int:
    image = ns.image
    if not image and not ns.image_url:
        if DEFAULT_IMAGE.is_file():
            image = str(DEFAULT_IMAGE)
        else:
            ns.image_url = SAMPLE_URL

    print(
        f"[i2v] GPU={ns.gpu} low_vram={ns.low_vram} res={ns.resolution} "
        f"→ Volume {VOL_OUT}/meshes/{ns.output_name}.glb",
        flush=True,
    )
    cmd = [
        _modal(),
        "run",
        str(MODAL_APP),
        "--action",
        "i2v",
        "--output-name",
        ns.output_name,
        "--seed",
        str(ns.seed),
        "--resolution",
        str(ns.resolution),
        "--fov",
        str(ns.fov),
        "--gpu",
        ns.gpu,
    ]
    if image:
        cmd.extend(["--image", image])
    if ns.image_url:
        cmd.extend(["--image-url", ns.image_url])
    if ns.low_vram:
        cmd.append("--low-vram")
    else:
        cmd.append("--no-low-vram")

    rc = _run(cmd)
    if rc == 0:
        print()
        print(">>> 远程 Volume:", flush=True)
        print(f"    modal volume ls {VOL_OUT} meshes", flush=True)
        print(
            f"    https://seachenxyt--modal-lab-pixal3d-download.modal.run"
            f"?name={ns.output_name}",
            flush=True,
        )
        _run([_modal(), "volume", "ls", VOL_OUT, "meshes"])
    return rc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="005-pixal3d · Pixal3D image→GLB on Modal")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="打印 app / volume / 下载链接")

    d = sub.add_parser("download", help="下载主权重 + 辅助模型到 Volume")
    d.add_argument("--force", action="store_true")
    d.add_argument(
        "--no-aux",
        action="store_true",
        help="只下 TencentARC/Pixal3D",
    )

    bn = sub.add_parser(
        "build-natten",
        help="在目标 GPU 编译 natten 缓存到 Volume（A100 首次需要）",
    )
    bn.add_argument("--gpu", default="A100-40GB", help="默认 A100-40GB")

    sm = sub.add_parser("smoke", help="官方样例图冒烟（默认 H100 · low_vram · 1024）")
    sm.add_argument("--gpu", default=DEFAULT_GPU)

    sub.add_parser("list-outputs", help="列出远程 Volume 上的 GLB")

    pl = sub.add_parser("pull", help="从 Volume 下载 GLB 到本地")
    pl.add_argument("--name", default="latest")
    pl.add_argument("--dest", default=str(EXP_DIR / "outputs" / "latest.glb"))

    t = sub.add_parser("i2v", help="image-to-3D → GLB")
    t.add_argument("--image", default="", help="本地图片路径")
    t.add_argument("--image-url", default="", help="远程图片 URL")
    t.add_argument("--output-name", default="i2v")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument(
        "--resolution",
        type=int,
        default=1024,
        choices=[1024, 1536],
    )
    t.add_argument("--fov", type=float, default=-1.0)
    t.add_argument(
        "--full-vram",
        action="store_true",
        help="关闭 low_vram（更快，需 ~18GB+）",
    )
    t.add_argument("--gpu", default=DEFAULT_GPU)

    ns = p.parse_args(argv)
    if ns.cmd == "i2v":
        ns.low_vram = not ns.full_vram

    return int(
        {
            "status": cmd_status,
            "download": cmd_download,
            "build-natten": cmd_build_natten,
            "smoke": cmd_smoke,
            "list-outputs": cmd_list_outputs,
            "pull": cmd_pull,
            "i2v": cmd_i2v,
        }[ns.cmd](ns)
        or 0
    )


if __name__ == "__main__":
    sys.exit(main())
