#!/usr/bin/env python3
"""Assemble GitHub Pages site from pages/ hub + each experiment gallery."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "_site"
PAGES = ROOT / "pages"


def copytree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(
        src,
        dst,
        symlinks=False,
        ignore=shutil.ignore_patterns(
            "serve.py",
            ".DS_Store",
            "__pycache__",
            "*.pyc",
        ),
    )


def main() -> int:
    if not PAGES.is_dir():
        print("missing pages/", file=sys.stderr)
        return 1

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    # hub
    for item in PAGES.iterdir():
        target = OUT / item.name
        if item.is_dir():
            copytree(item, target)
        else:
            shutil.copy2(item, target)

    # experiment galleries → /NNN-topic/
    n = 0
    for gallery in sorted(ROOT.glob("*/gallery")):
        exp = gallery.parent.name
        if exp.startswith("."):
            continue
        if not (gallery / "index.html").is_file():
            continue
        copytree(gallery, OUT / exp)
        n += 1
        print(f"  + {exp}/")

    # gpu-gallery special
    gpu = ROOT / "gpu-gallery"
    if (gpu / "index.html").is_file():
        copytree(gpu, OUT / "gpu-gallery")
        print("  + gpu-gallery/")
        n += 1

    # tts-gallery hub (parent may hold only gallery/ subdir — covered by glob;
    # also allow flat tts-gallery/index.html)
    tts = ROOT / "tts-gallery"
    if (tts / "index.html").is_file() and not (tts / "gallery" / "index.html").is_file():
        copytree(tts, OUT / "tts-gallery")
        print("  + tts-gallery/ (flat)")
        n += 1

    # nojekyll for GH Pages
    (OUT / ".nojekyll").write_text("")
    # simple 404 → home
    (OUT / "404.html").write_text(
        """<!DOCTYPE html><meta charset=utf-8><title>Redirect</title>
<meta http-equiv="refresh" content="0;url=./">
<script>location.replace('./')</script>
<p><a href="./">Back to modal-lab</a></p>
"""
    )

    # size report
    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"built {OUT} · {n} galleries · {total/1e6:.1f} MB")
    if not (OUT / "index.html").is_file():
        print("missing hub index", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
