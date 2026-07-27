"""Workspace entry — dispatches to experiment run.py (LightningAI-Lab / kaggle-lab style).

Usage:
  python main.py 001-longcat-video status
  python main.py 001-longcat-video download
  python main.py 001-longcat-video t2v

Short ids also work when unique:
  python main.py 001 status
  python main.py 001 t2v --prompt "a cat on the beach"

If the first arg is not an experiment id, defaults to 001-longcat-video.
Uses each experiment's .venv when present.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_EXP = "001-longcat-video"


def venv_python(exp_dir: Path) -> Path | None:
    if sys.platform == "win32":
        p = exp_dir / ".venv" / "Scripts" / "python.exe"
    else:
        p = exp_dir / ".venv" / "bin" / "python"
    return p if p.is_file() else None


def list_experiments() -> list[str]:
    """Dirs that look like experiments: NNN-topic with run.py."""
    names: list[str] = []
    for p in sorted(ROOT.iterdir()):
        if not p.is_dir() or p.name.startswith("."):
            continue
        if (p / "run.py").is_file():
            names.append(p.name)
    return names


def resolve_exp_id(token: str) -> str | None:
    """Exact dir name, or unique prefix (e.g. 001 → 001-longcat-video)."""
    exps = list_experiments()
    if token in exps:
        return token
    matches = [e for e in exps if e == token or e.startswith(token + "-")]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise SystemExit(
            f"ambiguous experiment '{token}': {', '.join(matches)} — use full id"
        )
    return None


def resolve_experiment(argv: list[str]) -> tuple[Path, list[str], str]:
    if not argv:
        return ROOT / DEFAULT_EXP / "run.py", [], DEFAULT_EXP
    first = argv[0]
    exp_id = resolve_exp_id(first)
    if exp_id is not None:
        return ROOT / exp_id / "run.py", argv[1:], exp_id
    return ROOT / DEFAULT_EXP / "run.py", argv, DEFAULT_EXP


def main() -> None:
    raw = sys.argv[1:]
    run_py, rest, exp_id = resolve_experiment(raw)
    if not run_py.is_file():
        known = ", ".join(list_experiments()) or "(none)"
        raise SystemExit(f"experiment script not found: {run_py}\nknown: {known}")

    exp_dir = run_py.parent
    vpy = venv_python(exp_dir)
    if vpy is not None and Path(sys.executable).resolve() != vpy.resolve():
        new_argv = [str(vpy), str(ROOT / "main.py"), exp_id, *rest]
        os.execv(str(vpy), new_argv)

    root_vpy = venv_python(ROOT)
    if (
        root_vpy is not None
        and vpy is None
        and Path(sys.executable).resolve() != root_vpy.resolve()
    ):
        new_argv = [str(root_vpy), str(ROOT / "main.py"), exp_id, *rest]
        os.execv(str(root_vpy), new_argv)

    sys.argv = [str(run_py), *rest]
    runpy.run_path(str(run_py), run_name="__main__")


if __name__ == "__main__":
    main()
