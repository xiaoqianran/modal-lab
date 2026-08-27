"""Workspace launcher — 只负责把实验 ID 分发到实验入口。

新实验默认使用 ``app.py``，由 Modal 原生 ``local_entrypoint`` 接管 CLI；尚未迁移的实验
临时兼容旧 ``run.py``。等所有实验迁移完成后即可删除 legacy 分支。

示例：
  python main.py 001 status
  python main.py 001 t2v --profile pro6000-2
  python main.py 022 probe --gpu L40S      # legacy experiment
"""

from __future__ import annotations

import os
import runpy
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_EXP = "001-longcat-video"


def entry_for(exp_dir: Path) -> Path | None:
    """新架构优先 app.py；run.py 仅作为迁移期兼容。"""
    for name in ("app.py", "run.py"):
        entry = exp_dir / name
        if entry.is_file():
            return entry
    return None


def list_experiments() -> list[str]:
    return [
        path.name
        for path in sorted(ROOT.iterdir())
        if path.is_dir()
        and not path.name.startswith(".")
        and entry_for(path) is not None
    ]


def resolve_exp_id(token: str) -> str | None:
    exps = list_experiments()
    if token in exps:
        return token

    matches = [name for name in exps if name.startswith(token + "-")]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise SystemExit(
            f"ambiguous experiment {token!r}: {', '.join(matches)} — use full id"
        )
    return None


def resolve_experiment(argv: list[str]) -> tuple[Path, list[str], str]:
    if not argv:
        exp_id = DEFAULT_EXP
        rest: list[str] = []
    else:
        resolved = resolve_exp_id(argv[0])
        if resolved is None:
            exp_id, rest = DEFAULT_EXP, argv
        else:
            exp_id, rest = resolved, argv[1:]

    exp_dir = ROOT / exp_id
    entry = entry_for(exp_dir)
    if entry is None:
        known = ", ".join(list_experiments()) or "(none)"
        raise SystemExit(f"experiment entry not found: {exp_dir}\nknown: {known}")
    return entry, rest, exp_id


def venv_python(exp_dir: Path) -> Path | None:
    relative = Path("Scripts/python.exe") if sys.platform == "win32" else Path("bin/python")
    path = exp_dir / ".venv" / relative
    return path if path.is_file() else None


def modal_executable(exp_dir: Path) -> str:
    relative = Path("Scripts/modal.exe") if sys.platform == "win32" else Path("bin/modal")
    for candidate in (exp_dir / ".venv" / relative, ROOT / ".venv" / relative):
        if candidate.is_file():
            return str(candidate)

    modal = shutil.which("modal")
    if modal:
        return modal
    raise SystemExit("找不到 modal CLI；请安装 Modal 并完成登录")


def modal_python(modal: str) -> str:
    """找到安装该 modal CLI 的 Python；用于纯本地 app.py 命令。"""
    executable = Path(modal).resolve()
    sibling = executable.parent / ("python.exe" if sys.platform == "win32" else "python")
    if sibling.is_file():
        return str(sibling)

    if sys.platform != "win32":
        try:
            first_line = executable.read_text(errors="ignore").splitlines()[0]
        except (OSError, IndexError):
            first_line = ""
        if first_line.startswith("#!"):
            interpreter = Path(first_line[2:].strip())
            if interpreter.is_file():
                return str(interpreter)

    raise SystemExit(f"无法确定 modal CLI 的 Python 环境: {modal}")


def is_local_invocation(argv: list[str]) -> bool:
    """这些操作只读/修改本地状态，不应初始化 Modal App 或构建镜像。"""
    if not argv or any(arg in {"-h", "--help"} for arg in argv):
        return True
    if argv[0] in {"status", "setup"}:
        return True
    return "--dry-run" in argv


def run_legacy(entry: Path, rest: list[str], exp_id: str) -> None:
    """迁移期兼容旧 run.py；新代码不要依赖这里。"""
    exp_dir = entry.parent
    exp_python = venv_python(exp_dir)
    if exp_python is not None and Path(sys.executable).resolve() != exp_python.resolve():
        os.execv(
            str(exp_python),
            [str(exp_python), str(ROOT / "main.py"), exp_id, *rest],
        )

    root_python = venv_python(ROOT)
    if (
        exp_python is None
        and root_python is not None
        and Path(sys.executable).resolve() != root_python.resolve()
    ):
        os.execv(
            str(root_python),
            [str(root_python), str(ROOT / "main.py"), exp_id, *rest],
        )

    sys.argv = [str(entry), *rest]
    runpy.run_path(str(entry), run_name="__main__")


def main() -> None:
    entry, rest, exp_id = resolve_experiment(sys.argv[1:])

    if entry.name == "app.py":
        modal = modal_executable(entry.parent)
        if is_local_invocation(rest):
            python = modal_python(modal)
            os.execv(python, [python, str(entry), *rest])
        os.execv(modal, [modal, "run", str(entry), *rest])

    run_legacy(entry, rest, exp_id)


if __name__ == "__main__":
    main()
