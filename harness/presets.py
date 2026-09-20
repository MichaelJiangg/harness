"""预置钩子的可复用逻辑：项目约定、写后格式化和会话记忆。"""

from pathlib import Path
import shutil
import subprocess


FORMAT_TIMEOUT = 60


def format_written_file(workspace, path):
    root = Path(workspace).resolve()
    target = (root / path).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        return None
    suffix = target.suffix.lower()
    command = None
    if suffix == ".py":
        if shutil.which("ruff"):
            command = ["ruff", "format", str(target)]
        elif shutil.which("black"):
            command = ["black", str(target)]
    elif suffix in {".js", ".jsx", ".ts", ".tsx", ".css", ".html"}:
        prettier = root / "node_modules" / ".bin" / "prettier"
        if prettier.is_file():
            command = [str(prettier), "--write", str(target)]
    if command is None:
        return None
    completed = subprocess.run(
        command, cwd=root, capture_output=True, text=True, timeout=FORMAT_TIMEOUT,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "formatter exited with error")
    return command[0]
