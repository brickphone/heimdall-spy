"""Run all quality gates and return failure if any gate fails."""

import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Run ruff, mypy and pytest independently so existing failures remain visible."""
    root = Path(__file__).resolve().parents[1]
    commands = [
        [sys.executable, "-m", "ruff", "check", "swespy", "scripts", "tests", "eval"],
        [sys.executable, "-m", "mypy", "--ignore-missing-imports", "swespy", "scripts", "eval"],
        [sys.executable, "-B", "-m", "pytest", "-p", "no:cacheprovider"],
    ]
    codes = [subprocess.run(command, cwd=root, check=False).returncode for command in commands]
    return int(any(codes))


if __name__ == "__main__":
    raise SystemExit(main())
