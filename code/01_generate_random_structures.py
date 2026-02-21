#!/usr/bin/env python
"""Legacy compatibility wrapper."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> int:
    return subprocess.call([sys.executable, "-m", "orr_vae.cli.main", "generate-random", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
