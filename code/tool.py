"""Legacy compatibility wrapper for the refactored toolkit."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orr_vae.tool import *  # noqa: F401,F403
