import subprocess
import sys
from pathlib import Path

from ase.db import connect


def test_generate_random_smoke(tmp_path: Path):
    out_dir = tmp_path / "data"
    cmd = [
        sys.executable,
        "-m",
        "orr_vae",
        "generate-random",
        "--num",
        "2",
        "--seed",
        "0",
        "--output_dir",
        str(out_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    db_path = out_dir / "iter0_structures.json"
    assert db_path.exists()

    db = connect(str(db_path))
    rows = list(db.select())
    assert len(rows) == 2
