import subprocess
import sys


def test_cli_help():
    cmd = [sys.executable, "-m", "orr_vae", "--help"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0
    assert "ORR catalyst screening workflows" in proc.stdout
    assert "calc-orr" in proc.stdout
    assert "train-cvae" in proc.stdout
    assert "generate-random" not in proc.stdout
    assert "run-pipeline" not in proc.stdout
