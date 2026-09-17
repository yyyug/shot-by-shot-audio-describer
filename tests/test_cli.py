import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_cli_help():
    # Use sys.executable (the interpreter running the tests) instead of a bare
    # "python": the latter resolves to the system interpreter, which does not
    # have the project dependencies installed.
    result = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "run_processing.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0
    assert "video" in result.stdout.lower()
