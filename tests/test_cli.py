import subprocess


def test_cli_help():
    result = subprocess.run(
        ["python", "run_processing.py", "--help"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "video" in result.stdout.lower()
