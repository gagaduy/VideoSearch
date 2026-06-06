import subprocess
import sys


def test_gpu_check_script_reports_cuda_field() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_gpu.py"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "cuda_available" in result.stdout

