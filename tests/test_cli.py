import subprocess
import sys

from opencv_preprocessing_advisor.io import encode_png


def test_cli_analyze_image_writes_report(tmp_path, sample_bgr):
    sample_path = tmp_path / "sample.png"
    sample_path.write_bytes(encode_png(sample_bgr))

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "opencv_preprocessing_advisor.cli",
            "analyze-image",
            "--image",
            str(sample_path),
            "--profile",
            "auto",
            "--output",
            str(tmp_path / "out"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert any((tmp_path / "out").glob("image_advisor/*/recommendations.json"))


def test_cli_rejects_missing_dataset(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "opencv_preprocessing_advisor.cli",
            "benchmark",
            "--dataset",
            str(tmp_path / "missing"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "does not exist" in completed.stderr


def test_cli_self_check_writes_verified_artifacts(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "opencv_preprocessing_advisor.cli",
            "self-check",
            "--output",
            str(tmp_path / "self-check"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "SELF-CHECK PASSED" in completed.stdout
    assert (tmp_path / "self-check" / "fixtures" / "low_contrast.png").exists()
