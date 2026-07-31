import subprocess
import sys

from scripts.build_portfolio_assets import (
    COMMITTED_KOREAN_FONT,
    _configure_font,
    build_assets,
    build_synthetic_sample,
)


def test_build_assets_creates_required_pngs(tmp_path):
    outputs = build_assets(tmp_path)

    assert set(outputs) == {
        "architecture",
        "workflow",
        "synthetic_advice_comparison",
    }
    for path in outputs.values():
        assert path.read_bytes().startswith(b"\x89PNG")


def test_asset_builder_uses_the_committed_korean_font_not_a_host_font():
    assert COMMITTED_KOREAN_FONT.is_file()
    assert _configure_font() == "Noto Sans KR"


def test_build_synthetic_sample_creates_a_redistributable_png(tmp_path):
    sample = build_synthetic_sample(tmp_path / "synthetic-tile.png")

    assert sample == tmp_path / "synthetic-tile.png"
    assert sample.read_bytes().startswith(b"\x89PNG")


def test_asset_builder_cli_creates_the_documented_synthetic_sample(tmp_path):
    output = tmp_path / "assets"
    sample = tmp_path / "synthetic-tile.png"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_portfolio_assets.py",
            "--output",
            str(output),
            "--sample",
            str(sample),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "synthetic_sample:" in result.stdout
    assert sample.read_bytes().startswith(b"\x89PNG")
