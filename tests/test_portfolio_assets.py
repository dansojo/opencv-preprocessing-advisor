from scripts.build_portfolio_assets import build_assets


def test_build_assets_creates_required_pngs(tmp_path):
    outputs = build_assets(tmp_path)

    assert set(outputs) == {
        "architecture",
        "workflow",
        "synthetic_advice_comparison",
    }
    for path in outputs.values():
        assert path.read_bytes().startswith(b"\x89PNG")
