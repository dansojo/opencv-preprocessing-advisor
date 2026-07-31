from pathlib import Path
from shutil import copytree, ignore_patterns

from scripts.validate_portfolio import validate_claims

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_claim_validator_accepts_current_repository():
    assert validate_claims(PROJECT_ROOT) == []


def test_claim_validator_rejects_incorrect_benchmark_accuracy(tmp_path):
    copied_root = tmp_path / "repository"
    copytree(
        PROJECT_ROOT,
        copied_root,
        ignore=ignore_patterns(".git", ".pytest_cache", ".ruff_cache", ".superpowers"),
    )
    readme = copied_root / "README.md"
    changed_readme = readme.read_text(encoding="utf-8").replace("0.804", "0.999")
    readme.write_text(f"{changed_readme}\nHistorical note: 0.804\n", encoding="utf-8")

    errors = validate_claims(copied_root)

    assert "README metric accuracy must use 0.804." in errors
