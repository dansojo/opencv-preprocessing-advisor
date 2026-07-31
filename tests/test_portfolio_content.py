import hashlib
import json
from pathlib import Path
from shutil import copytree, ignore_patterns

from scripts.validate_portfolio import validate_claims

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PORTFOLIO_ROOT = PROJECT_ROOT / "docs" / "portfolio"
BENCHMARK_EVIDENCE = PORTFOLIO_ROOT / "benchmark-evidence.json"
CONFUSION_MATRIX = PORTFOLIO_ROOT / "assets" / "mvtec-tile-best-confusion-matrix.png"
REQUIRED_CASE_STUDY_HEADINGS = {
    "문제 정의",
    "단계 진단",
    "단일 이미지 추천",
    "데이터셋 검증",
    "종합 결과",
    "실패 분석",
}
REQUIRED_EXPERIMENT_ROWS = {
    "| Original | RTrees | 0.804 | **0.789** |",
    "| CLAHE + Bilateral | RTrees | 0.766 | 0.731 |",
    "| LAB CLAHE | RTrees | 0.664 | 0.594 |",
}
REQUIRED_EVALUATION_PROTOCOL = {
    "117 images",
    "6 classes",
    "stratified 5-fold",
    "seed 42",
    "HOG + HSV/LAB histogram + Sobel/Laplacian/Gabor texture statistics",
    "SVM, kNN, and RTrees",
}
REQUIRED_LIMITATION_HEADINGS = {
    "휴리스틱 추천의 한계",
    "데이터셋 특이성",
    "GT와 MVTec 공식 평가의 부재",
    "고전 특징과 분류기의 한계",
}


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


def test_case_study_is_a_complete_canonical_source_document():
    case_study = PORTFOLIO_ROOT / "case-study.md"

    assert case_study.is_file()
    content = case_study.read_text(encoding="utf-8")
    assert all(f"## {heading}" in content for heading in REQUIRED_CASE_STUDY_HEADINGS)
    assert "LAB L-channel CLAHE" in content
    assert "fold-local scaling" in content
    assert "원본 파이프라인" in content


def test_experiment_results_records_exact_protocol_and_leaderboard():
    experiment_results = PORTFOLIO_ROOT / "experiment-results.md"

    assert experiment_results.is_file()
    content = experiment_results.read_text(encoding="utf-8")
    assert REQUIRED_EXPERIMENT_ROWS <= set(content.splitlines())
    assert all(item in content for item in REQUIRED_EVALUATION_PROTOCOL)
    assert "Hypothesis:" in content
    assert "not an official MVTec anomaly-detection metric" in content


def test_canonical_benchmark_values_match_path_free_regenerated_evidence():
    assert BENCHMARK_EVIDENCE.is_file()
    evidence = json.loads(BENCHMARK_EVIDENCE.read_text(encoding="utf-8"))

    assert evidence["dataset_interpretation"] == "MVTec AD tile/test status folders as six classes"
    assert evidence["sample_count"] == 117
    assert evidence["class_count"] == 6
    assert evidence["evaluation"] == {
        "folds": 5,
        "seed": 42,
        "feature_profile": "combined",
        "classifiers": ["svm", "knn", "rtrees"],
    }
    assert all("path" not in key.casefold() for key in evidence)
    assert not any(
        marker in BENCHMARK_EVIDENCE.read_text(encoding="utf-8")
        for marker in ("C:\\Users\\", "mvtec_anomaly_detection")
    )
    assert evidence["provenance"]["report_hashes"]
    assert (
        evidence["provenance"]["confusion_matrix_sha256"]
        == hashlib.sha256(CONFUSION_MATRIX.read_bytes()).hexdigest()
    )

    expected_rows = {
        "| {pipeline} | {classifier} | {accuracy:.3f} | {macro_f1} |".format(
            pipeline=entry["pipeline"],
            classifier=entry["classifier"],
            accuracy=entry["mean_accuracy"],
            macro_f1=(
                f"**{entry['mean_macro_f1']:.3f}**"
                if index == 0
                else f"{entry['mean_macro_f1']:.3f}"
            ),
        )
        for index, entry in enumerate(evidence["top_pipelines"])
    }
    experiment_results = (PORTFOLIO_ROOT / "experiment-results.md").read_text(encoding="utf-8")
    case_study = (PORTFOLIO_ROOT / "case-study.md").read_text(encoding="utf-8")
    assert expected_rows <= set(experiment_results.splitlines())
    assert f"Accuracy {evidence['top_pipelines'][0]['mean_accuracy']:.3f}" in case_study
    assert f"Macro F1 {evidence['top_pipelines'][0]['mean_macro_f1']:.3f}" in case_study


def test_sift_scope_is_truthful_about_current_benchmark_service_integration():
    case_study = (PORTFOLIO_ROOT / "case-study.md").read_text(encoding="utf-8")
    limitations = (PORTFOLIO_ROOT / "limitations.md").read_text(encoding="utf-8")

    for content in (case_study, limitations):
        assert "SiftBowExtractor" in content
        assert "not exposed as a BenchmarkService feature profile" in content
        assert "future fold-local vocabulary integration" in content


def test_limitations_distinguishes_heuristic_dataset_gt_and_feature_limits():
    limitations = PORTFOLIO_ROOT / "limitations.md"

    assert limitations.is_file()
    content = limitations.read_text(encoding="utf-8")
    assert all(f"## {heading}" in content for heading in REQUIRED_LIMITATION_HEADINGS)
    assert "not classification accuracy" in content
