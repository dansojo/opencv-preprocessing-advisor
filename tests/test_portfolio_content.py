import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from shutil import copytree, ignore_patterns

from scripts.validate_portfolio import validate_claims

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTION_CASE_STUDY_URL = "https://app.notion.com/p/3aed0dc3cc1d81c0977fd982867f94e1"

PORTFOLIO_ROOT = PROJECT_ROOT / "docs" / "portfolio"
BENCHMARK_EVIDENCE = PORTFOLIO_ROOT / "benchmark-evidence.json"
CONFUSION_MATRIX = PORTFOLIO_ROOT / "assets" / "mvtec-tile-best-confusion-matrix.png"
PIPELINE_CONFIG = (
    PROJECT_ROOT / "src" / "opencv_preprocessing_advisor" / "config" / "pipelines.yaml"
)
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
README_ASSET_PATHS = {
    "docs/portfolio/assets/architecture.png",
    "docs/portfolio/assets/workflow.png",
    "docs/portfolio/assets/synthetic-advice-comparison.png",
    "docs/portfolio/assets/mvtec-tile-best-confusion-matrix.png",
}
README_EVIDENCE_SOURCES = {
    "src/opencv_preprocessing_advisor/diagnostics.py",
    "src/opencv_preprocessing_advisor/transforms.py",
    "src/opencv_preprocessing_advisor/features.py",
    "src/opencv_preprocessing_advisor/evaluation.py",
    "src/opencv_preprocessing_advisor/reports.py",
}
README_RESULT_ROWS = {
    "| 1 | Original | RTrees | 0.804 | **0.789** |",
    "| 2 | CLAHE + Bilateral | RTrees | 0.766 | 0.731 |",
    "| 3 | LAB CLAHE | RTrees | 0.664 | 0.594 |",
}
NOTION_CASE_STUDY_HEADINGS = {
    "프로젝트 요약",
    "배경과 요구사항 변화",
    "추천과 평가의 분리",
    "이미지 진단",
    "전처리 선택",
    "특징",
    "분류기",
    "누수 방지와 재현성",
    "MVTec 실험",
    "실패 해석",
    "트러블슈팅",
    "한계와 다음 실험",
}
NOTION_CASE_STUDY_METRICS = {"117", "6", "0.804", "0.789"}
GITHUB_MAIN_SOURCE_LINK = re.compile(
    r"https://github\.com/dansojo/opencv-preprocessing-advisor/blob/main/"
    r"(?:src|tests|scripts|docs)/[^)\s]+"
)
GITHUB_MAIN_DOCUMENT_LINK = re.compile(
    r"https://github\.com/dansojo/opencv-preprocessing-advisor/blob/main/"
    r"(?:docs/portfolio|output/pdf)/[^)\s]+"
)
RELATIVE_MARKDOWN_LINK = re.compile(r"\]\((?!https?://|mailto:|#)[^)]+\)")
GFM_TABLE_HEADER = re.compile(r"(?m)^\|[^\n]+\|\n\|\s*:?-{3,}.*\|\s*$")
NOTION_CALLOUTS = {
    "휴리스틱 점수": ("⚠️", "yellow_bg"),
    "MVTec 공식 지표": ("⚠️", "yellow_bg"),
    "원본 파이프라인의 승리": ("💡", "green_bg"),
}
NOTION_TABLE_HEADERS = (
    ("Pipeline", "Classifier", "Accuracy", "Macro F1"),
    ("증상", "먼저 확인할 근거", "대응"),
)


def test_claim_validator_accepts_current_repository():
    assert validate_claims(PROJECT_ROOT) == []


def test_claim_validator_runs_as_a_script():
    result = subprocess.run(
        [sys.executable, "scripts/validate_portfolio.py"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Portfolio validation passed." in result.stdout


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


def test_claim_validator_rejects_missing_verified_notion_links(tmp_path):
    copied_root = tmp_path / "repository"
    copytree(
        PROJECT_ROOT,
        copied_root,
        ignore=ignore_patterns(".git", ".pytest_cache", ".ruff_cache", ".superpowers"),
    )
    for filename in ("README.md", "README_EN.md"):
        readme = copied_root / filename
        content = readme.read_text(encoding="utf-8").replace(
            NOTION_CASE_STUDY_URL,
            "NOTION_CASE_STUDY_URL" + ": pending",
        )
        readme.write_text(content, encoding="utf-8")

    errors = validate_claims(copied_root)

    assert "README.md must link the verified Notion case study." in errors
    assert "README_EN.md must link the verified Notion case study." in errors


def test_claim_validator_rejects_public_safety_markers_and_broken_markdown(tmp_path):
    copied_root = tmp_path / "repository"
    copytree(
        PROJECT_ROOT,
        copied_root,
        ignore=ignore_patterns(".git", ".pytest_cache", ".ruff_cache", ".superpowers"),
    )
    readme = copied_root / "README.md"
    unsafe_content = "\n".join(
        (
            readme.read_text(encoding="utf-8"),
            "local_path = " + "C:" + "\\Users" + "\\release-user\\dataset",
            "oauth = " + "gh" + "o_" + "a" * 36,
            "pat = " + "github" + "_pat_" + "a" * 24,
            "api_key = " + "s" + "k-" + "a" * 24,
            "project_api_key = " + "s" + "k-proj-" + "a" * 24,
            "dataset = " + "C:" + "\\datasets\\mvtec" + "_anomaly_detection\\tile",
            "[temporary render](" + "tmp" + "/pdfs/rendered/page-1.png)",
            "NOTION_CASE_STUDY_URL" + ": pending",
            "[missing file](docs/missing.md)",
            "[missing anchor](docs/portfolio/case-study.md#does-not-exist)",
        )
    )
    readme.write_text(unsafe_content, encoding="utf-8")

    errors = validate_claims(copied_root)

    expected_errors = {
        "README.md: contains a local Windows user path.",
        "README.md: contains a GitHub OAuth token marker.",
        "README.md: contains a GitHub personal-access-token marker.",
        "README.md: contains an API key marker.",
        "README.md: contains a local MVTec dataset path.",
        "README.md: contains a temporary PDF-render path.",
        "README.md: contains an unresolved Notion-link marker.",
        "README.md: Markdown link target does not exist: docs/missing.md.",
        "README.md: Markdown anchor does not exist: docs/portfolio/case-study.md#does-not-exist.",
    }
    assert expected_errors <= set(errors)


def test_claim_validator_rejects_hyphenated_openai_project_api_keys(tmp_path):
    copied_root = tmp_path / "repository"
    copytree(
        PROJECT_ROOT,
        copied_root,
        ignore=ignore_patterns(".git", ".pytest_cache", ".ruff_cache", ".superpowers"),
    )
    readme = copied_root / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8") + "\nproject_api_key = " + "s" + "k-proj-" + "a" * 24,
        encoding="utf-8",
    )

    errors = validate_claims(copied_root)

    assert "README.md: contains an API key marker." in errors


def test_claim_validator_scans_tracked_extensionless_environment_files(tmp_path):
    copied_root = tmp_path / "repository"
    copytree(
        PROJECT_ROOT,
        copied_root,
        ignore=ignore_patterns(".git", ".pytest_cache", ".ruff_cache", ".superpowers"),
    )
    environment_filename = "." + "env"
    environment_file = copied_root / environment_filename
    environment_file.write_text("GH_TOKEN=ghp_" + "a" * 36, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=copied_root, check=True)
    subprocess.run(["git", "add", "-f", "."], cwd=copied_root, check=True)

    errors = validate_claims(copied_root)

    assert f"{environment_filename}: contains a GitHub personal-access-token marker." in errors
    assert ".gitignore: contains an environment-file reference." not in errors
    assert "Portfolio PDF does not match a fresh deterministic build." not in errors


def test_claim_validator_checks_reference_links_autolinks_and_fenced_headings(tmp_path):
    copied_root = tmp_path / "repository"
    copytree(
        PROJECT_ROOT,
        copied_root,
        ignore=ignore_patterns(".git", ".pytest_cache", ".ruff_cache", ".superpowers"),
    )
    (copied_root / "docs" / "anchors.md").write_text(
        "# Visible heading\n\n```markdown\n# Hidden heading\n```\n",
        encoding="utf-8",
    )
    readme = copied_root / "README.md"
    readme.write_text(
        "\n".join(
            (
                readme.read_text(encoding="utf-8"),
                "[visible reference][visible]",
                "[missing reference][missing-target]",
                "[fenced heading][hidden-heading]",
                "[missing shortcut]",
                "![missing image][missing-image]",
                "[visible]: docs/anchors.md#visible-heading",
                "[missing-target]: docs/reference-missing.md",
                "[hidden-heading]: docs/anchors.md#hidden-heading",
                "[missing shortcut]: docs/shortcut-missing.md",
                "[missing-image]: docs/image-missing.png",
                "<https://github.com/dansojo/opencv-preprocessing-advisor/blob/main/docs/autolink-missing.md>",
            )
        ),
        encoding="utf-8",
    )

    errors = validate_claims(copied_root)

    expected_errors = {
        "README.md: Markdown link target does not exist: docs/reference-missing.md.",
        "README.md: Markdown anchor does not exist: docs/anchors.md#hidden-heading.",
        "README.md: Markdown link target does not exist: docs/autolink-missing.md.",
        "README.md: Markdown link target does not exist: docs/shortcut-missing.md.",
        "README.md: Markdown link target does not exist: docs/image-missing.png.",
    }
    assert expected_errors <= set(errors)


def test_public_release_checklist_records_the_private_notion_gate():
    checklist = PROJECT_ROOT / "docs" / "PUBLIC_RELEASE_CHECKLIST.md"

    content = checklist.read_text(encoding="utf-8")

    assert NOTION_CASE_STUDY_URL in content
    assert "private" in content.casefold()
    assert "explicit owner approval" in content.casefold()
    assert "anonymous" in content.casefold()


def test_clean_machine_portfolio_dependencies_and_synthetic_sample_are_documented():
    development_requirements = (PROJECT_ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
    for dependency in ("reportlab", "pypdf", "pdfplumber"):
        assert dependency in development_requirements

    sample = PROJECT_ROOT / "data" / "samples" / "synthetic-tile.png"
    assert sample.read_bytes().startswith(b"\x89PNG")

    for readme_name in ("README.md", "README_EN.md"):
        content = (PROJECT_ROOT / readme_name).read_text(encoding="utf-8")
        assert "python -m pip install -r requirements-dev.txt" in content
        assert "data/samples/synthetic-tile.png" in content
        assert "docs/portfolio/assets/streamlit-advisor-synthetic.png" in content


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
    assert "documented/default evidence run" in content
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
        evidence["provenance"]["pipeline_config_sha256"]
        == hashlib.sha256(PIPELINE_CONFIG.read_bytes()).hexdigest()
    )
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


def test_korean_readme_is_a_visual_evidence_driven_portfolio_entrypoint():
    content = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert all(path in content for path in README_ASSET_PATHS)
    assert all(f"]({source})" in content for source in README_EVIDENCE_SOURCES)
    assert "휴리스틱 점수는 정확도가 아닙니다" in content
    assert "`tile/test`" in content
    assert "6개 분류 클래스로 해석" in content
    assert README_RESULT_ROWS <= set(content.splitlines())
    assert "pytest -q" in content
    assert "## 빠른 시작" in content
    assert "](docs/portfolio/limitations.md)" in content
    assert "](output/pdf/opencv-preprocessing-advisor-portfolio.pdf)" in content
    assert f"[상세 Notion 케이스 스터디]({NOTION_CASE_STUDY_URL})" in content
    assert "NOTION_CASE_STUDY_URL" + ": pending" not in content


def test_english_readme_mirrors_the_portfolio_claims_without_new_metrics():
    content = (PROJECT_ROOT / "README_EN.md").read_text(encoding="utf-8")

    assert all(path in content for path in README_ASSET_PATHS)
    assert "heuristic, not an accuracy estimate" in content
    assert "117 images" in content
    assert "six classes" in content
    assert README_RESULT_ROWS <= set(content.splitlines())
    assert "pytest -q" in content
    assert "](docs/portfolio/limitations.md)" in content
    assert "](output/pdf/opencv-preprocessing-advisor-portfolio.pdf)" in content
    assert f"[Detailed Notion case study]({NOTION_CASE_STUDY_URL})" in content
    assert "NOTION_CASE_STUDY_URL" + ": pending" not in content


def test_local_notion_case_study_is_complete_and_traceable():
    notion_case_study = PORTFOLIO_ROOT / "notion-case-study.md"

    assert notion_case_study.is_file()
    content = notion_case_study.read_text(encoding="utf-8")
    assert all(f"## {heading}" in content for heading in NOTION_CASE_STUDY_HEADINGS)
    assert NOTION_CASE_STUDY_METRICS <= set(re.findall(r"\d+(?:\.\d+)?", content))
    assert len(set(GITHUB_MAIN_SOURCE_LINK.findall(content))) >= 15
    assert "<table_of_contents/>" in content
    assert "## 목차" not in content
    assert not RELATIVE_MARKDOWN_LINK.findall(content)
    assert "> [!WARNING]" not in content
    assert "> [!TIP]" not in content
    assert len(set(GITHUB_MAIN_DOCUMENT_LINK.findall(content))) >= 5
    assert not GFM_TABLE_HEADER.search(content)
    for title, (icon, color) in NOTION_CALLOUTS.items():
        matches = list(
            re.finditer(
                rf'<callout icon="{icon}" color="{color}">\n'
                rf"(?P<body>(?:\t.*\n)+?)</callout>",
                content,
            )
        )
        matching = [match for match in matches if f"\t**{title}**" in match["body"]]
        assert matching, f"missing Notion callout for {title}"
        assert all(
            not line or line.startswith("\t")
            for match in matching
            for line in match["body"].splitlines()
        )
    tables = re.findall(
        r'<table fit-page-width="true" header-row="true">\n'
        r"(?P<body>.*?)</table>",
        content,
        re.DOTALL,
    )
    assert len(tables) == 2
    for headers, table in zip(NOTION_TABLE_HEADERS, tables, strict=True):
        expected_header = "\n".join(
            ("\t<tr>", *(f"\t\t<td>{header}</td>" for header in headers), "\t</tr>")
        )
        assert table.startswith(f"{expected_header}\n")
        assert all(not line or line.startswith("\t") for line in table.splitlines())
