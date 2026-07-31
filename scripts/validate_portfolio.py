"""Validate recruiter-facing portfolio claims and public-release safety."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlparse

EXPECTED_METRICS = {
    "sample_count": "117",
    "class_count": "6",
    "accuracy": "0.804",
    "macro_f1": "0.789",
}

EVIDENCE_MAP = Path("docs/portfolio/evidence-map.md")
README = Path("README.md")
README_EN = Path("README_EN.md")
NOTION_SOURCE = Path("docs/portfolio/notion-case-study.md")
PORTFOLIO_PDF = Path("output/pdf/opencv-preprocessing-advisor-portfolio.pdf")
PYPROJECT = Path("pyproject.toml")
NOTION_CASE_STUDY_URL = "https://app.notion.com/p/3aed0dc3cc1d81c0977fd982867f94e1"
REQUIRED_UI_PAGES = {
    "dataset_benchmark.py",
    "image_advisor.py",
    "methodology.py",
    "overview.py",
    "technique_explorer.py",
}
TEXT_SUFFIXES = {
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
README_METRIC_PATTERNS = {
    "sample_count": re.compile(r"\b(?P<value>117)\s+images\b", re.IGNORECASE),
    "class_count": re.compile(r"\b(?P<value>6)(?:개\s*클래스|\s+classes)\b", re.IGNORECASE),
}
BENCHMARK_ROW_PATTERN = re.compile(
    r"(?m)^\|\s*1\s*\|\s*Original\s*\|\s*RTrees\s*\|\s*"
    r"(?P<accuracy>\d+\.\d+)\s*\|\s*\*\*(?P<macro_f1>\d+\.\d+)\*\*\s*\|\s*$"
)
MARKDOWN_LINK_PATTERN = re.compile(r"\[[^\]]*\]\((?P<target><[^>]+>|[^\s)]+)(?:\s+\"[^\"]*\")?\)")
WINDOWS_USER_PATH_PATTERN = re.compile(r"(?i)C:\\Users\\(?P<user>[A-Za-z0-9][A-Za-z0-9._-]*)")
SAFETY_PATTERNS = (
    ("a GitHub OAuth token marker", re.compile(r"\bgho_[A-Za-z0-9]{20,}\b")),
    (
        "a GitHub personal-access-token marker",
        re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    ),
    ("an API key marker", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("an environment-file reference", re.compile(r"(?<![\w.\\])\.env(?![\w.-])")),
    (
        "a local MVTec dataset path",
        re.compile(
            r"(?i)(?:[A-Z]:\\|/)(?:[^\\/\r\n]+[\\/])*"
            r"mvtec(?:_anomaly_detection|[_ -]?ad)?(?:[\\/]|$)"
        ),
    ),
    (
        "a temporary PDF-render path",
        re.compile(r"(?i)(?<![A-Za-z0-9_])tmp[\\/]pdfs[\\/]"),
    ),
    (
        "an unresolved Notion-link marker",
        re.compile(
            r"(?i)\b(?:NOTION_CASE_STUDY_URL|notion(?:\s+case\s+study)?\s+(?:url|link))"
            r"\s*:\s*(?:pending|tbd|todo)\b"
        ),
    ),
)
INTERNAL_PLAN_PREFIX = Path("docs/superpowers")
INTERNAL_PLAN_TEMPLATE_MARKERS = {
    "an environment-file reference",
    "a temporary PDF-render path",
}


def _evidence_sources(evidence_map: Path) -> list[str]:
    sources: list[str] = []
    for line in evidence_map.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or line.startswith("| ---"):
            continue
        columns = line.split("|")
        if len(columns) < 7:
            continue
        sources.extend(re.findall(r"`([^`]+)`", columns[4]))
    return sources


def _readme_metrics(readme_text: str) -> dict[str, str]:
    metrics: dict[str, str] = {}
    for metric_name, pattern in README_METRIC_PATTERNS.items():
        if match := pattern.search(readme_text):
            metrics[metric_name] = match["value"]
    if match := BENCHMARK_ROW_PATTERN.search(readme_text):
        metrics["accuracy"] = match["accuracy"]
        metrics["macro_f1"] = match["macro_f1"]
    return metrics


def _tracked_text_files(repo_root: Path) -> list[Path]:
    """Return tracked text files, with a deterministic no-Git fallback for tests."""
    result = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "-z"],
        check=False,
        capture_output=True,
    )
    if result.returncode == 0:
        candidates = [Path(item) for item in result.stdout.decode().split("\0") if item]
    else:
        candidates = [
            path.relative_to(repo_root)
            for path in repo_root.rglob("*")
            if path.is_file() and ".git" not in path.parts
        ]
    return sorted(
        (
            path
            for path in candidates
            if path.suffix.casefold() in TEXT_SUFFIXES or path.name == ".gitignore"
        ),
        key=lambda path: path.as_posix(),
    )


def _markdown_anchor(heading: str) -> str:
    heading = re.sub(r"<[^>]+>", "", heading).casefold().strip()
    heading = re.sub(r"[^\w\s-]", "", heading)
    return re.sub(r"\s+", "-", heading)


def _markdown_anchors(markdown_path: Path) -> set[str]:
    anchors: set[str] = set()
    used: dict[str, int] = {}
    for line in markdown_path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line)
        if not match:
            continue
        base_anchor = _markdown_anchor(match.group(1))
        suffix = used.get(base_anchor, 0)
        used[base_anchor] = suffix + 1
        anchors.add(base_anchor if suffix == 0 else f"{base_anchor}-{suffix}")
    return anchors


def _is_intentional_troubleshooting_example(relative_path: Path, user: str) -> bool:
    return relative_path == Path("TROUBLESHOOTING.md") and user.casefold() in {
        "<username>",
        "your-username",
        "username",
    }


def _append_once(errors: list[str], error: str) -> None:
    if error not in errors:
        errors.append(error)


def _validate_public_safety(repo_root: Path, errors: list[str]) -> None:
    for relative_path in _tracked_text_files(repo_root):
        path = repo_root / relative_path
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        display_path = relative_path.as_posix()

        for match in WINDOWS_USER_PATH_PATTERN.finditer(text):
            if not _is_intentional_troubleshooting_example(relative_path, match["user"]):
                _append_once(errors, f"{display_path}: contains a local Windows user path.")

        for description, pattern in SAFETY_PATTERNS:
            if relative_path.is_relative_to(INTERNAL_PLAN_PREFIX) and description in (
                INTERNAL_PLAN_TEMPLATE_MARKERS
            ):
                continue
            if pattern.search(text):
                _append_once(errors, f"{display_path}: contains {description}.")


def _local_link_target(repo_root: Path, source_path: Path, target: str) -> Path | None:
    parsed = urlparse(target)
    if parsed.scheme or parsed.netloc:
        github_prefix = "/dansojo/opencv-preprocessing-advisor/blob/main/"
        if parsed.netloc == "github.com" and parsed.path.startswith(github_prefix):
            return repo_root / unquote(parsed.path.removeprefix(github_prefix))
        return None
    target_path = target.split("#", maxsplit=1)[0]
    if not target_path:
        return source_path
    return (source_path.parent / unquote(target_path)).resolve()


def _validate_markdown_links(repo_root: Path, errors: list[str]) -> None:
    markdown_files = [path for path in _tracked_text_files(repo_root) if path.suffix == ".md"]
    for relative_path in markdown_files:
        source_path = repo_root / relative_path
        content = source_path.read_text(encoding="utf-8")
        display_path = relative_path.as_posix()
        for match in MARKDOWN_LINK_PATTERN.finditer(content):
            target = match["target"].strip("<>")
            target_path = _local_link_target(repo_root, source_path, target)
            if target_path is None:
                continue
            try:
                target_relative = target_path.relative_to(repo_root)
            except ValueError:
                _append_once(
                    errors, f"{display_path}: Markdown link escapes the repository: {target}."
                )
                continue
            if not target_path.exists():
                _append_once(
                    errors,
                    f"{display_path}: Markdown link target does not exist: {target_relative.as_posix()}.",
                )
                continue
            if "#" not in target:
                continue
            anchor = unquote(target.split("#", maxsplit=1)[1]).casefold()
            if target_path.suffix != ".md" or anchor not in _markdown_anchors(target_path):
                _append_once(
                    errors,
                    f"{display_path}: Markdown anchor does not exist: "
                    f"{target_relative.as_posix()}#{anchor}.",
                )


def _validate_portfolio_consistency(repo_root: Path, errors: list[str]) -> None:
    notion_source = repo_root / NOTION_SOURCE
    if not notion_source.is_file():
        errors.append(f"Missing Notion source: {NOTION_SOURCE}")
    else:
        notion_text = notion_source.read_text(encoding="utf-8")
        for metric_name, expected_value in EXPECTED_METRICS.items():
            if expected_value not in notion_text:
                errors.append(f"Notion source metric {metric_name} must use {expected_value}.")

    pdf = repo_root / PORTFOLIO_PDF
    if not pdf.is_file():
        errors.append(f"Missing portfolio PDF: {PORTFOLIO_PDF}")
        return
    try:
        from pypdf import PdfReader
    except ImportError:
        errors.append("Portfolio PDF validation requires pypdf.")
        return
    reader = PdfReader(pdf)
    pdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    if len(reader.pages) != 6:
        errors.append("Portfolio PDF must contain exactly six pages.")
    for required_text in (
        "OpenCV Preprocessing Advisor",
        "117 images",
        "Macro F1 0.789",
        "https://github.com/dansojo/opencv-preprocessing-advisor",
    ):
        if required_text not in pdf_text:
            errors.append(f"Portfolio PDF is missing required evidence: {required_text}.")


def validate_claims(repo_root: Path) -> list[str]:
    """Return every missing, unsafe, or inconsistent portfolio claim as readable text."""
    errors: list[str] = []
    evidence_map = repo_root / EVIDENCE_MAP
    if not evidence_map.is_file():
        errors.append(f"Missing evidence map: {EVIDENCE_MAP}")
    else:
        sources = _evidence_sources(evidence_map)
        if not sources:
            errors.append("Evidence map contains no OpenCV source paths.")
        for source in sources:
            if not (repo_root / source).is_file():
                errors.append(f"Evidence-map source path does not exist: {source}")

    readme = repo_root / README
    if not readme.is_file():
        errors.append(f"Missing README: {README}")
    else:
        readme_metrics = _readme_metrics(readme.read_text(encoding="utf-8"))
        for metric_name, expected_value in EXPECTED_METRICS.items():
            if readme_metrics.get(metric_name) != expected_value:
                errors.append(f"README metric {metric_name} must use {expected_value}.")

    private_notion_terms = {README: "비공개", README_EN: "private"}
    for notion_readme, private_term in private_notion_terms.items():
        readme_path = repo_root / notion_readme
        if not readme_path.is_file():
            errors.append(f"Missing README: {notion_readme}")
            continue
        readme_text = readme_path.read_text(encoding="utf-8")
        if NOTION_CASE_STUDY_URL not in readme_text:
            errors.append(f"{notion_readme} must link the verified Notion case study.")
        if private_term not in readme_text.casefold():
            errors.append(
                f"{notion_readme} must disclose that the verified Notion page is currently private."
            )

    pyproject = repo_root / PYPROJECT
    required_dependency = '"opencv-python>=4.10,<5"'
    if not pyproject.is_file():
        errors.append(f"Missing package declaration: {PYPROJECT}")
    elif required_dependency not in pyproject.read_text(encoding="utf-8"):
        errors.append("Package must declare opencv-python>=4.10,<5.")

    if len(list((repo_root / "tests").glob("test_*.py"))) < 14:
        errors.append("Repository must contain at least fourteen tests/test_*.py files.")

    if not (repo_root / "app.py").is_file():
        errors.append("Missing Streamlit entry point: app.py")
    existing_pages = {
        path.name for path in (repo_root / "ui").glob("*.py") if path.name != "__init__.py"
    }
    missing_pages = sorted(REQUIRED_UI_PAGES - existing_pages)
    if missing_pages:
        errors.append(f"Missing Streamlit UI page(s): {', '.join(missing_pages)}")

    _validate_markdown_links(repo_root, errors)
    _validate_public_safety(repo_root, errors)
    _validate_portfolio_consistency(repo_root, errors)
    return errors


def main() -> None:
    errors = validate_claims(Path(__file__).resolve().parents[1])
    if errors:
        print("Portfolio validation failed:")
        print("\n".join(f"- {error}" for error in errors))
        raise SystemExit(1)
    print("Portfolio validation passed.")


if __name__ == "__main__":
    main()
