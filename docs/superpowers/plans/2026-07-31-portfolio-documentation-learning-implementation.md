# Portfolio Documentation and OpenCV Learning Track Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a recruiter-ready GitHub README, a visually verified six-page PDF portfolio, a detailed Notion case study, and a complete four-week OpenCV learning pack grounded in the existing repository.

**Architecture:** Repository code, tests, and reproducible benchmark artifacts are the evidence source. Focused Markdown source documents under `docs/portfolio/` provide one factual source for README, PDF, and Notion; a small ReportLab builder converts structured portfolio content into a stable PDF. Learning documents under `docs/learning/` reuse the same code locations and evidence while adding daily recall, explanation, experiment, and implementation exercises.

**Tech Stack:** Markdown, Python 3.11+, OpenCV, Pandas, Matplotlib, ReportLab, pdfplumber, pypdf, Poppler, Streamlit, Notion connector, pytest, ruff.

## Global Constraints

- Write the portfolio as if the author can explain every implemented decision.
- Ground every technical and numerical claim in repository code, tests, configuration, or generated reports.
- Do not claim mastery of all OpenCV APIs, classification accuracy from a single-image heuristic, or official MVTec anomaly-detection performance.
- Keep the PDF at exactly six pages or fewer, with one primary message per page.
- Keep the README scannable in three to five minutes.
- Preserve the distinction between heuristic single-image suitability and dataset classification performance.
- Do not commit MVTec source images, local absolute paths, tokens, temporary reports, or restricted dataset assets.
- Use only redistributable project-generated diagrams and synthetic image comparisons in committed portfolio assets.
- Complete documentation before the four-week learning track, as selected by the user.
- Maintain identical dataset counts, metrics, terminology, and limitations across README, PDF, and Notion.

---

## File Structure

```text
docs/
├─ portfolio/
│  ├─ evidence-map.md
│  ├─ case-study.md
│  ├─ experiment-results.md
│  ├─ limitations.md
│  ├─ notion-case-study.md
│  └─ assets/
│     ├─ architecture.png
│     ├─ workflow.png
│     ├─ synthetic-advice-comparison.png
│     ├─ mvtec-tile-best-confusion-matrix.png
│     └─ portfolio-cover.png
├─ learning/
│  ├─ README.md
│  ├─ week-01-image-foundations.md
│  ├─ week-02-preprocessing-diagnostics.md
│  ├─ week-03-features-classifiers-evaluation.md
│  ├─ week-04-explanation-reimplementation.md
│  ├─ exercises.md
│  ├─ interview-qa.md
│  └─ progress-checklist.md
└─ images/
   └─ mvtec-tile-best-confusion-matrix.png
output/
└─ pdf/
   └─ opencv-preprocessing-advisor-portfolio.pdf
scripts/
├─ build_portfolio_assets.py
├─ build_portfolio_pdf.py
└─ validate_portfolio.py
tests/
├─ test_portfolio_assets.py
├─ test_portfolio_pdf.py
└─ test_portfolio_content.py
README.md
README_EN.md
```

---

### Task 1: Evidence Manifest and Deterministic Portfolio Assets

**Files:**
- Create: `scripts/build_portfolio_assets.py`
- Create: `scripts/validate_portfolio.py`
- Create: `tests/test_portfolio_assets.py`
- Create: `tests/test_portfolio_content.py`
- Create: `docs/portfolio/evidence-map.md`
- Create: `docs/portfolio/assets/architecture.png`
- Create: `docs/portfolio/assets/workflow.png`
- Create: `docs/portfolio/assets/synthetic-advice-comparison.png`
- Copy: `docs/images/mvtec-tile-best-confusion-matrix.png` to `docs/portfolio/assets/mvtec-tile-best-confusion-matrix.png`

**Interfaces:**
- Consumes: `ImageAdvisorService.analyze(image, profile) -> ImageAdviceResult`, repository Python source, test files, YAML pipeline configuration.
- Produces: `build_assets(output_dir: Path) -> dict[str, Path]`, `validate_claims(repo_root: Path) -> list[str]`, and four committed PNG assets used by README and PDF.

- [ ] **Step 1: Write failing asset and evidence tests**

Create tests that require:

```python
def test_build_assets_creates_required_pngs(tmp_path):
    outputs = build_assets(tmp_path)
    assert set(outputs) == {
        "architecture",
        "workflow",
        "synthetic_advice_comparison",
    }
    for path in outputs.values():
        assert path.read_bytes().startswith(b"\x89PNG")


def test_claim_validator_accepts_current_repository():
    assert validate_claims(PROJECT_ROOT) == []
```

The claim validator must confirm:

- all OpenCV evidence-map source paths exist;
- README metrics use `117`, `6`, `0.804`, and `0.789`;
- the package declares `opencv-python>=4.10,<5`;
- at least fourteen `tests/test_*.py` files exist;
- `app.py` and all five `ui/*.py` pages exist.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
pytest tests/test_portfolio_assets.py tests/test_portfolio_content.py -v
```

Expected: collection failure because the two scripts do not exist.

- [ ] **Step 3: Implement deterministic asset generation**

Implement `build_portfolio_assets.py` with Matplotlib `Agg` backend:

- architecture diagram: Streamlit/CLI → application services → diagnostics/pipelines/features/evaluation → reports;
- workflow diagram: image input → diagnosis → candidate execution → Top 3 → optional dataset cross-validation;
- synthetic advice comparison: create a low-contrast synthetic industrial-style tile with OpenCV, run `ImageAdvisorService`, and show original plus three recommendations with score labels;
- use Noto Sans CJK or Malgun Gothic when available and fall back to DejaVu Sans;
- export 1600-pixel-wide PNGs at 160 DPI;
- never load MVTec source images.

Implement `validate_portfolio.py` with explicit constants:

```python
EXPECTED_METRICS = {
    "sample_count": "117",
    "class_count": "6",
    "accuracy": "0.804",
    "macro_f1": "0.789",
}
```

Return human-readable validation errors rather than raising on the first mismatch.

- [ ] **Step 4: Write the evidence map**

Create a table with columns:

```text
Area | OpenCV technique | Why it is used | Source | Test | Portfolio explanation
```

Cover image representation, color spaces, enhancement, filtering, gradients, morphology, thresholding, contours/components, diagnostics, HOG, histogram, Gabor, SIFT, SVM, kNN, RTrees, cross-validation, metrics, reporting, and Streamlit integration.

- [ ] **Step 5: Generate and inspect assets**

Run:

```powershell
python scripts/build_portfolio_assets.py --output docs/portfolio/assets
```

Expected: three generated PNG paths and no warnings about missing fonts or images.

Open each PNG and verify:

- no clipped labels;
- Korean text is rendered, not squares;
- arrows do not overlap nodes;
- comparison titles and scores are readable.

- [ ] **Step 6: Run tests and commit**

Run:

```powershell
pytest tests/test_portfolio_assets.py tests/test_portfolio_content.py -v
ruff check scripts tests
ruff format --check scripts tests
```

Expected: PASS.

Commit:

```powershell
git add scripts tests docs/portfolio
git commit -m "docs: add verified OpenCV portfolio evidence"
```

---

### Task 2: Canonical Portfolio Source Documents

**Files:**
- Create: `docs/portfolio/case-study.md`
- Create: `docs/portfolio/experiment-results.md`
- Create: `docs/portfolio/limitations.md`
- Modify: `tests/test_portfolio_content.py`

**Interfaces:**
- Consumes: `docs/portfolio/evidence-map.md`, existing reports documentation, benchmark metrics, source code.
- Produces: the canonical prose and tables used by README, PDF, and Notion.

- [ ] **Step 1: Extend content tests**

Require each canonical document to exist and contain:

```python
REQUIRED_CASE_STUDY_HEADINGS = {
    "문제 정의",
    "설계 판단",
    "단일 이미지 추천",
    "데이터셋 검증",
    "핵심 결과",
    "실패 분석",
}
```

Require `experiment-results.md` to contain the three pipeline rows and the exact evaluation protocol. Require `limitations.md` to contain heuristic, dataset-specific, GT, and classical-feature limitations.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
pytest tests/test_portfolio_content.py -v
```

Expected: FAIL because the canonical documents do not exist.

- [ ] **Step 3: Write `case-study.md`**

Use confident first-person technical reasoning. Explain:

- why recommendation and evaluation are separated;
- why LAB L-channel CLAHE protects color better than channel-wise BGR equalization;
- why different filters serve different noise assumptions;
- how diagnostics and score components support transparent recommendations;
- why fixed OpenCV features and `cv2.ml` classifiers keep the project OpenCV-centered;
- how fold-local scaling prevents leakage;
- why the original pipeline winning is a useful engineering conclusion.

- [ ] **Step 4: Write experiment and limitation documents**

Document:

- dataset interpretation and counts;
- seed and folds;
- feature composition;
- classifier comparison;
- leaderboard and confusion-matrix reading;
- observed failure modes;
- hypotheses clearly labeled as hypotheses;
- no official MVTec anomaly metric claim.

- [ ] **Step 5: Verify and commit**

Run:

```powershell
pytest tests/test_portfolio_content.py -v
python scripts/validate_portfolio.py
```

Expected: PASS and `Portfolio claims validated`.

Commit:

```powershell
git add docs/portfolio tests/test_portfolio_content.py
git commit -m "docs: write canonical OpenCV case study"
```

---

### Task 3: Recruiter-Ready GitHub README

**Files:**
- Modify: `README.md`
- Modify: `README_EN.md`
- Modify: `tests/test_portfolio_content.py`

**Interfaces:**
- Consumes: canonical portfolio documents and committed assets.
- Produces: a three-to-five-minute Korean README and concise English companion.

- [ ] **Step 1: Add README contract tests**

Require:

- architecture, workflow, synthetic comparison, and confusion matrix image references;
- a compact OpenCV evidence table with source links;
- explicit “heuristic is not accuracy” language;
- MVTec interpretation and result table;
- tests, quick start, limitations, and the stable PDF artifact path.

- [ ] **Step 2: Run test and verify RED**

Run:

```powershell
pytest tests/test_portfolio_content.py -v
```

Expected: FAIL because the current README lacks the new visual and evidence contracts.

- [ ] **Step 3: Rewrite Korean README**

Use this high-level order:

```text
Hero → Impact metrics → Problem/solution → Workflow visual →
OpenCV evidence → Architecture → Recommendation example →
Dataset result/failure insight → Reproducibility/tests →
Quick start → Limitations → PDF/Notion links
```

Keep implementation details behind links to canonical documents.

- [ ] **Step 4: Align English README**

Mirror the same claims and metrics with shorter prose. Do not introduce facts absent from the Korean README.

- [ ] **Step 5: Verify rendering and commit**

Run:

```powershell
pytest tests/test_portfolio_content.py -v
python scripts/validate_portfolio.py
```

Inspect Markdown image paths and local anchors.

Commit:

```powershell
git add README.md README_EN.md tests/test_portfolio_content.py
git commit -m "docs: reshape README as OpenCV portfolio"
```

---

### Task 4: Six-Page PDF Portfolio

**Files:**
- Create: `scripts/build_portfolio_pdf.py`
- Create: `tests/test_portfolio_pdf.py`
- Create: `output/pdf/opencv-preprocessing-advisor-portfolio.pdf`
- Create temporarily: `tmp/pdfs/rendered/page-*.png`

**Interfaces:**
- Consumes: canonical portfolio documents and `docs/portfolio/assets/*.png`.
- Produces: `build_pdf(output_path: Path, assets_dir: Path) -> Path`, a static six-page PDF.

- [ ] **Step 1: Write failing PDF tests**

Require:

```python
reader = PdfReader(output)
assert len(reader.pages) == 6
assert output.stat().st_size > 100_000
```

Use `pdfplumber` to extract all text and assert:

- `OpenCV Preprocessing Advisor`
- `Macro F1 0.789`
- `117 images`
- `휴리스틱`
- `전처리가 항상 성능을 높이지 않는다`
- GitHub repository URL

- [ ] **Step 2: Run test and verify RED**

Run:

```powershell
pytest tests/test_portfolio_pdf.py -v
```

Expected: collection failure because the PDF builder is missing.

- [ ] **Step 3: Implement ReportLab PDF builder**

Build exactly six pages:

1. project and impact;
2. problem and solution;
3. OpenCV capability matrix;
4. architecture and recommendation;
5. benchmark and failure insight;
6. evidence, limitations, and links.

Requirements:

- landscape A4;
- embedded Korean TrueType font;
- consistent dark navy, cyan, and warm-orange palette;
- page number and short footer;
- vector headings and tables;
- raster images placed without stretching;
- no paragraph smaller than 9 pt;
- no page uses more than 70% text area.

- [ ] **Step 4: Generate and structurally test PDF**

Run:

```powershell
python scripts/build_portfolio_pdf.py `
  --assets docs/portfolio/assets `
  --output output/pdf/opencv-preprocessing-advisor-portfolio.pdf
pytest tests/test_portfolio_pdf.py -v
```

Expected: six pages and all text assertions pass.

- [ ] **Step 5: Render all PDF pages**

Use bundled `pdftoppm`:

```powershell
New-Item -ItemType Directory -Force tmp/pdfs/rendered | Out-Null
pdftoppm -png -r 150 `
  output/pdf/opencv-preprocessing-advisor-portfolio.pdf `
  tmp/pdfs/rendered/page
```

Expected: six PNG files.

- [ ] **Step 6: Inspect and iterate**

Inspect every rendered page. Fix and re-render until:

- no clipped or overlapping text;
- no missing Korean glyphs;
- tables fit their cells;
- body text is readable at normal zoom;
- image captions are aligned;
- page hierarchy is consistent;
- the final insight on page 5 is visually dominant.

- [ ] **Step 7: Commit final PDF**

Run:

```powershell
pytest tests/test_portfolio_pdf.py -v
python scripts/validate_portfolio.py
```

Commit:

```powershell
git add scripts/build_portfolio_pdf.py tests/test_portfolio_pdf.py `
  output/pdf/opencv-preprocessing-advisor-portfolio.pdf
git commit -m "docs: create six-page OpenCV portfolio"
```

Do not commit `tmp/pdfs/`.

---

### Task 5: Detailed Notion Case Study

**Files:**
- Create: `docs/portfolio/notion-case-study.md`
- Modify after page creation: `README.md`
- Modify after page creation: `README_EN.md`
- Modify: `tests/test_portfolio_content.py`
- External create: one Notion documentation page.

**Interfaces:**
- Consumes: canonical case study, evidence map, experiment results, limitations, GitHub URLs.
- Produces: a complete local Notion source and one Notion page URL.

- [ ] **Step 1: Write local Notion source test**

Require the twelve headings from the design and at least fifteen GitHub source links. Require the same four core benchmark numbers.

- [ ] **Step 2: Write `notion-case-study.md`**

Structure:

1. project summary;
2. background and requirement evolution;
3. recommendation/evaluation separation;
4. image diagnosis;
5. preprocessing selection;
6. features;
7. classifiers;
8. leakage and reproducibility;
9. MVTec experiment;
10. failure interpretation;
11. troubleshooting;
12. limitations and next experiments.

Use callouts for:

- heuristic score disclaimer;
- MVTec official metric disclaimer;
- original pipeline winning insight.

- [ ] **Step 3: Locate Notion destination**

Use `Notion:search` with the literal query `OpenCV Preprocessing Advisor`. If an existing project page exists, fetch it and update it; otherwise locate the primary documentation/wiki parent. If multiple writable parents are equally plausible, use the workspace root rather than inventing a database schema.

- [ ] **Step 4: Create or update the Notion page**

Title:

```text
OpenCV Preprocessing Advisor - Explainable Preprocessing Portfolio
```

Include:

- summary;
- table of contents;
- GitHub repository link;
- PDF path/link note;
- all twelve sections;
- code links to `main`;
- metric and limitation callouts.

- [ ] **Step 5: Add the real Notion URL to README**

Replace the temporary marker in Korean and English READMEs with the actual page URL.

- [ ] **Step 6: Verify and commit**

Fetch the created page and confirm the title, top-level headings, GitHub link, and metrics. Run:

```powershell
pytest tests/test_portfolio_content.py -v
python scripts/validate_portfolio.py
```

Commit:

```powershell
git add docs/portfolio/notion-case-study.md README.md README_EN.md `
  tests/test_portfolio_content.py
git commit -m "docs: publish detailed Notion case study"
```

---

### Task 6: Four-Week OpenCV Learning Pack

**Files:**
- Create: `docs/learning/README.md`
- Create: `docs/learning/week-01-image-foundations.md`
- Create: `docs/learning/week-02-preprocessing-diagnostics.md`
- Create: `docs/learning/week-03-features-classifiers-evaluation.md`
- Create: `docs/learning/week-04-explanation-reimplementation.md`
- Create: `docs/learning/exercises.md`
- Create: `docs/learning/interview-qa.md`
- Create: `docs/learning/progress-checklist.md`
- Create: `tests/test_learning_content.py`

**Interfaces:**
- Consumes: repository source paths, evidence map, pipeline configuration, evaluation results.
- Produces: a 28-session, 30-minute-per-day learning curriculum and assessment pack.

- [ ] **Step 1: Write failing learning-content tests**

Require:

- seven daily sessions in every week file;
- every session contains `회상`, `개념`, `코드 연결`, `실습`, `말로 설명`;
- every source path in the learning pack exists;
- at least twenty implementation/experiment exercises;
- at least thirty interview questions with answer keys;
- one progress checkbox per day.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
pytest tests/test_learning_content.py -v
```

Expected: FAIL because learning files are missing.

- [ ] **Step 3: Write Week 1 and Week 2**

Week 1:

- NumPy image arrays;
- dimensions, shape, dtype;
- BGR/Gray;
- HSV;
- LAB;
- Unicode-safe I/O;
- week review and mini implementation.

Week 2:

- brightness/contrast/histogram;
- normalization/gamma;
- CLAHE;
- Gaussian/Median/Bilateral;
- Sobel/Scharr/Laplacian/Canny;
- morphology/threshold/contours/components;
- diagnostics and week review.

Each day must fit:

```text
5 min recall + 10 min concept/code + 10 min experiment + 5 min oral explanation
```

- [ ] **Step 4: Write Week 3 and Week 4**

Week 3:

- color histogram;
- HOG;
- Sobel/Laplacian/Gabor texture;
- SIFT and BoW concept;
- SVM;
- kNN/RTrees;
- cross-validation, Macro F1, confusion matrix, leakage.

Week 4:

- explain project in one minute;
- explain project in five minutes;
- rebuild a preprocessing pipeline;
- add one diagnostic;
- compare two parameter settings;
- diagnose a failure case;
- mock technical interview and final self-assessment.

- [ ] **Step 5: Write exercises, interview Q&A, and checklist**

Exercises must progress from guided to blank-page implementation. Interview answers must explain why, when to avoid, parameter effects, and project-specific evidence.

- [ ] **Step 6: Verify and commit**

Run:

```powershell
pytest tests/test_learning_content.py -v
python scripts/validate_portfolio.py
```

Commit:

```powershell
git add docs/learning tests/test_learning_content.py
git commit -m "docs: add four-week OpenCV learning pack"
```

---

### Task 7: Public-Release Audit and Final Quality Gate

**Files:**
- Create: `docs/PUBLIC_RELEASE_CHECKLIST.md`
- Modify: `.gitignore`
- Modify if required: `README.md`
- Modify if required: `README_EN.md`
- Modify: `scripts/validate_portfolio.py`
- Test: complete repository.

**Interfaces:**
- Consumes: all final artifacts and Git history.
- Produces: a documented public-release decision and a clean, verified feature branch.

- [ ] **Step 1: Add public-safety validation**

Scan tracked text files for:

- `C:\Users\`
- `.env`
- `gho_`
- `github_pat_`
- `sk-`
- local MVTec dataset paths;
- temporary PDF paths;
- unresolved temporary Notion-link markers.

Exclude intentional examples in `TROUBLESHOOTING.md` only when they contain no real user path.

- [ ] **Step 2: Write release checklist**

Include:

- dataset redistribution;
- secrets;
- personal paths;
- generated outputs;
- license;
- repository topics and description;
- installation on a clean machine;
- README/PDF/Notion consistency;
- image/link rendering;
- final test status.

- [ ] **Step 3: Run complete verification**

Run:

```powershell
pytest -q
ruff check .
ruff format --check .
python scripts/build_portfolio_assets.py --output docs/portfolio/assets
python scripts/build_portfolio_pdf.py `
  --assets docs/portfolio/assets `
  --output output/pdf/opencv-preprocessing-advisor-portfolio.pdf
python scripts/validate_portfolio.py
python -m opencv_preprocessing_advisor.cli self-check
git diff --check
git status --short
```

Expected:

- all tests pass;
- lint and formatting pass;
- assets and PDF regenerate deterministically;
- no safety or consistency errors;
- self-check prints `SELF-CHECK PASSED`;
- working tree contains only intentional regenerated artifacts or is clean after commit.

- [ ] **Step 4: Render PDF one final time**

Render all six pages and visually inspect every page after the final regeneration.

- [ ] **Step 5: Commit final release preparation**

```powershell
git add .gitignore README.md README_EN.md docs scripts tests
git commit -m "docs: prepare OpenCV portfolio for public release"
```

- [ ] **Step 6: Request code and documentation review**

Review the full range from `origin/main` to `HEAD` for:

- factual accuracy;
- broken links;
- public-safety issues;
- PDF visual defects;
- contradictions;
- learning-pack completeness.

Fix all Critical and Important findings, rerun the complete verification, and commit the fixes.
