# Notion OpenCV Learning Hub Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the verified Notion portfolio into a deep 10-day OpenCV learning hub with separate technical Q&A, interview, exercises, and progress pages, backed by tested Markdown sources in the repository.

**Architecture:** Repository Markdown remains the canonical source. A content-contract suite verifies structure, depth, counts, links, metrics, and safety before Notion publication. The existing Notion portfolio page becomes the hub, with one course index, ten Day pages, and four independent reference pages; a committed URL map records the external page topology and allows README/hub consistency checks.

**Tech Stack:** Markdown, Python 3.11, pytest, existing portfolio validator, Notion enhanced Markdown/MCP, Git/GitHub links.

## Global Constraints

- There is no daily time limit; completion is based on understanding, implementation, interpretation, and explanation criteria.
- Create exactly ten deep Day pages and keep technical Q&A and interview questions outside the ten-day schedule.
- Every Day page contains the ten shared sections from the approved design.
- Provide at least 50 technical Q&A entries, 35 interview questions, and 30 exercises.
- Use current repository code, configuration, tests, benchmark evidence, synthetic assets, and verified metrics only.
- Preserve the distinction between single-image heuristic recommendations and labeled-dataset performance evaluation.
- Scope MVTec results to 117 images, six `tile/test` status-folder classes, stratified five-fold CV, seed 42, Accuracy 0.804, and Macro F1 0.789; never present them as official anomaly-detection metrics.
- Do not upload MVTec source images, GT masks, local absolute paths, credentials, or private identifiers.
- Existing 28-day learning files stay in the repository; the new 10-day course becomes the default Notion learning route.
- Notion content must use native `<table_of_contents/>`, `<callout>`, and `<table>` syntax where relevant, with tab-indented children.
- Every external Notion page must link back to the main portfolio hub and its related course/reference pages.
- Do not change Notion sharing permissions or GitHub repository visibility without explicit owner approval.

---

### Task 1: Ten-Day Learning Content Contract and Course Index

**Files:**
- Create: `docs/learning-10day/README.md`
- Create: `tests/test_learning_10day_content.py`
- Modify: `scripts/validate_portfolio.py`

**Interfaces:**
- Consumes: `docs/superpowers/specs/2026-07-31-notion-learning-hub-redesign.md`, existing `docs/learning/`, repository code paths.
- Produces: `LEARNING_10DAY_DIR`, required filename/headings/count contracts, and the ordered Day 1–10 course index used by every later task.

- [ ] **Step 1: Write failing structure tests**

Add tests that require the canonical directory, exact files, ordered Day links, and the ten shared Day headings:

```python
LEARNING_10DAY = PROJECT_ROOT / "docs" / "learning-10day"
DAY_FILES = tuple(f"day-{day:02d}.md" for day in range(1, 11))
REFERENCE_FILES = (
    "technical-qa.md",
    "interview-qa.md",
    "exercises.md",
    "progress-checklist.md",
)
DAY_SECTIONS = (
    "오늘 답해야 할 핵심 질문",
    "개념과 원리",
    "OpenCV API와 파라미터",
    "언제 사용하고 피하는가",
    "프로젝트 코드 연결",
    "직접 실험",
    "예상 결과와 해석",
    "자주 하는 실수와 디버깅",
    "본인 말로 설명하기",
    "완료 기준",
)

def test_ten_day_learning_hub_has_exact_topology():
    assert (LEARNING_10DAY / "README.md").is_file()
    assert all((LEARNING_10DAY / name).is_file() for name in (*DAY_FILES, *REFERENCE_FILES))
```

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_learning_10day_content.py -v`

Expected: FAIL because `docs/learning-10day/` and required files do not exist.

- [ ] **Step 3: Write the course index**

Create `README.md` with:

- purpose and learning outcome;
- prerequisites and use of `data/samples/synthetic-tile.png`;
- ordered Day 1–10 links and dependency narrative;
- separate links for technical Q&A, interview, exercises, and checklist;
- explanation that the previous 28-day pack remains available but this is the deep Notion route;
- links to README, PDF, case study, evidence map, benchmark evidence, and Streamlit screenshot.

Create minimal files containing only title plus the ten required section headings for Days and titles for reference files so Task 1's topology contract turns green. Later tasks replace the minimal content completely.

- [ ] **Step 4: Extend the portfolio validator**

Add a `validate_learning_hub(root: Path) -> list[str]` check called by `validate_claims()` that reports missing canonical files, missing Day headings, invalid internal links, forbidden local paths, and use of official-MVTec claims.

- [ ] **Step 5: Run GREEN and full regression**

Run:

```powershell
pytest tests/test_learning_10day_content.py -v
pytest -q
python scripts/validate_portfolio.py
ruff check scripts tests
ruff format --check scripts tests
```

Expected: focused and full suites pass; validator prints `Portfolio validation passed.`

- [ ] **Step 6: Commit**

```powershell
git add docs/learning-10day tests/test_learning_10day_content.py scripts/validate_portfolio.py
git commit -m "docs: scaffold ten-day OpenCV learning hub"
```

---

### Task 2: Days 1–5 Deep OpenCV Learning Pages

**Files:**
- Modify: `docs/learning-10day/day-01.md`
- Modify: `docs/learning-10day/day-02.md`
- Modify: `docs/learning-10day/day-03.md`
- Modify: `docs/learning-10day/day-04.md`
- Modify: `docs/learning-10day/day-05.md`
- Modify: `tests/test_learning_10day_content.py`

**Interfaces:**
- Consumes: Task 1 Day template, `io.py`, `diagnostics.py`, `transforms.py`, `ui/technique_explorer.py`, synthetic sample.
- Produces: complete foundational image, diagnostic, contrast, filtering, edge/threshold/morphology curriculum pages.

- [ ] **Step 1: Add failing depth and evidence tests**

Require for Days 1–5:

```python
def test_days_one_to_five_are_deep_and_traceable():
    required_code = {
        1: ("io.py", "transforms.py", "BGR", "LAB"),
        2: ("diagnostics.py", "entropy", "sharpness", "noise"),
        3: ("clipLimit", "tileGridSize", "LAB L-channel", "gamma"),
        4: ("Gaussian", "Median", "Bilateral", "oversmoothing"),
        5: ("Sobel", "Scharr", "Canny", "connected components"),
    }
    for day, terms in required_code.items():
        text = (LEARNING_10DAY / f"day-{day:02d}.md").read_text(encoding="utf-8")
        assert len(text) >= 6000
        assert all(term in text for term in terms)
        assert all(f"## {section}" in text for section in DAY_SECTIONS)
```

Also require at least three GitHub/local code links, one runnable code block, one expected-results table, four completion checkboxes, and no paragraph that claims a visual improvement guarantees classifier improvement.

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_learning_10day_content.py -v`

Expected: FAIL because scaffold pages lack depth and required evidence.

- [ ] **Step 3: Write Day 1–5 pages**

Follow the approved detailed scope exactly. Each page must:

- explain concepts in Korean before showing APIs;
- include formulas in readable Markdown where useful;
- show minimal runnable Python/OpenCV examples using the committed synthetic sample;
- link to implementation and tests on relative repository paths and GitHub `main` where Notion needs absolute links;
- state expected numeric/visual observations without fabricating exact outputs;
- provide a one-minute and a deeper explanation script;
- include four completion checks: understanding, implementation, interpretation, explanation.

Day-specific non-negotiable details:

- Day 1: dtype ranges, overflow/clipping, BGR/RGB, HSV/LAB, interpolation, Unicode I/O.
- Day 2: every project diagnostic and why no single metric is quality.
- Day 3: normalize/gamma/global HE/CLAHE, correct `tileGridSize` semantics, LAB L-channel rationale.
- Day 4: noise-model assumptions, filter parameters, texture loss, timing and oversmoothing.
- Day 5: derivative operators, full Canny stages, threshold variants, morphology, contour vs components.

- [ ] **Step 4: Run focused and full verification**

Run:

```powershell
pytest tests/test_learning_10day_content.py -v
pytest -q
python scripts/validate_portfolio.py
```

Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add docs/learning-10day/day-0[1-5].md tests/test_learning_10day_content.py
git commit -m "docs: write OpenCV foundations learning days"
```

---

### Task 3: Days 6–10 Project Mastery Pages

**Files:**
- Modify: `docs/learning-10day/day-06.md`
- Modify: `docs/learning-10day/day-07.md`
- Modify: `docs/learning-10day/day-08.md`
- Modify: `docs/learning-10day/day-09.md`
- Modify: `docs/learning-10day/day-10.md`
- Modify: `tests/test_learning_10day_content.py`

**Interfaces:**
- Consumes: Task 1 template, project scoring/pipeline/features/classifier/evaluation/report code, canonical benchmark evidence.
- Produces: complete pipeline, feature, classifier, evaluation, and project-explanation curriculum pages.

- [ ] **Step 1: Add failing technical-coverage tests**

Require at least 6000 characters per page, all common sections, runnable examples, code/test links, completion checks, and:

```python
required_terms = {
    6: ("YAML", "heuristic", "Top 3", "clipping", "oversmoothing"),
    7: ("HSV/LAB histogram", "HOG", "Gabor", "SIFT", "fold-local vocabulary"),
    8: ("cv2.ml", "SVM", "kNN", "RTrees", "float32"),
    9: ("stratified", "fold-local scaling", "Macro F1", "confusion matrix", "leakage"),
    10: ("5분", "15분", "0.804", "0.789", "not official"),
}
```

Day 10 must contain a complete five-minute script, a fifteen-minute outline, at least ten follow-up questions, and an explicit limitations/next-experiment section.

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_learning_10day_content.py -v`

Expected: FAIL because Days 6–10 are scaffolds.

- [ ] **Step 3: Write Day 6–10 pages**

Use the approved scope. Preserve these truth boundaries:

- heuristic score is experiment priority, not accuracy;
- `SiftBowExtractor` exists but is not exposed as a current benchmark profile;
- SIFT vocabulary must be fitted inside each training fold before fair comparison;
- scaling is fit only on training-fold features;
- the reported MVTec example is six-class status-folder classification, not official anomaly detection;
- Original + RTrees winning is an engineering conclusion, not a failed project.

Every page includes project source, test, configuration, and report links plus an experiment that can run without private MVTec data where possible.

- [ ] **Step 4: Verify**

Run:

```powershell
pytest tests/test_learning_10day_content.py -v
pytest -q
python scripts/validate_portfolio.py
ruff check tests
ruff format --check tests
```

Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add docs/learning-10day/day-0[6-9].md docs/learning-10day/day-10.md tests/test_learning_10day_content.py
git commit -m "docs: write OpenCV project mastery days"
```

---

### Task 4: Independent Q&A, Interview, Exercises, and Progress References

**Files:**
- Modify: `docs/learning-10day/technical-qa.md`
- Modify: `docs/learning-10day/interview-qa.md`
- Modify: `docs/learning-10day/exercises.md`
- Modify: `docs/learning-10day/progress-checklist.md`
- Modify: `tests/test_learning_10day_content.py`

**Interfaces:**
- Consumes: Days 1–10, existing 28-day material, repository evidence.
- Produces: at least 50 technical Q&A, 35 interview responses, 30 graded exercises, and a four-level mastery tracker outside the course schedule.

- [ ] **Step 1: Add failing count, format, and adjacency tests**

Require:

```python
TECH_Q = re.compile(r"(?m)^## TQ(\d+): .+$")
INTERVIEW_Q = re.compile(r"(?m)^## IQ(\d+): .+$")
EXERCISE = re.compile(r"(?m)^## EX(\d+): .+$")

assert len(TECH_Q.findall(technical_text)) >= 50
assert len(INTERVIEW_Q.findall(interview_text)) >= 35
assert len(EXERCISE.findall(exercise_text)) >= 30
```

Enforce sequential numbering and block formats:

- technical: `### 한 문장 답`, `### 상세 설명`, `### 프로젝트 근거`, `### 주의할 오해`;
- interview: `### 30초 답변`, `### 2분 심화 답변`, `### 근거 코드·결과`, `### 추가 질문`;
- exercise: `### 선수 지식`, `### 문제`, `### 입력`, `### 요구 산출물`, `### 힌트`, `### 해설`, `### 평가 기준`.

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_learning_10day_content.py -v`

Expected: FAIL on counts and formats.

- [ ] **Step 3: Write 50+ technical Q&A**

Distribute across image/color, diagnostics, contrast, filtering, edges/morphology, features, classifiers, evaluation/reproducibility, architecture/limitations. Answers must be explanatory rather than one-line definitions and must cite current project evidence.

- [ ] **Step 4: Write 35+ interview questions**

Cover project purpose, requirement evolution, technical choices/alternatives, failed preprocessing, original pipeline result, evaluation/leakage, architecture/tests, industrial-image transfer, limitations, prioritization, and next experiments.

- [ ] **Step 5: Write 30+ exercises and full solutions**

Create at least ten Guided, ten Analytical, and ten Reimplementation exercises. Use synthetic sample or user-provided local images. Never require committed MVTec data.

- [ ] **Step 6: Write mastery checklist**

Provide Day 1–10 completion, concept-level four-stage mastery, links/evidence fields, presentation rehearsal, and relearning backlog.

- [ ] **Step 7: Verify and commit**

Run:

```powershell
pytest tests/test_learning_10day_content.py -v
pytest -q
python scripts/validate_portfolio.py
```

Commit:

```powershell
git add docs/learning-10day tests/test_learning_10day_content.py
git commit -m "docs: add OpenCV learning references and practice"
```

---

### Task 5: Notion Page Publication and URL Topology

**Files:**
- Create: `docs/portfolio/notion-learning-hub-map.json`
- Modify: `docs/portfolio/notion-case-study.md`
- Modify: `README.md`
- Modify: `README_EN.md`
- Modify: `tests/test_portfolio_content.py`
- Modify: `scripts/validate_portfolio.py`
- External create/update: existing Notion portfolio page and fifteen learning/reference pages.

**Interfaces:**
- Consumes: all `docs/learning-10day/*.md`, existing Notion portfolio page `3aed0dc3-cc1d-81c0-977f-d982867f94e1`.
- Produces: verified Notion child-page topology and stable page URLs recorded without workspace-private metadata.

- [ ] **Step 1: Add failing URL-map and hub tests**

Require a JSON object with these keys:

```json
{
  "hub": "https://app.notion.com/p/...",
  "course_index": "https://app.notion.com/p/...",
  "days": {"1": "https://app.notion.com/p/...", "10": "https://app.notion.com/p/..."},
  "technical_qa": "https://app.notion.com/p/...",
  "interview_qa": "https://app.notion.com/p/...",
  "exercises": "https://app.notion.com/p/...",
  "progress_checklist": "https://app.notion.com/p/..."
}
```

Tests require ten day URLs, five hub/reference URLs, no duplicate URLs, no pending markers, and README links to the course index.

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_portfolio_content.py -v`

Expected: FAIL because the map and learning hub links do not exist.

- [ ] **Step 3: Read Notion enhanced Markdown specification and fetch the existing hub**

Use Notion `fetch` for `notion://docs/enhanced-markdown-spec` and page `3aed0dc3-cc1d-81c0-977f-d982867f94e1`. Preserve its verified portfolio content and parent relation.

- [ ] **Step 4: Create course and reference pages**

Create the course index and four reference pages under the existing portfolio hub. Then create Day 1–10 under the course index when parent nesting is supported; otherwise create them under the hub and link them from the course index. Use titles from the design and content from canonical Markdown with local links converted to GitHub `main` URLs.

- [ ] **Step 5: Cross-link and update the main hub**

Append/update a native Notion `학습 허브` section that links to the course index and four references. Each created page links back to the main hub; the course index links all ten Days in order.

- [ ] **Step 6: Fetch and verify every external page**

Verify exact title, required top-level headings, key metrics where applicable, GitHub links, hub backlink, and absence of local paths/secrets. Write only page titles and URLs to `notion-learning-hub-map.json`.

- [ ] **Step 7: Update repository links and commit**

Update Korean/English README and local Notion source to describe the portfolio + learning hub and link the course index.

Run:

```powershell
pytest tests/test_portfolio_content.py tests/test_learning_10day_content.py -v
pytest -q
python scripts/validate_portfolio.py
```

Commit:

```powershell
git add docs/portfolio README.md README_EN.md tests scripts/validate_portfolio.py
git commit -m "docs: publish Notion OpenCV learning hub"
```

---

### Task 6: Final Learning-Hub Quality Gate

**Files:**
- Modify: `docs/PUBLIC_RELEASE_CHECKLIST.md`
- Test: complete repository and fetched Notion topology.

**Interfaces:**
- Consumes: Tasks 1–5 and all external page URLs.
- Produces: a clean, evidence-backed branch and a verified private Notion learning hub ready for owner-controlled sharing.

- [ ] **Step 1: Audit spec coverage**

Check every requirement in `2026-07-31-notion-learning-hub-redesign.md` against canonical files, tests, and fetched Notion pages. Record completion and known external visibility constraints in the release checklist.

- [ ] **Step 2: Run the complete repository gate**

Run:

```powershell
pytest -q
ruff check .
ruff format --check .
python scripts/build_portfolio_assets.py --output docs/portfolio/assets
python scripts/build_portfolio_pdf.py --assets docs/portfolio/assets --output output/pdf/opencv-preprocessing-advisor-portfolio.pdf
python scripts/validate_portfolio.py
python -m opencv_preprocessing_advisor.cli self-check
git diff --check
git status --short
```

Expected: all commands succeed and only intentional regenerated artifacts remain before the final commit.

- [ ] **Step 3: Re-fetch and inspect the Notion topology**

Fetch all URLs in `notion-learning-hub-map.json`. Confirm 15 learning/reference pages plus the hub, correct titles, Day ordering, backlink presence, minimum counts, and no truncated content.

- [ ] **Step 4: Commit final audit adjustments**

```powershell
git add docs/PUBLIC_RELEASE_CHECKLIST.md docs tests scripts README.md README_EN.md
git commit -m "docs: complete Notion learning hub quality gate"
```

- [ ] **Step 5: Request final whole-branch review**

Review from commit `76a18c9` through `HEAD` for technical truth, pedagogical depth, count contracts, broken links, Notion consistency, safety, and contradictions. Fix every Critical and Important finding and repeat the full gate.

---

## Execution Decision

The user explicitly approved continuous unattended execution and requested no further questions. Use **Subagent-Driven Development** with a fresh implementer and reviewer per task, continue through all six tasks, and stop only for an external authentication failure that cannot be resolved with the connected Notion workspace.
