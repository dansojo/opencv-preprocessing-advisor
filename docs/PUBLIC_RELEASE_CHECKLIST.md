# Public Release Checklist

## Current delivery decision

**The requested private deliverable is complete and locally release-ready.** The GitHub repository and the verified Notion portfolio/learning hub at https://app.notion.com/p/3aed0dc3cc1d81c0977fd982867f94e1 remain private by design. This audit did not change repository visibility, Notion sharing, or any publication setting.

**Public release is not authorized.** The remaining publication boundary is owner-controlled: obtain explicit approval, change the intended sharing settings, and verify anonymous access to the exact URLs before describing either surface as public.

## Final quality-gate evidence — 2026-07-31

- [x] `pytest -q` — 151 tests passed.
- [x] `ruff check .` — all checks passed.
- [x] `ruff format --check .` — 48 files already formatted.
- [x] `python scripts/build_portfolio_assets.py --output docs/portfolio/assets` — all four deterministic assets rebuilt.
- [x] `python scripts/build_portfolio_pdf.py --assets docs/portfolio/assets --output output/pdf/opencv-preprocessing-advisor-portfolio.pdf` — six-page PDF rebuilt.
- [x] `python scripts/validate_portfolio.py` — `Portfolio validation passed.`
- [x] `python -m opencv_preprocessing_advisor.cli self-check` — `SELF-CHECK PASSED`.
- [x] `git diff --check` — no whitespace errors.
- [x] Regeneration produced no unintended asset or PDF diff.
- [x] The rebuilt PDF was rendered to six PNG pages at 120 DPI and visually inspected: no clipping, overlap, missing Korean glyphs, unreadable table text, or broken page numbering was found.

## Repository and content audit

- [x] No MVTec source image, GT mask, local dataset directory, environment file, credential, token-shaped secret, user-home path, or temporary PDF render is tracked.
- [x] README, README_EN, the local Notion source, benchmark evidence, and PDF agree on 117 images, six classes, stratified 5-fold, seed 42, Accuracy 0.804, and Macro F1 0.789.
- [x] The documents state that the result is status-folder classification and not an official MVTec anomaly-detection metric.
- [x] The heuristic Advisor score remains explicitly separated from classifier accuracy and generalization performance.
- [x] Markdown local links, anchors, GitHub `main` blob targets, portfolio claims, and private-Notion wording pass `scripts/validate_portfolio.py`.
- [x] The bundled Noto Sans KR font retains its SIL Open Font License at `docs/portfolio/fonts/OFL.txt`.
- [ ] A repository-wide source-code license has not been selected. This is not required for the approved private deliverable; the owner must choose one before any public distribution.

## Notion topology audit

- [x] Re-fetched the private portfolio hub plus all 15 learning/reference pages from `docs/portfolio/notion-learning-hub-map.json` (16 pages total).
- [x] Exact title properties match the approved portfolio, course, Day 1–10, technical Q&A, interview Q&A, exercises, and progress-checklist titles.
- [x] The course index is a child of the hub; Day 1–10 are children of the course; the four reference pages are children of the hub.
- [x] The course links Day 1–10 in order, and every learning/reference page links back to the private portfolio hub.
- [x] Every Day page contains the ten required sections in order and returns a complete closing page payload.
- [x] Fetched reference counts are exactly 50 sequential technical Q&A, 35 sequential interview Q&A, and 30 sequential exercises.
- [x] Every technical Q&A contains `한 문장 답`, `상세 설명`, `프로젝트 근거`, and `주의할 오해`; every exercise contains `선수 지식`, `문제`, `입력`, `요구 산출물`, `힌트`, `해설`, and `평가 기준`.
- [x] No fetched page contains a Windows absolute path, `file://` URL, credential-shaped token, source MVTec image asset/path, or GT-mask asset/path; truthful scope disclaimers may mention those terms.
- [x] The fetched hub preserves the scoped MVTec metrics and the explicit non-official-anomaly-metric warning.

## Owner-controlled public-release actions

- [ ] Obtain explicit owner approval before changing the private Notion hub or GitHub repository visibility.
- [ ] Select a repository-wide source-code license and review GitHub description/topics for the intended public audience.
- [ ] After sharing, verify the hub, course, ten Days, and four references in an anonymous browser session; do not infer public access from an authenticated Notion fetch.
- [ ] Re-run the full command gate on the exact public-release commit and confirm GitHub-rendered images and links.

Until those owner-controlled steps are performed, describe the result as a **verified private portfolio and learning hub**, not as a public site.
