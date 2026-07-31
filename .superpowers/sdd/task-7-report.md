# Task 7 Report: Public-Release Audit and Final Quality Gate

## Implementation

- Added `docs/PUBLIC_RELEASE_CHECKLIST.md` with public-release gates for dataset redistribution, secrets, personal paths, generated artifacts, license, GitHub metadata, clean-machine installation, consistency, image/link rendering, and final verification.
- Added tracked-text public-safety validation for real Windows user paths, environment-file references, token-shaped strings, absolute local MVTec paths, temporary PDF render paths, and unresolved Notion markers.
- Added Markdown local-path and anchor validation across README, portfolio, and learning documents. GitHub `main` blob links for this repository are checked against local tracked paths.
- Added README/PDF/Notion consistency checks and explicit private-Notion disclosure requirements.
- Marked `tmp/` ignored and PDFs as Git binary files. The PDF builder now uses ReportLab invariant mode and has a byte-determinism test.
- Changed both READMEs to say that the verified Notion page is private, rather than public.

## Verification

The complete command gate passed after the final rebuild:

```text
pytest -q                                      114 passed
ruff check .                                   passed
ruff format --check .                          47 files already formatted
build_portfolio_assets.py                      passed
build_portfolio_pdf.py                         passed
validate_portfolio.py                          passed
opencv_preprocessing_advisor.cli self-check    SELF-CHECK PASSED
git diff --check                               passed
```

The PDF was rebuilt and all six pages were rendered at 150 DPI under ignored `tmp/pdfs/final-rendered/`. Visual inspection found no clipped or overlapping text, missing Korean glyphs, unreadable table content, or misaligned captions.

## Review follow-up fixes

- The public-safety scan now includes tracked extensionless text files, including a deliberately force-tracked `.env` in its regression test. `.env` is ignored by default, while a tracked file is still scanned for `ghp_` tokens and the existing token markers.
- Markdown validation now resolves full and collapsed reference-style links, GitHub blob URL autolinks, and ignores headings inside fenced code blocks when building anchors.
- The release validator rebuilds the portfolio PDF in a temporary directory and compares it byte-for-byte with the committed artifact. The PDF builder now keys embedded image objects from image data, rather than source paths, so builds are identical across checkout locations. The committed PDF was regenerated from the current sources.
- Removed the terminal blank line from the learning design specification so the committed range passes `git diff --check`.

Focused regressions passed: `5 passed`. The full suite passed: `119 passed`.

### Re-review follow-up

- Extended Markdown reference validation to resolve shortcut references such as `[label]` and reference-style images such as `![alt][label]` through their definitions.
- Focused validator coverage passed: `3 passed`; Ruff check and format checks passed for the changed files.

## Public-release decision

**Blocked pending explicit owner action.** The Notion URL `https://app.notion.com/p/3aed0dc3cc1d81c0977fd982867f94e1` is content-verified but private. This task did not change Notion permissions, GitHub visibility, or publish anything. Public release requires explicit owner approval to share the page and a subsequent anonymous-access verification.
