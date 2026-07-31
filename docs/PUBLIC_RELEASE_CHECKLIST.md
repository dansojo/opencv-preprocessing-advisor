# Public Release Checklist

## Current release decision

**Public release is blocked.** The repository artifacts have a local release-audit gate, but the verified Notion case-study page at https://app.notion.com/p/3aed0dc3cc1d81c0977fd982867f94e1 is currently private. Do not describe it as public, change its sharing setting, or publish it without explicit owner approval. After approval, verify anonymous access to the exact URL before removing this block.

## Repository and artifact checks

- [ ] Confirm that no MVTec source images, ground-truth masks, or restricted dataset files are committed; publish only project-generated diagrams and synthetic comparisons.
- [ ] Confirm that no secrets, user-home paths, environment files, access tokens, local dataset locations, or temporary render outputs are tracked. `python scripts/validate_portfolio.py` is the local automated gate.
- [ ] Confirm that generated assets and the six-page PDF are rebuilt from the committed source, then review the rendered pages for clipped text, missing glyphs, and unreadable links.
- [ ] Confirm the repository license is present, correct for the intended distribution, and compatible with all included assets and dependencies.
- [ ] Set and review the GitHub repository description and topics before changing repository visibility or announcing the project.

## Public-facing consistency checks

- [ ] Check that Korean and English README metrics, limitations, images, and local links agree with the canonical portfolio sources.
- [ ] Check that the PDF contains the same project title, 117-image count, Macro F1 0.789 result, and repository URL as the README and Notion source.
- [ ] Check that the local Notion source has the same four benchmark values and limitation language as the README and PDF.
- [ ] Check all README, portfolio, and learning-pack Markdown links and local anchors with `python scripts/validate_portfolio.py`; inspect external destinations manually.
- [ ] Confirm image rendering on GitHub after the repository is public; the local audit can verify paths but cannot prove remote hosting or browser rendering.

## Release execution checks

- [ ] Install on a clean machine from the README quick-start instructions, then run `pytest -q` and `python -m opencv_preprocessing_advisor.cli self-check`.
- [ ] Rebuild portfolio assets and PDF, run the complete command gate, and record the final command output in the release/PR discussion.
- [ ] Obtain explicit owner approval before enabling public Notion sharing. Verify anonymous access afterward and update the README wording only after that verification.
- [ ] Review the GitHub repository visibility, description, topics, and license one final time before announcing the project.

## Required local command gate

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

The final release decision remains blocked until every applicable checkbox is complete and the private Notion gate has explicit owner approval plus anonymous-access verification.
