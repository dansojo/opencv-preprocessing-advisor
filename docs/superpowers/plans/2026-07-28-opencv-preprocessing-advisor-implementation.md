# OpenCV Preprocessing Advisor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an explainable OpenCV application that recommends three preprocessing pipelines for a single image and benchmarks pipelines on labeled classification datasets with OpenCV features and classifiers.

**Architecture:** A framework-independent Python package owns diagnostics, transformations, recommendation rules, feature extraction, OpenCV classifiers, evaluation, and reporting. Thin CLI and Streamlit adapters call application services from that package. Single-image recommendations use transparent heuristic scores; dataset recommendations use leakage-safe stratified cross-validation ranked by macro F1.

**Tech Stack:** Python 3.11+, `opencv-python`, NumPy, Pandas, Matplotlib, Streamlit, PyYAML, pytest, ruff.

## Global Constraints

- Do not add deep learning, external vision APIs, MVTec-specific logic, video processing, or scikit-learn.
- Use OpenCV for image processing, feature extraction, and classifiers; use NumPy for metric formulas and fold generation.
- Top-three single-image scores must be labeled heuristic suitability, never accuracy or probability.
- Fit scalers and learned visual vocabularies on training folds only.
- Keep the core package independent of Streamlit.
- Every behavior change follows RED → GREEN → REFACTOR.
- All random behavior uses an explicit default seed of `42`.
- Save reports with parameters, seed, OpenCV version, and input manifest.

---

## Planned repository map

```text
opencv-preprocessing-advisor/
├─ app.py
├─ pyproject.toml
├─ requirements.txt
├─ requirements-dev.txt
├─ README.md
├─ README_EN.md
├─ TROUBLESHOOTING.md
├─ .gitignore
├─ config/
│  ├─ pipelines.yaml
│  └─ scoring.yaml
├─ src/opencv_preprocessing_advisor/
│  ├─ __init__.py
│  ├─ models.py
│  ├─ io.py
│  ├─ diagnostics.py
│  ├─ transforms.py
│  ├─ pipelines.py
│  ├─ scoring.py
│  ├─ features.py
│  ├─ datasets.py
│  ├─ classifiers.py
│  ├─ evaluation.py
│  ├─ reports.py
│  ├─ services.py
│  └─ cli.py
├─ ui/
│  ├─ overview.py
│  ├─ image_advisor.py
│  ├─ dataset_benchmark.py
│  ├─ technique_explorer.py
│  └─ methodology.py
├─ tests/
│  ├─ conftest.py
│  ├─ test_io.py
│  ├─ test_diagnostics.py
│  ├─ test_transforms.py
│  ├─ test_pipelines.py
│  ├─ test_scoring.py
│  ├─ test_features.py
│  ├─ test_datasets.py
│  ├─ test_classifiers.py
│  ├─ test_evaluation.py
│  ├─ test_reports.py
│  ├─ test_services.py
│  └─ test_cli.py
├─ data/
│  ├─ samples/.gitkeep
│  └─ classification/.gitkeep
├─ outputs/.gitkeep
└─ docs/
   ├─ images/.gitkeep
   └─ superpowers/
```

---

### Task 1: Reproducible package and test foundation

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `requirements-dev.txt`
- Create: `.gitignore`
- Create: `src/opencv_preprocessing_advisor/__init__.py`
- Create: `tests/test_package.py`

**Interfaces:**
- Produces: importable package `opencv_preprocessing_advisor`
- Produces: `opencv_preprocessing_advisor.__version__: str`

- [ ] **Step 1: Write the failing package test**

```python
from opencv_preprocessing_advisor import __version__


def test_package_exposes_version():
    assert __version__ == "0.1.0"
```

- [ ] **Step 2: Run the test and confirm the package is missing**

Run:

```powershell
python -m pytest tests/test_package.py -v
```

Expected: collection fails with `ModuleNotFoundError: No module named 'opencv_preprocessing_advisor'`.

- [ ] **Step 3: Add the minimal package metadata**

`pyproject.toml` must contain:

```toml
[build-system]
requires = ["setuptools>=75", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "opencv-preprocessing-advisor"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "opencv-python>=4.10",
  "numpy>=2.0",
  "pandas>=2.2",
  "matplotlib>=3.9",
  "streamlit>=1.45",
  "PyYAML>=6.0",
]

[project.scripts]
opencv-prep = "opencv_preprocessing_advisor.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"
```

`src/opencv_preprocessing_advisor/__init__.py`:

```python
__version__ = "0.1.0"
```

`requirements.txt` mirrors runtime dependencies; `requirements-dev.txt` contains `-r requirements.txt`, `pytest>=8.3`, and `ruff>=0.11`.

`.gitignore` must ignore `.venv/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `*.pyc`, `outputs/*`, `data/classification/*`, and uploaded images while preserving `.gitkeep` files.

- [ ] **Step 4: Install editable dependencies and pass the test**

Run:

```powershell
python -m pip install -e . -r requirements-dev.txt
python -m pytest tests/test_package.py -v
```

Expected: one passing test.

- [ ] **Step 5: Commit the foundation**

```powershell
git add pyproject.toml requirements.txt requirements-dev.txt .gitignore src tests
git commit -m "chore: initialize preprocessing advisor package"
```

---

### Task 2: Domain models and safe image I/O

**Files:**
- Create: `src/opencv_preprocessing_advisor/models.py`
- Create: `src/opencv_preprocessing_advisor/io.py`
- Create: `tests/test_io.py`

**Interfaces:**
- Produces: `TaskProfile(str, Enum)` with `AUTO`, `SHAPE`, `COLOR`, `TEXTURE`
- Produces: `ImageDiagnostics`, `MetricChange`, `PipelineRun`, `Recommendation` frozen dataclasses
- Produces: `decode_image(path: Path) -> np.ndarray`
- Produces: `encode_png(image: np.ndarray) -> bytes`
- Produces: `validate_bgr_image(image: np.ndarray) -> None`

- [ ] **Step 1: Write failing I/O tests**

```python
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_preprocessing_advisor.io import decode_image, encode_png, validate_bgr_image


def test_unicode_path_round_trip(tmp_path: Path):
    image = np.full((16, 20, 3), 127, np.uint8)
    path = tmp_path / "표면_이미지.png"
    path.write_bytes(encode_png(image))
    loaded = decode_image(path)
    assert loaded.shape == image.shape
    assert np.array_equal(loaded, image)


def test_validate_rejects_float_image():
    with pytest.raises(ValueError, match="uint8"):
        validate_bgr_image(np.zeros((8, 8, 3), np.float32))
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
pytest tests/test_io.py -v
```

Expected: import failure because `io.py` does not exist.

- [ ] **Step 3: Implement models and Unicode-safe OpenCV I/O**

Use `np.fromfile` + `cv2.imdecode` for reading and `cv2.imencode(".png", image)` for encoding. `validate_bgr_image` must require a nonempty `uint8` array with shape `(H, W, 3)`.

The model definitions must include:

```python
class TaskProfile(str, Enum):
    AUTO = "auto"
    SHAPE = "shape"
    COLOR = "color"
    TEXTURE = "texture"


@dataclass(frozen=True)
class ImageDiagnostics:
    mean_brightness: float
    dark_clip_ratio: float
    bright_clip_ratio: float
    global_contrast: float
    local_contrast: float
    entropy: float
    sharpness: float
    noise_estimate: float
    illumination_nonuniformity: float
    edge_density: float
    edge_continuity: float
    colorfulness: float
    saturation_spread: float
```

Add frozen dataclasses for metric deltas, pipeline steps/runs, score components, and recommendations. Images remain NumPy arrays; serialization is handled by reports.

- [ ] **Step 4: Verify GREEN**

Run:

```powershell
pytest tests/test_io.py -v
```

Expected: both tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/opencv_preprocessing_advisor/models.py src/opencv_preprocessing_advisor/io.py tests/test_io.py
git commit -m "feat: add domain models and safe image IO"
```

---

### Task 3: Measurable image diagnostics

**Files:**
- Create: `src/opencv_preprocessing_advisor/diagnostics.py`
- Create: `tests/conftest.py`
- Create: `tests/test_diagnostics.py`

**Interfaces:**
- Consumes: validated BGR `uint8` images
- Produces: `analyze_image(image: np.ndarray) -> ImageDiagnostics`
- Produces: `compare_diagnostics(before, after) -> dict[str, MetricChange]`

- [ ] **Step 1: Create synthetic fixtures and failing directional tests**

```python
import cv2
import numpy as np

from opencv_preprocessing_advisor.diagnostics import analyze_image


def test_checkerboard_has_more_contrast_than_flat_image():
    flat = np.full((128, 128, 3), 120, np.uint8)
    board = np.indices((128, 128)).sum(axis=0) % 2
    checker = cv2.cvtColor((board * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    assert analyze_image(checker).global_contrast > analyze_image(flat).global_contrast


def test_blurred_edges_have_lower_sharpness():
    image = np.zeros((128, 128, 3), np.uint8)
    cv2.rectangle(image, (32, 32), (96, 96), (255, 255, 255), -1)
    blurred = cv2.GaussianBlur(image, (15, 15), 0)
    assert analyze_image(blurred).sharpness < analyze_image(image).sharpness


def test_impulse_noise_increases_noise_estimate():
    clean = np.full((128, 128, 3), 120, np.uint8)
    noisy = clean.copy()
    noisy[::4, ::4] = 255
    noisy[2::4, 2::4] = 0
    assert analyze_image(noisy).noise_estimate > analyze_image(clean).noise_estimate
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
pytest tests/test_diagnostics.py -v
```

Expected: import failure for `diagnostics`.

- [ ] **Step 3: Implement every documented metric**

Required formulas:

- grayscale with `cv2.cvtColor`;
- entropy from a 256-bin normalized `cv2.calcHist`;
- local contrast from mean standard deviation over a 16×16 grid;
- sharpness from `cv2.Laplacian(gray, cv2.CV_64F).var()`;
- noise estimate from median absolute deviation of a Laplacian residual;
- illumination nonuniformity from standard deviation of a large Gaussian background divided by its mean;
- edge density from Canny nonzero ratio;
- edge continuity from `cv2.connectedComponentsWithStats`, counting edge pixels in components of at least eight pixels;
- colorfulness from opponent-channel means and variances;
- saturation spread from HSV.

All ratios must be finite and clipped to meaningful ranges. Constant images must not cause division by zero.

- [ ] **Step 4: Pass diagnostics tests and add constant-image edge cases**

Run:

```powershell
pytest tests/test_diagnostics.py -v
```

Expected: all tests pass with no warnings.

- [ ] **Step 5: Commit**

```powershell
git add src/opencv_preprocessing_advisor/diagnostics.py tests
git commit -m "feat: measure explainable image diagnostics"
```

---

### Task 4: OpenCV transforms and pipeline catalog

**Files:**
- Create: `src/opencv_preprocessing_advisor/transforms.py`
- Create: `src/opencv_preprocessing_advisor/pipelines.py`
- Create: `config/pipelines.yaml`
- Create: `tests/test_transforms.py`
- Create: `tests/test_pipelines.py`

**Interfaces:**
- Produces: pure transform functions accepting/returning BGR `uint8`
- Produces: `PipelineCatalog.from_yaml(path) -> PipelineCatalog`
- Produces: `PipelineCatalog.run(pipeline_id, image) -> PipelineRun`
- Produces: intermediate step images for every pipeline

- [ ] **Step 1: Write failing transform tests**

```python
import numpy as np

from opencv_preprocessing_advisor.transforms import (
    apply_lab_clahe,
    apply_median,
    apply_unsharp,
)


def test_lab_clahe_preserves_bgr_shape_and_dtype(low_contrast_bgr):
    result = apply_lab_clahe(low_contrast_bgr, clip_limit=2.0, grid_size=8)
    assert result.shape == low_contrast_bgr.shape
    assert result.dtype == np.uint8


def test_median_filter_reduces_impulse_pixels(impulse_noise_bgr):
    result = apply_median(impulse_noise_bgr, kernel_size=5)
    before = np.count_nonzero((impulse_noise_bgr == 0) | (impulse_noise_bgr == 255))
    after = np.count_nonzero((result == 0) | (result == 255))
    assert after < before


def test_unsharp_rejects_even_kernel(sample_bgr):
    with pytest.raises(ValueError, match="odd"):
        apply_unsharp(sample_bgr, kernel_size=4, amount=1.0)
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
pytest tests/test_transforms.py tests/test_pipelines.py -v
```

Expected: missing-module failures.

- [ ] **Step 3: Implement focused transforms**

Implement:

```python
apply_lab_clahe(image, clip_limit, grid_size)
apply_auto_gamma(image, target_midpoint)
apply_gaussian(image, kernel_size, sigma)
apply_median(image, kernel_size)
apply_bilateral(image, diameter, sigma_color, sigma_space)
apply_unsharp(image, kernel_size, sigma, amount, threshold)
apply_gray_bgr(image)
apply_blackhat(image, kernel_size, shape)
normalize_uint8(image)
```

All kernels validate positive odd dimensions. CLAHE modifies LAB luminance only. Auto gamma clamps gamma to `[0.5, 2.0]`. Unsharp uses `cv2.addWeighted` and clips safely.

- [ ] **Step 4: Add the ten explicit YAML recipes**

Each recipe specifies `id`, Korean and English display names, compatible profiles, ordered transforms, parameters, rationale keys, and warning keys. Reject unknown transform names and invalid parameters during catalog load.

Pipeline tests must assert:

- ten unique IDs load;
- every run preserves dimensions and `uint8`;
- each step is retained in order;
- the same image and config produce identical output;
- an invalid even kernel fails before processing.

- [ ] **Step 5: Verify GREEN**

Run:

```powershell
pytest tests/test_transforms.py tests/test_pipelines.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add config src/opencv_preprocessing_advisor/transforms.py src/opencv_preprocessing_advisor/pipelines.py tests
git commit -m "feat: add validated preprocessing pipeline catalog"
```

---

### Task 5: Transparent scoring and top-three recommendation

**Files:**
- Create: `src/opencv_preprocessing_advisor/scoring.py`
- Create: `config/scoring.yaml`
- Create: `tests/test_scoring.py`

**Interfaces:**
- Consumes: before/after `ImageDiagnostics`, `TaskProfile`, timing
- Produces: `score_pipeline(...) -> ScoreBreakdown`
- Produces: `rank_recommendations(...) -> list[Recommendation]`

- [ ] **Step 1: Write failing ranking tests**

```python
from opencv_preprocessing_advisor.models import TaskProfile
from opencv_preprocessing_advisor.scoring import rank_recommendations


def test_rank_returns_exactly_three_unique_recommendations(scored_runs):
    ranked = rank_recommendations(scored_runs, TaskProfile.AUTO, limit=3)
    assert len(ranked) == 3
    assert len({item.pipeline_id for item in ranked}) == 3
    assert ranked[0].suitability_score >= ranked[1].suitability_score


def test_excessive_clipping_creates_warning(clipping_run):
    ranked = rank_recommendations([clipping_run], TaskProfile.AUTO, limit=3)
    assert "clipping" in ranked[0].warning_codes


def test_score_is_finite_and_bounded(scored_runs):
    for recommendation in rank_recommendations(scored_runs, TaskProfile.TEXTURE, limit=3):
        assert 0.0 <= recommendation.suitability_score <= 100.0
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
pytest tests/test_scoring.py -v
```

Expected: missing-module failure.

- [ ] **Step 3: Implement published score components**

`config/scoring.yaml` contains weights totaling `1.0` per profile. Implement bounded, signed changes using:

```python
def relative_change(before: float, after: float, epsilon: float = 1e-9) -> float:
    return (after - before) / max(abs(before), epsilon)
```

Map changes through clipped piecewise-linear functions. Keep component values, weights, weighted values, rule codes, and warnings in the output. Apply penalties for clipping growth, excessive edge-density growth, oversmoothing, and color loss.

Tie-break ordering:

1. suitability score descending;
2. warning count ascending;
3. processing time ascending;
4. pipeline ID ascending.

- [ ] **Step 4: Verify GREEN and snapshot formulas**

Run:

```powershell
pytest tests/test_scoring.py -v
```

Expected: all tests pass. Add an exact-score fixture so accidental formula changes are reviewed rather than silently accepted.

- [ ] **Step 5: Commit**

```powershell
git add config/scoring.yaml src/opencv_preprocessing_advisor/scoring.py tests/test_scoring.py
git commit -m "feat: rank preprocessing pipelines transparently"
```

---

### Task 6: Classification dataset discovery and leakage-safe folds

**Files:**
- Create: `src/opencv_preprocessing_advisor/datasets.py`
- Create: `tests/test_datasets.py`

**Interfaces:**
- Produces: `discover_dataset(root: Path) -> DatasetManifest`
- Produces: `stratified_folds(labels, n_splits=5, seed=42) -> list[Fold]`
- Produces: manifest rows containing path, class name, class index, size, checksum

- [ ] **Step 1: Write failing discovery and split tests**

```python
def test_discovers_class_directories_in_sorted_order(classification_dataset):
    manifest = discover_dataset(classification_dataset)
    assert manifest.class_names == ("circle", "square")
    assert len(manifest.samples) == 20


def test_stratified_folds_are_disjoint_and_complete():
    labels = np.array([0] * 10 + [1] * 10)
    folds = stratified_folds(labels, n_splits=5, seed=42)
    seen_test = set()
    for fold in folds:
        assert set(fold.train_indices).isdisjoint(fold.test_indices)
        assert set(labels[fold.test_indices]) == {0, 1}
        seen_test.update(fold.test_indices.tolist())
    assert seen_test == set(range(20))


def test_dataset_rejects_fewer_than_two_classes(one_class_dataset):
    with pytest.raises(ValueError, match="at least two classes"):
        discover_dataset(one_class_dataset)
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
pytest tests/test_datasets.py -v
```

Expected: missing-module failure.

- [ ] **Step 3: Implement deterministic discovery and splitting**

Support `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, and `.tiff`. Decode each candidate once to validate it. Preserve a skipped-file list with reasons. Refuse fewer than five valid samples in any class. If the requested fold count exceeds the smallest class count, reduce folds and record a warning.

Generate fold indices per class with `np.random.default_rng(seed)`, then concatenate and sort. Never use global random state.

- [ ] **Step 4: Verify GREEN**

Run:

```powershell
pytest tests/test_datasets.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/opencv_preprocessing_advisor/datasets.py tests/test_datasets.py
git commit -m "feat: discover labeled datasets and create safe folds"
```

---

### Task 7: OpenCV feature profiles

**Files:**
- Create: `src/opencv_preprocessing_advisor/features.py`
- Create: `tests/test_features.py`

**Interfaces:**
- Produces: `ColorHistogramExtractor`
- Produces: `HOGExtractor`
- Produces: `TextureStatsExtractor`
- Produces: `CombinedExtractor`
- Produces: `SiftBowExtractor.fit(train_images)` and `.transform(images)`
- All transforms return finite `np.float32` matrices shaped `(n_samples, n_features)`

- [ ] **Step 1: Write failing feature tests**

```python
def test_color_histogram_is_fixed_length_and_normalized(sample_images):
    matrix = ColorHistogramExtractor().transform(sample_images)
    assert matrix.shape == (len(sample_images), 96)
    assert matrix.dtype == np.float32
    assert np.allclose(matrix.sum(axis=1), 1.0, atol=1e-5)


def test_hog_is_deterministic(sample_images):
    extractor = HOGExtractor(size=(128, 128))
    assert np.array_equal(extractor.transform(sample_images), extractor.transform(sample_images))


def test_sift_bow_requires_fit(sample_images):
    extractor = SiftBowExtractor(vocabulary_size=8, seed=42)
    with pytest.raises(RuntimeError, match="fit"):
        extractor.transform(sample_images)
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
pytest tests/test_features.py -v
```

Expected: missing-module failure.

- [ ] **Step 3: Implement fixed feature profiles**

- HSV histogram: H=32, S=32; LAB luminance histogram: 32; concatenate and L1-normalize to 96 dimensions.
- HOG: resize to 128×128; block 16×16, stride 8×8, cell 8×8, nine bins.
- Texture: mean/std/percentiles of Sobel magnitude, Laplacian magnitude, and four Gabor orientations.
- Combined: concatenate color, HOG, and texture.

Use only OpenCV and NumPy.

- [ ] **Step 4: Implement training-fold-only SIFT BoVW**

Use `cv2.SIFT_create`, collect training descriptors, cap descriptors per image deterministically, train with `cv2.BOWKMeansTrainer`, and build L1-normalized visual-word histograms. If descriptors are insufficient, return a structured unavailable reason rather than silently producing zeros.

The test must fit on train fixtures and transform separate test fixtures. Add an instrumentation assertion proving test images are not passed to `fit`.

- [ ] **Step 5: Verify GREEN**

Run:

```powershell
pytest tests/test_features.py -v
```

Expected: all feature tests pass.

- [ ] **Step 6: Commit**

```powershell
git add src/opencv_preprocessing_advisor/features.py tests/test_features.py
git commit -m "feat: extract OpenCV classification features"
```

---

### Task 8: OpenCV classifiers and metric formulas

**Files:**
- Create: `src/opencv_preprocessing_advisor/classifiers.py`
- Create: `src/opencv_preprocessing_advisor/evaluation.py`
- Create: `tests/test_classifiers.py`
- Create: `tests/test_evaluation.py`

**Interfaces:**
- Produces: `Standardizer.fit(X_train)` and `.transform(X)`
- Produces: classifier adapters `OpenCvSvm`, `OpenCvKnn`, `OpenCvRTrees`
- Produces: `confusion_matrix`, `classification_metrics`
- Produces: `cross_validate(...) -> BenchmarkResult`

- [ ] **Step 1: Write failing classifier tests**

```python
@pytest.mark.parametrize("factory", [OpenCvSvm, OpenCvKnn, OpenCvRTrees])
def test_classifier_learns_separable_points(factory):
    X = np.array([[0, 0], [0, 1], [10, 10], [10, 11]], np.float32)
    y = np.array([0, 0, 1, 1], np.int32)
    model = factory(seed=42)
    model.fit(X, y)
    assert np.array_equal(model.predict(X), y)


def test_standardizer_uses_training_statistics_only():
    train = np.array([[0.0], [2.0]], np.float32)
    test = np.array([[100.0]], np.float32)
    scaler = Standardizer().fit(train)
    assert scaler.mean_[0] == pytest.approx(1.0)
    assert scaler.transform(test)[0, 0] > 50
```

- [ ] **Step 2: Write failing metric tests**

```python
def test_macro_metrics_match_hand_calculation():
    truth = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 1, 1])
    result = classification_metrics(truth, pred, class_count=2)
    assert result.accuracy == pytest.approx(0.75)
    assert result.macro_recall == pytest.approx(0.75)
    assert result.macro_precision == pytest.approx((1.0 + 2 / 3) / 2)
```

- [ ] **Step 3: Verify RED**

Run:

```powershell
pytest tests/test_classifiers.py tests/test_evaluation.py -v
```

Expected: missing-module failures.

- [ ] **Step 4: Implement adapters and evaluation**

OpenCV configuration:

```python
# SVM
model.setType(cv2.ml.SVM_C_SVC)
model.setKernel(cv2.ml.SVM_RBF)
model.setC(2.0)
model.setGamma(0.01)
model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 1000, 1e-6))

# kNN
model.setDefaultK(5)
model.setIsClassifier(True)

# RTrees
model.setMaxDepth(12)
model.setMinSampleCount(2)
model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 200, 0))
```

Call `cv2.setRNGSeed(seed)` immediately before stochastic OpenCV training. Convert features to `float32` and labels to `int32`. Calculate confusion matrix and zero-division-safe per-class metrics with NumPy.

- [ ] **Step 5: Add cross-validation timing and leakage tests**

Mock/instrument `Standardizer.fit` and learned extractors to assert they see only train indices. Record preprocessing, extraction, fit, and prediction milliseconds separately. Aggregate fold results as mean and sample standard deviation.

- [ ] **Step 6: Verify GREEN**

Run:

```powershell
pytest tests/test_classifiers.py tests/test_evaluation.py -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```powershell
git add src/opencv_preprocessing_advisor/classifiers.py src/opencv_preprocessing_advisor/evaluation.py tests
git commit -m "feat: benchmark OpenCV classifiers safely"
```

---

### Task 9: Application services and report artifacts

**Files:**
- Create: `src/opencv_preprocessing_advisor/services.py`
- Create: `src/opencv_preprocessing_advisor/reports.py`
- Create: `tests/test_services.py`
- Create: `tests/test_reports.py`

**Interfaces:**
- Produces: `ImageAdvisorService.analyze(image, profile) -> ImageAdviceResult`
- Produces: `BenchmarkService.run(manifest, config) -> BenchmarkResult`
- Produces: `ReportWriter.write_image_advice(...) -> Path`
- Produces: `ReportWriter.write_benchmark(...) -> Path`

- [ ] **Step 1: Write a failing end-to-end image-service test**

```python
def test_image_advisor_returns_three_explained_results(service, sample_bgr):
    result = service.analyze(sample_bgr, TaskProfile.AUTO)
    assert len(result.recommendations) == 3
    for item in result.recommendations:
        assert item.reasons
        assert item.score_components
        assert item.pipeline_run.intermediate_images
```

- [ ] **Step 2: Write failing report tests**

```python
def test_image_report_contains_reproducibility_metadata(tmp_path, advice_result):
    output = ReportWriter(tmp_path).write_image_advice(advice_result)
    metadata = json.loads((output / "recommendations.json").read_text(encoding="utf-8"))
    assert metadata["opencv_version"]
    assert metadata["scoring_config_hash"]
    assert (output / "diagnostics.csv").exists()
    assert (output / "comparison.png").exists()
```

- [ ] **Step 3: Verify RED**

Run:

```powershell
pytest tests/test_services.py tests/test_reports.py -v
```

Expected: missing-module failures.

- [ ] **Step 4: Implement services without UI imports**

`ImageAdvisorService` validates, diagnoses, runs the catalog, times pipelines, calculates before/after changes, ranks results, and returns structured models. `BenchmarkService` validates the dataset, runs requested pipeline/feature/classifier combinations, aggregates folds, and ranks by macro F1 → accuracy → speed.

- [ ] **Step 5: Implement atomic report writing**

Write into a temporary sibling directory and rename only after every artifact succeeds. Use Unicode-safe PNG encoding. Serialize arrays as images or lists only where bounded. Add SHA-256 hashes of configs and the dataset manifest.

- [ ] **Step 6: Verify GREEN**

Run:

```powershell
pytest tests/test_services.py tests/test_reports.py -v
```

Expected: all tests pass and temporary output is cleaned.

- [ ] **Step 7: Commit**

```powershell
git add src/opencv_preprocessing_advisor/services.py src/opencv_preprocessing_advisor/reports.py tests
git commit -m "feat: orchestrate advice and reproducible reports"
```

---

### Task 10: CLI and built-in self-check

**Files:**
- Create: `src/opencv_preprocessing_advisor/cli.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- Produces commands: `analyze-image`, `benchmark`, `self-check`
- Exit codes: `0` success, `2` invalid input/configuration, `1` unexpected processing failure

- [ ] **Step 1: Write failing CLI tests**

```python
def test_cli_analyze_image_writes_report(tmp_path, sample_path):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "opencv_preprocessing_advisor.cli",
            "analyze-image",
            "--image",
            str(sample_path),
            "--profile",
            "auto",
            "--output",
            str(tmp_path / "out"),
        ],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert any((tmp_path / "out").glob("*/recommendations.json"))


def test_cli_rejects_missing_dataset(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-m", "opencv_preprocessing_advisor.cli", "benchmark",
         "--dataset", str(tmp_path / "missing")],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 2
    assert "does not exist" in completed.stderr
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
pytest tests/test_cli.py -v
```

Expected: module or command failure.

- [ ] **Step 3: Implement `argparse` subcommands**

Print concise progress and final artifact paths. `self-check` creates synthetic low-contrast, noisy, colored-shape images, runs image advice and a tiny two-class benchmark, and exits nonzero on invariant failure. It accepts `--output`; when supplied, it saves the generated fixtures and reports so documentation can use verified artifacts without external data.

- [ ] **Step 4: Verify GREEN**

Run:

```powershell
pytest tests/test_cli.py -v
python -m opencv_preprocessing_advisor.cli --help
python -m opencv_preprocessing_advisor.cli self-check --output outputs/self-check
```

Expected: tests pass, help lists all three commands, and self-check prints `SELF-CHECK PASSED`.

- [ ] **Step 5: Commit**

```powershell
git add src/opencv_preprocessing_advisor/cli.py tests/test_cli.py
git commit -m "feat: add CLI and synthetic self check"
```

---

### Task 11: Streamlit application

**Files:**
- Create: `app.py`
- Create: `ui/overview.py`
- Create: `ui/image_advisor.py`
- Create: `ui/dataset_benchmark.py`
- Create: `ui/technique_explorer.py`
- Create: `ui/methodology.py`
- Create: `tests/test_ui_imports.py`

**Interfaces:**
- Consumes only `services`, `models`, and report artifacts from the core package
- Produces five navigable Streamlit pages

- [ ] **Step 1: Write failing import-boundary tests**

```python
@pytest.mark.parametrize(
    "module",
    [
        "ui.overview",
        "ui.image_advisor",
        "ui.dataset_benchmark",
        "ui.technique_explorer",
        "ui.methodology",
    ],
)
def test_ui_modules_import_without_starting_processing(module):
    imported = importlib.import_module(module)
    assert callable(imported.render)
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
pytest tests/test_ui_imports.py -v
```

Expected: missing-module failures.

- [ ] **Step 3: Implement navigation and shared visual language**

Use `st.navigation` and `st.Page`. Set a wide layout, Korean primary labels with English technical terms, consistent colors for original/recommended/warning states, and cached resource loading for YAML catalogs.

- [ ] **Step 4: Implement Image Advisor**

Required UI:

- image uploader and profile selector;
- explicit Analyze button inside a form;
- original diagnostics cards;
- top-three recommendation tabs;
- score-component bar chart;
- before/after side-by-side images;
- intermediate-step expander;
- raw/delta metric table;
- reasons, warnings, and suitability disclaimer;
- ZIP/CSV/JSON downloads.

- [ ] **Step 5: Implement Dataset Benchmark**

Accept a ZIP with class folders or a local path in developer mode. Display validation before running. Controls include folds, feature profile, classifier list, and pipeline subset. Show progress, leaderboard with mean±std, confusion matrix, per-class metrics, timings, warnings, and downloads.

- [ ] **Step 6: Implement Technique Explorer and Methodology**

Explorer controls must enforce valid odd kernels and parameter ranges. Each technique shows `cv2` function mapping and “use when / avoid when.” Methodology renders score weights, formulas, classifier settings, leakage controls, and limitations from static structured content.

- [ ] **Step 7: Verify UI**

Run:

```powershell
pytest tests/test_ui_imports.py -v
streamlit run app.py
```

Manually verify:

1. all five pages navigate;
2. reruns do not discard completed analysis inside the session;
3. an invalid upload shows a user-facing error;
4. analysis services execute only after button press;
5. downloads open and contain metadata.

- [ ] **Step 8: Commit**

```powershell
git add app.py ui tests/test_ui_imports.py
git commit -m "feat: add Streamlit preprocessing laboratory"
```

---

### Task 12: Recruiter-facing documentation and final verification

**Files:**
- Create: `README.md`
- Create: `README_EN.md`
- Create: `TROUBLESHOOTING.md`
- Create: `docs/images/.gitkeep`
- Create: `data/samples/.gitkeep`
- Create: `data/classification/.gitkeep`
- Create: `outputs/.gitkeep`
- Modify: documentation after verified screenshots and outputs exist

**Interfaces:**
- Produces: two-minute Korean project narrative and concise English equivalent
- Produces: repeatable troubleshooting case-study format

- [ ] **Step 1: Generate verified sample artifacts**

Run:

```powershell
python -m opencv_preprocessing_advisor.cli self-check --output outputs/self-check
python -m opencv_preprocessing_advisor.cli analyze-image `
  --image outputs/self-check/fixtures/low_contrast.png `
  --profile auto `
  --output outputs/readme-example
```

Expected: self-check passes and a report directory contains three recommendations.

- [ ] **Step 2: Write Korean README around evidence**

The first screen must contain:

- project name and one-line outcome;
- one representative Streamlit screenshot;
- a compact “input → diagnosis → top three → benchmark” flow;
- one numerical before/after result;
- quick-start commands.

Later sections must cover the OpenCV technique matrix, architecture, heuristic-score disclaimer, dataset methodology, leakage prevention, reports, tests, failure cases, limitations, and roadmap.

- [ ] **Step 3: Write concise English README**

Mirror the claims and commands without translating every troubleshooting narrative. Keep function names and metric definitions identical across both documents.

- [ ] **Step 4: Write troubleshooting cases**

Use this mandatory structure:

```markdown
### Case: <observed failure>

- Input characteristic:
- Recommended pipeline:
- Expected effect:
- Measured result:
- Why the recommendation failed or traded off:
- Parameter/technique attempted:
- Remaining limitation:
```

Document at least three successful and three unsuccessful cases before a portfolio release. Until user images arrive, clearly label synthetic cases and do not present them as industrial evidence.

- [ ] **Step 5: Run the complete quality gate**

Run:

```powershell
pytest -q
ruff check .
ruff format --check .
python -m opencv_preprocessing_advisor.cli --help
python -m opencv_preprocessing_advisor.cli self-check
git status --short
```

Expected:

- all tests pass;
- ruff reports no violations or formatting changes;
- CLI help succeeds;
- self-check prints `SELF-CHECK PASSED`;
- `git status --short` shows only intentional documentation/sample artifacts, or is clean after the final commit.

- [ ] **Step 6: Commit the portfolio release candidate**

```powershell
git add README.md README_EN.md TROUBLESHOOTING.md docs data outputs
git commit -m "docs: present OpenCV preprocessing advisor portfolio"
```

---

## Plan self-review record

- **Spec coverage:** single-image recommendations, numerical evidence, top three, profile selection, OpenCV classifiers, dataset validation, Streamlit, CLI, reports, bilingual documentation, and limitations each map to a task.
- **Scope control:** MVTec, anomaly masks, video, deep learning, external model scoring, and deployment are absent.
- **Leakage control:** fold-specific scaling and learned vocabulary are explicit in Tasks 7–8.
- **Type consistency:** images are BGR `uint8`; features are finite `float32`; labels are `int32`; ranking models flow through services into CLI/UI.
- **Placeholder scan:** the implementation contains no deferred production feature. Real portfolio case studies remain an explicit release gate dependent on user-supplied images.

## Execution checkpoints

1. Tasks 1–5: single-image recommendation engine works without UI.
2. Tasks 6–10: labeled-dataset OpenCV benchmark and CLI work.
3. Task 11: Streamlit presents both engines.
4. Task 12: verified artifacts become the recruiter-facing portfolio.
