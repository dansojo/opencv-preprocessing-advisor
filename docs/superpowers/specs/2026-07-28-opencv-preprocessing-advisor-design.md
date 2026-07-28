# OpenCV Preprocessing Advisor — Final Project Design

## 1. Project identity

- **Display name:** OpenCV Preprocessing Advisor
- **Repository/folder name:** `opencv-preprocessing-advisor`
- **One-line description:** Explainable, measurable OpenCV preprocessing recommendations for image-classification datasets.
- **Primary audience:** Recruiters and computer-vision team leads evaluating practical OpenCV competence.

The project is not an anomaly detector, a deep-learning model, or an image-quality “beautifier.” It is a reproducible laboratory that diagnoses image characteristics, recommends three transparent preprocessing pipelines, applies them, and measures their effects. When a labeled classification dataset is supplied, it also benchmarks whether each pipeline improves classical OpenCV classification performance.

## 2. Portfolio success criterion

A reviewer should be able to conclude:

> The author understands OpenCV image representation, color spaces, filtering, contrast enhancement, gradients, morphology, feature extraction, classical ML, evaluation, and the trade-offs between them—and can explain why a technique was selected and what changed after it was applied.

The repository must provide evidence through:

1. explainable technique selection rather than a list of API calls;
2. before/after numerical measurements;
3. reproducible dataset benchmarks;
4. intermediate images and visual comparisons;
5. failure cases and limitations;
6. modular code, tests, CLI commands, and a Streamlit interface.

## 3. Final scope

### Included

1. Single-image diagnostic and recommendation mode.
2. Labeled classification-dataset benchmark mode.
3. Technique Explorer for manual OpenCV experimentation.
4. Three ranked preprocessing pipeline recommendations.
5. OpenCV-based image diagnostics and feature extraction.
6. OpenCV `kNN`, `SVM`, and `RTrees` classifiers.
7. Stratified cross-validation and transparent metrics.
8. Streamlit visualization and an `argparse` CLI.
9. CSV/JSON/PNG report export.
10. Korean primary README and concise English README.
11. Automated tests with synthetic images, so development does not wait for user data.

### Excluded

1. Video processing.
2. MVTec-specific ground-truth masks or anomaly detection.
3. Deep learning and external vision-model evaluation.
4. LLM/API-based image scoring.
5. Geometric augmentation as a primary feature.
6. Claims that a single-image heuristic score predicts classification accuracy.
7. Deployment to a public server in the first implementation.

## 4. User modes

### 4.1 Single Image Advisor

Input:

- one JPG, JPEG, PNG, BMP, or TIFF image;
- optional analysis profile:
  - Auto / general classification;
  - shape-focused classification;
  - color-focused classification;
  - texture-focused classification.

Processing:

1. Validate and decode the image.
2. Measure diagnostic characteristics.
3. Apply every compatible pipeline in the catalog.
4. Re-measure the result of every pipeline.
5. Calculate a transparent profile-specific heuristic score.
6. Return the top three pipelines.

Output:

- original-image diagnostics;
- top-three pipeline cards;
- step-by-step intermediate images;
- before/after values and percentage changes;
- score breakdown, recommendation reasons, and warnings;
- processing time;
- CSV, JSON, and result-image downloads.

The heuristic score is explicitly labeled **preprocessing suitability**, not accuracy, probability, or defect confidence.

### 4.2 Classification Dataset Benchmark

Input folder:

```text
dataset/
├─ class_a/
│  ├─ image_001.png
│  └─ ...
├─ class_b/
└─ class_c/
```

Validation:

- at least two classes;
- at least five decodable images per class;
- ten or more images per class recommended;
- unreadable files are reported and skipped;
- class imbalance is displayed.

Processing:

1. Run baseline images and each preprocessing pipeline.
2. Extract fixed OpenCV feature profiles.
3. Train and evaluate OpenCV classifiers with deterministic stratified five-fold cross-validation.
4. Fit scaling and learned feature vocabularies only on each training fold.
5. Rank pipelines by mean macro F1.
6. Break ties by accuracy, then preprocessing time.

Output:

- mean and standard deviation for accuracy, macro precision, macro recall, and macro F1;
- per-class precision/recall/F1;
- confusion matrix;
- preprocessing, feature-extraction, training, and inference time;
- classifier and feature-profile comparison;
- top-three evidence-based dataset recommendations;
- downloadable detailed and summary reports.

### 4.3 Technique Explorer

The explorer demonstrates individual OpenCV concepts without claiming that every operation belongs in the recommendation catalog.

Categories:

- color conversion: BGR, RGB, Gray, HSV, LAB;
- histogram: per-channel histograms and equalization;
- contrast: CLAHE and gamma correction;
- denoising: Gaussian, median, and bilateral filters;
- detail: unsharp mask, Sobel, Scharr, Laplacian, and Canny;
- texture: Gabor responses and morphology;
- segmentation study: global, Otsu, and adaptive threshold;
- region analysis: contours and connected components.

Every operation shows its OpenCV function, parameters, output, and a short “use when / avoid when” explanation.

## 5. Numerical diagnostics

The single-image engine calculates:

| Metric | Interpretation |
|---|---|
| mean brightness | overall exposure on a 0–255 scale |
| dark/bright clipping ratio | lost detail near 0 and 255 |
| global contrast | grayscale standard deviation |
| local contrast | average local standard deviation |
| entropy | intensity-distribution information |
| sharpness | variance of Laplacian |
| noise estimate | robust high-frequency residual estimate |
| illumination uniformity | low-frequency background variation |
| edge density | Canny edge-pixel ratio |
| edge continuity | proportion of edges belonging to nontrivial connected components |
| colorfulness | opponent-channel color spread |
| saturation spread | HSV saturation standard deviation |
| processing time | milliseconds per image |

Before/after reports contain raw values, absolute deltas, and percentage deltas where the denominator is valid.

## 6. Recommendation design

### 6.1 Catalog

The first version contains explainable recipes rather than arbitrary parameter search:

1. `baseline-normalize` — resize and safe dtype normalization.
2. `lab-clahe` — LAB luminance CLAHE for local contrast.
3. `auto-gamma` — histogram-derived gamma correction.
4. `gaussian-clean` — light Gaussian denoising.
5. `median-clean` — impulse-noise reduction.
6. `bilateral-detail` — edge-preserving smoothing.
7. `clahe-bilateral` — local contrast plus edge-preserving denoising.
8. `unsharp-detail` — controlled detail enhancement.
9. `gray-clahe-unsharp` — shape-focused grayscale enhancement.
10. `texture-blackhat` — dark local-structure emphasis.

Pipelines use conservative defaults and expose only valid odd kernel sizes and bounded parameters.

### 6.2 Ranking

The engine applies all compatible recipes, then scores observed changes. Score components are normalized to 0–100 and published in the README.

- **Auto:** balanced contrast and entropy improvement, detail preservation, clipping/noise penalties.
- **Shape:** local contrast, edge continuity, sharpness, thin-detail preservation, excessive-edge penalty.
- **Color:** color-distribution preservation, saturation separation, luminance balance, clipping penalty.
- **Texture:** local contrast, gradient energy, texture response, noise penalty.

Every recommendation includes:

- component scores and weights;
- rules that triggered it;
- measured gains;
- warnings such as clipping, excessive edge growth, color loss, or oversmoothing.

## 7. OpenCV feature and classifier strategy

### Feature profiles

1. **Color:** normalized HSV and LAB histograms using `cv2.calcHist`.
2. **Shape:** HOG using `cv2.HOGDescriptor`.
3. **Texture:** Sobel/Laplacian/Gabor statistics.
4. **Combined:** concatenated, standardized color + shape + texture features.
5. **Advanced optional:** SIFT Bag of Visual Words using `cv2.SIFT_create` and `cv2.BOWKMeansTrainer`, fit inside each training fold.

### Classifiers

- **Primary:** `cv2.ml.SVM_create`, RBF kernel.
- **Baseline:** `cv2.ml.KNearest_create`.
- **Secondary:** `cv2.ml.RTrees_create`.

All classifiers receive the same fold and feature matrix for a fair comparison. SVM is the default leaderboard because it is a strong classical baseline; kNN reveals local-distance behavior; RTrees provides a nonlinear ensemble comparison.

No scikit-learn dependency is used. NumPy implements stratified fold indices, scaling, confusion matrices, and metric formulas. Pandas is used only for report tables and CSV export.

## 8. UI structure

Streamlit pages:

1. **Overview** — project purpose, workflow, limitations, and quick start.
2. **Image Advisor** — upload, profile selection, diagnostics, top-three pipelines, comparisons, downloads.
3. **Dataset Benchmark** — folder/ZIP input, validation, benchmark controls, leaderboard, confusion matrices, exports.
4. **Technique Explorer** — interactive OpenCV operation laboratory.
5. **Methodology** — formulas, scoring weights, leakage controls, and interpretation guidance.

The UI imports the core package; it contains no duplicate image-processing or scoring logic.

## 9. CLI

```powershell
python -m opencv_preprocessing_advisor.cli analyze-image `
  --image path/to/image.png `
  --profile auto `
  --output outputs/example

python -m opencv_preprocessing_advisor.cli benchmark `
  --dataset data/classification `
  --folds 5 `
  --classifiers svm,knn,rtrees `
  --output outputs/benchmark
```

The CLI and Streamlit interface call identical application services.

## 10. Reports

```text
outputs/
├─ image_advisor/<run_id>/
│  ├─ diagnostics.csv
│  ├─ recommendations.json
│  ├─ comparison.png
│  └─ steps/<pipeline_id>/*.png
└─ benchmark/<run_id>/
   ├─ leaderboard.csv
   ├─ fold_metrics.csv
   ├─ class_metrics.csv
   ├─ timings.csv
   ├─ run_config.json
   └─ confusion_matrices/*.png
```

Each report records the random seed, OpenCV version, parameters, and input manifest for reproducibility.

## 11. Repository documentation

`README.md` is Korean-first and optimized for a two-minute recruiter scan:

1. project outcome and screenshot;
2. problem and non-goals;
3. architecture;
4. top-three recommendation example;
5. numerical evidence;
6. OpenCV technique map;
7. dataset benchmark example;
8. failure cases and trade-offs;
9. quick-start commands;
10. test and repository structure.

`README_EN.md` provides a concise English equivalent. `TROUBLESHOOTING.md` records observed issue, cause, attempted techniques, numerical result, and remaining limitation.

## 12. Testing and quality gates

- TDD for all domain behavior.
- Synthetic images cover darkness, clipping, blur, impulse noise, gradients, color cast, lines, and textures.
- Tests prove that diagnostic directions are correct, for example CLAHE raises local contrast on a low-contrast fixture and median filtering reduces impulse-noise estimates.
- Dataset tests generate tiny labeled shape/color datasets.
- Split tests prove class stratification and no train/test overlap.
- Scaling and BoVW vocabulary tests prove training-fold-only fitting.
- Integration tests exercise one CLI image run and one small benchmark.
- UI smoke tests verify page imports and service calls; numerical correctness remains in the core tests.

Quality gate before release:

```powershell
pytest -q
ruff check .
ruff format --check .
python -m opencv_preprocessing_advisor.cli --help
python -m opencv_preprocessing_advisor.cli self-check --output outputs/self-check
```

## 13. Risks and explicit limitations

1. A single-image score cannot predict classification accuracy.
2. Dataset recommendations are specific to the supplied data, split, features, and classifier.
3. Higher contrast or sharpness can amplify irrelevant patterns and noise.
4. Accuracy alone is misleading on imbalanced datasets; macro F1 is primary.
5. Very small datasets produce unstable rankings; mean and standard deviation must be shown.
6. Preprocessing choices can leak information if fitted globally; all learned transformations are fitted per training fold.
7. The tool recommends experiments, not universally optimal preprocessing.

## 14. Definition of done

The first release is done when:

1. a user can upload one image and receive three explained, numerically supported pipeline recommendations;
2. a user can supply a class-folder dataset and receive reproducible OpenCV classifier comparisons;
3. all five Streamlit pages are functional;
4. CLI and UI results are produced by the same core services;
5. reports are exportable;
6. automated tests and static checks pass;
7. Korean and English README documents make the project understandable without reading the source;
8. at least three successful and three unsuccessful recommendation case studies are documented using user-supplied images or clearly labeled public/sample data.

## 15. Approved decision summary

- General classification preprocessing, not anomaly detection.
- No MVTec GT dependency.
- OpenCV-centered evaluation.
- Streamlit UI.
- Top-three explained pipelines.
- Optional user focus: auto, shape, color, or texture.
- Numerical outputs are mandatory.
- GitHub repository first; a Codex skill may be added only after the repository works.
