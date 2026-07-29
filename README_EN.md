# OpenCV Preprocessing Advisor

An explainable OpenCV portfolio project that diagnoses image quality, recommends three preprocessing pipelines, and validates their downstream impact with classical image classification.

## Core idea

The project deliberately separates two questions:

- **Single image:** Which pipelines are worth trying based on measurable image changes?
- **Class-folder dataset:** Which pipeline actually improves cross-validated classification?

The single-image score is a transparent heuristic, not an accuracy estimate. Dataset mode uses stratified folds, OpenCV features (HOG, color histograms, Sobel/Laplacian/Gabor statistics), and `cv2.ml` SVM, kNN, and RTrees.

## MVTec tile case study

The local MVTec AD `tile/test` folders were treated as six classification classes without anomaly masks: 117 images, five folds, seed 42.

| Pipeline | Classifier | Accuracy | Macro F1 |
|---|---|---:|---:|
| Original | RTrees | 0.804 | **0.789** |
| CLAHE + Bilateral | RTrees | 0.766 | 0.731 |
| LAB CLAHE | RTrees | 0.664 | 0.594 |

The baseline won. This is useful evidence that visually stronger contrast is not automatically better for the downstream task.

## Run

```powershell
python -m pip install -e .
streamlit run app.py
python -m opencv_preprocessing_advisor.cli self-check
```

```powershell
opencv-prep analyze-image --image path\to\image.png --profile auto
opencv-prep benchmark --dataset path\to\class-folder-dataset
```

Reports include diagnostic CSV, recommendation JSON, intermediate images, leaderboard, fold and per-class metrics, timing data, reproducibility metadata, and confusion matrices.

See [README.md](README.md) for the full Korean documentation and [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for environment notes.

