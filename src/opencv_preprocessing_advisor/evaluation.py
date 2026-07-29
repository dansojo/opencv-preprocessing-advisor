"""Leakage-safe cross-validation and classification metric formulas."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .classifiers import Standardizer, create_classifier
from .datasets import Fold


@dataclass(frozen=True)
class PerClassMetrics:
    class_index: int
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    confusion_matrix: np.ndarray
    per_class: tuple[PerClassMetrics, ...]


@dataclass(frozen=True)
class FoldEvaluation:
    fold_index: int
    metrics: ClassificationMetrics
    fit_ms: float
    predict_ms: float


@dataclass(frozen=True)
class CrossValidationResult:
    classifier_name: str
    folds: tuple[FoldEvaluation, ...]

    @property
    def mean_accuracy(self) -> float:
        return float(np.mean([fold.metrics.accuracy for fold in self.folds]))

    @property
    def std_accuracy(self) -> float:
        return _sample_std([fold.metrics.accuracy for fold in self.folds])

    @property
    def mean_macro_f1(self) -> float:
        return float(np.mean([fold.metrics.macro_f1 for fold in self.folds]))

    @property
    def std_macro_f1(self) -> float:
        return _sample_std([fold.metrics.macro_f1 for fold in self.folds])


def _sample_std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def confusion_matrix(
    truth: np.ndarray,
    predicted: np.ndarray,
    class_count: int,
) -> np.ndarray:
    truth_array = np.asarray(truth, dtype=np.int32).reshape(-1)
    predicted_array = np.asarray(predicted, dtype=np.int32).reshape(-1)
    if truth_array.size != predicted_array.size:
        raise ValueError("truth and predicted must contain the same number of values")
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    for actual, guess in zip(truth_array, predicted_array, strict=True):
        if not 0 <= actual < class_count or not 0 <= guess < class_count:
            raise ValueError("class index is outside the configured class_count")
        matrix[actual, guess] += 1
    return matrix


def classification_metrics(
    truth: np.ndarray,
    predicted: np.ndarray,
    class_count: int,
) -> ClassificationMetrics:
    matrix = confusion_matrix(truth, predicted, class_count)
    total = int(matrix.sum())
    per_class: list[PerClassMetrics] = []
    for class_index in range(class_count):
        true_positive = int(matrix[class_index, class_index])
        predicted_positive = int(matrix[:, class_index].sum())
        actual_positive = int(matrix[class_index, :].sum())
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / actual_positive if actual_positive else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append(
            PerClassMetrics(
                class_index=class_index,
                precision=float(precision),
                recall=float(recall),
                f1=float(f1),
                support=actual_positive,
            )
        )
    return ClassificationMetrics(
        accuracy=float(np.trace(matrix) / total) if total else 0.0,
        macro_precision=float(np.mean([item.precision for item in per_class])),
        macro_recall=float(np.mean([item.recall for item in per_class])),
        macro_f1=float(np.mean([item.f1 for item in per_class])),
        confusion_matrix=matrix,
        per_class=tuple(per_class),
    )


def cross_validate(
    features: np.ndarray,
    labels: np.ndarray,
    folds: list[Fold],
    classifier_name: str,
    seed: int = 42,
) -> CrossValidationResult:
    matrix = np.asarray(features, dtype=np.float32)
    vector = np.asarray(labels, dtype=np.int32).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != vector.size:
        raise ValueError("features and labels have incompatible shapes")
    class_count = int(np.max(vector)) + 1
    evaluations: list[FoldEvaluation] = []
    for fold_index, fold in enumerate(folds):
        scaler = Standardizer().fit(matrix[fold.train_indices])
        train_features = scaler.transform(matrix[fold.train_indices])
        test_features = scaler.transform(matrix[fold.test_indices])
        classifier = create_classifier(classifier_name, seed + fold_index)
        started = perf_counter()
        classifier.fit(train_features, vector[fold.train_indices])
        fit_ms = (perf_counter() - started) * 1000.0
        started = perf_counter()
        predicted = classifier.predict(test_features)
        predict_ms = (perf_counter() - started) * 1000.0
        metrics = classification_metrics(
            vector[fold.test_indices],
            predicted,
            class_count,
        )
        evaluations.append(
            FoldEvaluation(
                fold_index=fold_index,
                metrics=metrics,
                fit_ms=fit_ms,
                predict_ms=predict_ms,
            )
        )
    return CrossValidationResult(classifier_name=classifier_name, folds=tuple(evaluations))
