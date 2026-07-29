import numpy as np
import pytest

from opencv_preprocessing_advisor.datasets import stratified_folds
from opencv_preprocessing_advisor.evaluation import (
    classification_metrics,
    cross_validate,
)


def test_macro_metrics_match_hand_calculation():
    truth = np.array([0, 0, 1, 1])
    predicted = np.array([0, 1, 1, 1])

    result = classification_metrics(truth, predicted, class_count=2)

    assert result.accuracy == pytest.approx(0.75)
    assert result.macro_recall == pytest.approx(0.75)
    assert result.macro_precision == pytest.approx((1.0 + 2 / 3) / 2)
    assert result.confusion_matrix.tolist() == [[1, 1], [0, 2]]


def test_metrics_handle_zero_division():
    truth = np.array([0, 0, 1, 1])
    predicted = np.array([0, 0, 0, 0])

    result = classification_metrics(truth, predicted, class_count=2)

    assert np.isfinite(result.macro_f1)
    assert result.per_class[1].precision == 0.0


def test_cross_validation_returns_every_fold():
    class_zero = np.array([[x, 0] for x in range(10)], np.float32)
    class_one = np.array([[20 + x, 20] for x in range(10)], np.float32)
    features = np.vstack((class_zero, class_one))
    labels = np.array([0] * 10 + [1] * 10, np.int32)
    folds = stratified_folds(labels, n_splits=5, seed=42)

    result = cross_validate(features, labels, folds, classifier_name="svm", seed=42)

    assert len(result.folds) == 5
    assert result.mean_accuracy >= 0.9
    assert result.mean_macro_f1 >= 0.9
