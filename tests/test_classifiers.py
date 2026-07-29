import numpy as np
import pytest

from opencv_preprocessing_advisor.classifiers import (
    OpenCvKnn,
    OpenCvRTrees,
    OpenCvSvm,
    Standardizer,
)


@pytest.mark.parametrize("factory", [OpenCvSvm, OpenCvKnn, OpenCvRTrees])
def test_classifier_learns_separable_points(factory):
    class_zero = np.array([[x, x * 0.5] for x in range(8)], np.float32)
    class_one = np.array([[20 + x, 20 + x * 0.5] for x in range(8)], np.float32)
    features = np.vstack((class_zero, class_one))
    labels = np.array([0] * 8 + [1] * 8, np.int32)
    model = factory(seed=42)

    model.fit(features, labels)
    predictions = model.predict(features)

    assert np.mean(predictions == labels) >= 0.9


def test_standardizer_uses_training_statistics_only():
    train = np.array([[0.0], [2.0]], np.float32)
    test = np.array([[100.0]], np.float32)

    scaler = Standardizer().fit(train)

    assert scaler.mean_[0] == pytest.approx(1.0)
    assert scaler.transform(test)[0, 0] > 50


def test_predict_before_fit_fails():
    with pytest.raises(RuntimeError, match="fit"):
        OpenCvSvm().predict(np.zeros((1, 2), np.float32))
