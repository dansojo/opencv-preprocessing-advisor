"""Thin, consistent adapters around OpenCV classical ML classifiers."""

from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np


def _feature_matrix(features: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("features must be a nonempty two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("features must be finite")
    return matrix


def _label_vector(labels: np.ndarray, row_count: int) -> np.ndarray:
    vector = np.asarray(labels, dtype=np.int32).reshape(-1)
    if vector.size != row_count:
        raise ValueError("labels and features must contain the same number of rows")
    return vector


class Standardizer:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, features: np.ndarray) -> Standardizer:
        matrix = _feature_matrix(features)
        self.mean_ = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        self.scale_ = np.where(scale < 1e-8, 1.0, scale).astype(np.float32)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Standardizer must be fit before transform")
        matrix = _feature_matrix(features)
        return ((matrix - self.mean_) / self.scale_).astype(np.float32)


class Classifier(Protocol):
    def fit(self, features: np.ndarray, labels: np.ndarray) -> Classifier: ...

    def predict(self, features: np.ndarray) -> np.ndarray: ...


class OpenCvSvm:
    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._model = cv2.ml.SVM_create()
        self._fitted = False

    def fit(self, features: np.ndarray, labels: np.ndarray) -> OpenCvSvm:
        matrix = _feature_matrix(features)
        vector = _label_vector(labels, matrix.shape[0])
        cv2.setRNGSeed(self.seed)
        self._model.setType(cv2.ml.SVM_C_SVC)
        self._model.setKernel(cv2.ml.SVM_RBF)
        self._model.setC(2.0)
        self._model.setGamma(0.01)
        self._model.setTermCriteria(
            (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 1000, 1e-6)
        )
        if not self._model.train(matrix, cv2.ml.ROW_SAMPLE, vector):
            raise RuntimeError("OpenCV SVM training failed")
        self._fitted = True
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("OpenCvSvm must be fit before predict")
        _, response = self._model.predict(_feature_matrix(features))
        return response.reshape(-1).astype(np.int32)


class OpenCvKnn:
    def __init__(self, seed: int = 42, neighbors: int = 5) -> None:
        self.seed = seed
        self.neighbors = neighbors
        self._model = cv2.ml.KNearest_create()
        self._fitted = False
        self._training_rows = 0

    def fit(self, features: np.ndarray, labels: np.ndarray) -> OpenCvKnn:
        matrix = _feature_matrix(features)
        vector = _label_vector(labels, matrix.shape[0])
        self._model.setDefaultK(self.neighbors)
        self._model.setIsClassifier(True)
        if not self._model.train(matrix, cv2.ml.ROW_SAMPLE, vector):
            raise RuntimeError("OpenCV kNN training failed")
        self._training_rows = matrix.shape[0]
        self._fitted = True
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("OpenCvKnn must be fit before predict")
        k = min(self.neighbors, self._training_rows)
        _, response, _, _ = self._model.findNearest(_feature_matrix(features), k)
        return response.reshape(-1).astype(np.int32)


class OpenCvRTrees:
    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._model = cv2.ml.RTrees_create()
        self._fitted = False

    def fit(self, features: np.ndarray, labels: np.ndarray) -> OpenCvRTrees:
        matrix = _feature_matrix(features)
        vector = _label_vector(labels, matrix.shape[0])
        cv2.setRNGSeed(self.seed)
        self._model.setMaxDepth(12)
        self._model.setMinSampleCount(2)
        self._model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 200, 0))
        if not self._model.train(matrix, cv2.ml.ROW_SAMPLE, vector):
            raise RuntimeError("OpenCV RTrees training failed")
        self._fitted = True
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("OpenCvRTrees must be fit before predict")
        _, response = self._model.predict(_feature_matrix(features))
        return response.reshape(-1).astype(np.int32)


def create_classifier(name: str, seed: int = 42) -> Classifier:
    factories = {
        "svm": OpenCvSvm,
        "knn": OpenCvKnn,
        "rtrees": OpenCvRTrees,
    }
    try:
        return factories[name](seed=seed)
    except KeyError as error:
        raise ValueError(f"unsupported classifier: {name}") from error
