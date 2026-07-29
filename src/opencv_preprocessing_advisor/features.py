"""OpenCV-native feature extractors for classical image classification."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .io import validate_bgr_image


def _as_float_matrix(rows: list[np.ndarray]) -> np.ndarray:
    if not rows:
        raise ValueError("at least one image is required")
    matrix = np.vstack(rows).astype(np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("feature matrix contains non-finite values")
    return matrix


@dataclass(frozen=True)
class ColorHistogramExtractor:
    bins: int = 32

    def transform(self, images: list[np.ndarray]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for image in images:
            validate_bgr_image(image)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            histograms = [
                cv2.calcHist([hsv], [0], None, [self.bins], [0, 180]).ravel(),
                cv2.calcHist([hsv], [1], None, [self.bins], [0, 256]).ravel(),
                cv2.calcHist([lab], [0], None, [self.bins], [0, 256]).ravel(),
            ]
            vector = np.concatenate(histograms).astype(np.float32)
            total = float(vector.sum())
            if total > 0:
                vector /= total
            rows.append(vector[None, :])
        return _as_float_matrix(rows)


class HOGExtractor:
    def __init__(self, size: tuple[int, int] = (128, 128)) -> None:
        if size[0] % 16 or size[1] % 16:
            raise ValueError("HOG size dimensions must be divisible by 16")
        self.size = size
        self._hog = cv2.HOGDescriptor(
            size,
            (16, 16),
            (8, 8),
            (8, 8),
            9,
        )

    def transform(self, images: list[np.ndarray]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for image in images:
            validate_bgr_image(image)
            gray = cv2.cvtColor(cv2.resize(image, self.size), cv2.COLOR_BGR2GRAY)
            vector = self._hog.compute(gray).reshape(1, -1)
            rows.append(vector)
        return _as_float_matrix(rows)


@dataclass(frozen=True)
class TextureStatsExtractor:
    size: tuple[int, int] = (128, 128)

    @staticmethod
    def _stats(response: np.ndarray) -> np.ndarray:
        absolute = np.abs(response).astype(np.float32)
        return np.asarray(
            [
                absolute.mean(),
                absolute.std(),
                np.percentile(absolute, 75),
                np.percentile(absolute, 95),
            ],
            dtype=np.float32,
        )

    def transform(self, images: list[np.ndarray]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for image in images:
            validate_bgr_image(image)
            gray = cv2.cvtColor(cv2.resize(image, self.size), cv2.COLOR_BGR2GRAY)
            gray_float = gray.astype(np.float32) / 255.0
            sobel_x = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3)
            responses = [
                cv2.magnitude(sobel_x, sobel_y),
                cv2.Laplacian(gray_float, cv2.CV_32F, ksize=3),
            ]
            for theta in (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4):
                kernel = cv2.getGaborKernel((15, 15), 3.0, theta, 8.0, 0.5, 0)
                responses.append(cv2.filter2D(gray_float, cv2.CV_32F, kernel))
            vector = np.concatenate([self._stats(response) for response in responses])
            rows.append(vector[None, :])
        return _as_float_matrix(rows)


class CombinedExtractor:
    def __init__(self, size: tuple[int, int] = (128, 128)) -> None:
        self.color = ColorHistogramExtractor()
        self.hog = HOGExtractor(size)
        self.texture = TextureStatsExtractor(size)

    def transform(self, images: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(
            [
                self.color.transform(images),
                self.hog.transform(images),
                self.texture.transform(images),
            ],
            axis=1,
        ).astype(np.float32)


class SiftBowExtractor:
    def __init__(
        self,
        vocabulary_size: int = 32,
        seed: int = 42,
        max_descriptors_per_image: int = 250,
    ) -> None:
        if vocabulary_size < 2:
            raise ValueError("vocabulary_size must be at least 2")
        self.vocabulary_size = vocabulary_size
        self.seed = seed
        self.max_descriptors_per_image = max_descriptors_per_image
        self._sift = cv2.SIFT_create()
        self.vocabulary_: np.ndarray | None = None

    def _descriptors(self, image: np.ndarray) -> np.ndarray | None:
        validate_bgr_image(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self._sift.detectAndCompute(gray, None)
        if not keypoints or descriptors is None:
            return None
        order = np.argsort([-keypoint.response for keypoint in keypoints])
        return descriptors[order[: self.max_descriptors_per_image]].astype(np.float32)

    def fit(self, images: list[np.ndarray]) -> SiftBowExtractor:
        trainer = cv2.BOWKMeansTrainer(self.vocabulary_size)
        descriptor_count = 0
        for image in images:
            descriptors = self._descriptors(image)
            if descriptors is not None:
                trainer.add(descriptors)
                descriptor_count += descriptors.shape[0]
        if descriptor_count < self.vocabulary_size:
            raise ValueError(
                "insufficient SIFT descriptors to build the requested visual vocabulary"
            )
        cv2.setRNGSeed(self.seed)
        self.vocabulary_ = trainer.cluster().astype(np.float32)
        return self

    def transform(self, images: list[np.ndarray]) -> np.ndarray:
        if self.vocabulary_ is None:
            raise RuntimeError("SiftBowExtractor must be fit before transform")
        rows: list[np.ndarray] = []
        for image in images:
            descriptors = self._descriptors(image)
            histogram = np.zeros(self.vocabulary_size, dtype=np.float32)
            if descriptors is not None:
                distances = np.linalg.norm(
                    descriptors[:, None, :] - self.vocabulary_[None, :, :],
                    axis=2,
                )
                assignments = np.argmin(distances, axis=1)
                histogram = np.bincount(
                    assignments,
                    minlength=self.vocabulary_size,
                ).astype(np.float32)
                total = float(histogram.sum())
                if total > 0:
                    histogram /= total
            rows.append(histogram[None, :])
        return _as_float_matrix(rows)
