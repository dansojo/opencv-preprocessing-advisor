import cv2
import numpy as np
import pytest

from opencv_preprocessing_advisor.features import (
    ColorHistogramExtractor,
    CombinedExtractor,
    HOGExtractor,
    SiftBowExtractor,
    TextureStatsExtractor,
)


def feature_images() -> list[np.ndarray]:
    images: list[np.ndarray] = []
    for index in range(10):
        image = np.full((128, 128, 3), 30 + index * 12, np.uint8)
        cv2.circle(
            image,
            (30 + index * 5, 64),
            15 + index,
            (220, 80 + index * 8, 30),
            2,
        )
        cv2.line(image, (8, 8 + index * 6), (120, 110 - index * 4), (255, 255, 255), 2)
        cv2.putText(
            image,
            str(index),
            (45, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (10, 240, 120),
            2,
        )
        images.append(image)
    return images


def test_color_histogram_is_fixed_length_and_normalized():
    images = feature_images()[:3]

    matrix = ColorHistogramExtractor().transform(images)

    assert matrix.shape == (3, 96)
    assert matrix.dtype == np.float32
    assert np.allclose(matrix.sum(axis=1), 1.0, atol=1e-5)


def test_hog_is_deterministic():
    images = feature_images()[:2]
    extractor = HOGExtractor(size=(128, 128))

    first = extractor.transform(images)
    second = extractor.transform(images)

    assert first.shape[0] == 2
    assert np.array_equal(first, second)


def test_texture_features_are_finite_and_fixed_length():
    matrix = TextureStatsExtractor().transform(feature_images()[:2])

    assert matrix.shape == (2, 24)
    assert np.isfinite(matrix).all()


def test_combined_concatenates_color_shape_and_texture():
    images = feature_images()[:2]
    color = ColorHistogramExtractor().transform(images)
    hog = HOGExtractor().transform(images)
    texture = TextureStatsExtractor().transform(images)

    combined = CombinedExtractor().transform(images)

    assert combined.shape == (2, color.shape[1] + hog.shape[1] + texture.shape[1])


def test_sift_bow_requires_fit():
    extractor = SiftBowExtractor(vocabulary_size=8, seed=42)

    with pytest.raises(RuntimeError, match="fit"):
        extractor.transform(feature_images()[:2])


def test_sift_bow_fits_and_returns_normalized_histograms():
    images = feature_images()
    extractor = SiftBowExtractor(vocabulary_size=8, seed=42)

    extractor.fit(images[:8])
    matrix = extractor.transform(images[8:])

    assert matrix.shape == (2, 8)
    assert matrix.dtype == np.float32
    assert np.allclose(matrix.sum(axis=1), 1.0, atol=1e-5)
