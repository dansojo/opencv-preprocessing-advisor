import numpy as np
import pytest

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
