import cv2
import numpy as np

from opencv_preprocessing_advisor.diagnostics import analyze_image, compare_diagnostics


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


def test_impulse_noise_increases_noise_estimate(impulse_noise_bgr):
    clean = np.full((128, 128, 3), 120, np.uint8)

    assert analyze_image(impulse_noise_bgr).noise_estimate > analyze_image(clean).noise_estimate


def test_constant_image_returns_only_finite_metrics():
    result = analyze_image(np.full((64, 64, 3), 128, np.uint8))

    assert all(np.isfinite(value) for value in vars(result).values())


def test_compare_diagnostics_reports_absolute_and_percent_change(sample_bgr):
    before = analyze_image(sample_bgr)
    after = analyze_image(cv2.convertScaleAbs(sample_bgr, alpha=1.2))

    changes = compare_diagnostics(before, after)

    assert "mean_brightness" in changes
    assert changes["mean_brightness"].absolute_delta == (
        after.mean_brightness - before.mean_brightness
    )
