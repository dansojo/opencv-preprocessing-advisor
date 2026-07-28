import cv2
import numpy as np
import pytest


@pytest.fixture
def sample_bgr() -> np.ndarray:
    image = np.full((128, 128, 3), 80, np.uint8)
    cv2.rectangle(image, (24, 24), (104, 104), (180, 120, 60), -1)
    cv2.line(image, (16, 110), (112, 18), (240, 240, 240), 3)
    return image


@pytest.fixture
def low_contrast_bgr() -> np.ndarray:
    gradient = np.tile(np.linspace(105, 135, 128, dtype=np.uint8), (128, 1))
    return cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)


@pytest.fixture
def impulse_noise_bgr() -> np.ndarray:
    image = np.full((128, 128, 3), 120, np.uint8)
    image[::4, ::4] = 255
    image[2::4, 2::4] = 0
    return image
