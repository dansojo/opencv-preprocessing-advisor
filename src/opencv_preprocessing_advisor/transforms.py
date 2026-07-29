"""Reusable OpenCV preprocessing operations."""

from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from .io import validate_bgr_image


def _validate_odd_kernel(kernel_size: int) -> None:
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")


def apply_lab_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    grid_size: int = 8,
) -> np.ndarray:
    validate_bgr_image(image)
    if clip_limit <= 0 or grid_size <= 0:
        raise ValueError("CLAHE parameters must be positive")
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    luminance, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(grid_size), int(grid_size)),
    )
    enhanced = cv2.merge((clahe.apply(luminance), a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def apply_auto_gamma(image: np.ndarray, target_midpoint: float = 0.5) -> np.ndarray:
    validate_bgr_image(image)
    if not 0 < target_midpoint < 1:
        raise ValueError("target_midpoint must be between 0 and 1")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    current = float(gray.mean() / 255.0)
    if current <= 1e-6 or current >= 1 - 1e-6:
        return image.copy()
    gamma = float(np.log(target_midpoint) / np.log(current))
    gamma = float(np.clip(gamma, 0.5, 2.0))
    lookup = np.array(
        [np.clip((value / 255.0) ** gamma * 255.0, 0, 255) for value in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(image, lookup)


def apply_gaussian(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 0.0,
) -> np.ndarray:
    validate_bgr_image(image)
    _validate_odd_kernel(kernel_size)
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def apply_median(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    validate_bgr_image(image)
    _validate_odd_kernel(kernel_size)
    return cv2.medianBlur(image, kernel_size)


def apply_bilateral(
    image: np.ndarray,
    diameter: int = 7,
    sigma_color: float = 45.0,
    sigma_space: float = 45.0,
) -> np.ndarray:
    validate_bgr_image(image)
    if diameter <= 0 or sigma_color <= 0 or sigma_space <= 0:
        raise ValueError("bilateral parameters must be positive")
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)


def apply_unsharp(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.0,
    amount: float = 1.0,
    threshold: int = 0,
) -> np.ndarray:
    validate_bgr_image(image)
    _validate_odd_kernel(kernel_size)
    if sigma < 0 or amount < 0 or threshold < 0:
        raise ValueError("unsharp parameters must be non-negative")
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    if threshold:
        low_contrast = cv2.cvtColor(cv2.absdiff(image, blurred), cv2.COLOR_BGR2GRAY) < threshold
        sharpened[low_contrast] = image[low_contrast]
    return sharpened


def apply_gray_bgr(image: np.ndarray) -> np.ndarray:
    validate_bgr_image(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def apply_blackhat(
    image: np.ndarray,
    kernel_size: int = 9,
    shape: str = "ellipse",
) -> np.ndarray:
    validate_bgr_image(image)
    _validate_odd_kernel(kernel_size)
    shapes = {
        "rect": cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
        "cross": cv2.MORPH_CROSS,
    }
    if shape not in shapes:
        raise ValueError(f"unsupported morphology shape: {shape}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(shapes[shape], (kernel_size, kernel_size))
    result = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


def normalize_uint8(image: np.ndarray) -> np.ndarray:
    validate_bgr_image(image)
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


TRANSFORMS: dict[str, Callable[..., np.ndarray]] = {
    "lab_clahe": apply_lab_clahe,
    "auto_gamma": apply_auto_gamma,
    "gaussian": apply_gaussian,
    "median": apply_median,
    "bilateral": apply_bilateral,
    "unsharp": apply_unsharp,
    "gray_bgr": apply_gray_bgr,
    "blackhat": apply_blackhat,
    "normalize": normalize_uint8,
}


def validate_transform_config(name: str, params: dict[str, Any]) -> None:
    if name not in TRANSFORMS:
        raise ValueError(f"unknown transform: {name}")
    for key in ("kernel_size",):
        if key in params:
            _validate_odd_kernel(int(params[key]))
