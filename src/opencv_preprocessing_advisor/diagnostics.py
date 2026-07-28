"""Explainable, model-free image diagnostics."""

from dataclasses import fields

import cv2
import numpy as np

from .io import validate_bgr_image
from .models import ImageDiagnostics, MetricChange


def _entropy(gray: np.ndarray) -> float:
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    probabilities = histogram / max(float(histogram.sum()), 1.0)
    nonzero = probabilities[probabilities > 0]
    return float(-np.sum(nonzero * np.log2(nonzero)))


def _local_contrast(gray: np.ndarray, grid_size: int = 16) -> float:
    values: list[float] = []
    for y in range(0, gray.shape[0], grid_size):
        for x in range(0, gray.shape[1], grid_size):
            tile = gray[y : y + grid_size, x : x + grid_size]
            if tile.size:
                values.append(float(tile.std()))
    return float(np.mean(values)) if values else 0.0


def _noise_estimate(gray: np.ndarray) -> float:
    median = cv2.medianBlur(gray, 3)
    residual = cv2.absdiff(gray, median).astype(np.float64)
    return float(np.sqrt(np.mean(residual**2)))


def _illumination_nonuniformity(gray: np.ndarray) -> float:
    shortest = min(gray.shape[:2])
    kernel = min(51, max(3, shortest // 4))
    if kernel % 2 == 0:
        kernel += 1
    background = cv2.GaussianBlur(gray, (kernel, kernel), 0).astype(np.float64)
    mean = float(background.mean())
    return float(background.std() / mean) if mean > 1e-9 else 0.0


def _edges(gray: np.ndarray) -> tuple[float, float]:
    median = float(np.median(gray))
    low = int(max(0, 0.66 * median))
    high = int(min(255, max(low + 1, 1.33 * median)))
    edges = cv2.Canny(gray, low, high)
    edge_pixels = int(np.count_nonzero(edges))
    density = edge_pixels / edges.size
    if edge_pixels == 0:
        return float(density), 0.0
    count, _, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    connected_pixels = sum(
        int(stats[index, cv2.CC_STAT_AREA])
        for index in range(1, count)
        if stats[index, cv2.CC_STAT_AREA] >= 8
    )
    return float(density), float(connected_pixels / edge_pixels)


def _colorfulness(image: np.ndarray) -> float:
    b, g, r = cv2.split(image.astype(np.float32))
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    spread = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    center = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float(spread + 0.3 * center)


def analyze_image(image: np.ndarray) -> ImageDiagnostics:
    validate_bgr_image(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_density, edge_continuity = _edges(gray)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    return ImageDiagnostics(
        mean_brightness=float(gray.mean()),
        dark_clip_ratio=float(np.mean(gray <= 5)),
        bright_clip_ratio=float(np.mean(gray >= 250)),
        global_contrast=float(gray.std()),
        local_contrast=_local_contrast(gray),
        entropy=_entropy(gray),
        sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        noise_estimate=_noise_estimate(gray),
        illumination_nonuniformity=_illumination_nonuniformity(gray),
        edge_density=edge_density,
        edge_continuity=edge_continuity,
        colorfulness=_colorfulness(image),
        saturation_spread=float(hsv[:, :, 1].std()),
    )


def compare_diagnostics(
    before: ImageDiagnostics,
    after: ImageDiagnostics,
) -> dict[str, MetricChange]:
    changes: dict[str, MetricChange] = {}
    for item in fields(ImageDiagnostics):
        before_value = float(getattr(before, item.name))
        after_value = float(getattr(after, item.name))
        delta = after_value - before_value
        percent = None if abs(before_value) < 1e-9 else delta / abs(before_value) * 100.0
        changes[item.name] = MetricChange(
            name=item.name,
            before=before_value,
            after=after_value,
            absolute_delta=delta,
            percent_delta=percent,
        )
    return changes
