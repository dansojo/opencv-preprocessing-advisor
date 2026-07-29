"""Shared domain models used by the core, CLI, and Streamlit adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class TaskProfile(str, Enum):
    AUTO = "auto"
    SHAPE = "shape"
    COLOR = "color"
    TEXTURE = "texture"


@dataclass(frozen=True)
class ImageDiagnostics:
    mean_brightness: float
    dark_clip_ratio: float
    bright_clip_ratio: float
    global_contrast: float
    local_contrast: float
    entropy: float
    sharpness: float
    noise_estimate: float
    illumination_nonuniformity: float
    edge_density: float
    edge_continuity: float
    colorfulness: float
    saturation_spread: float


@dataclass(frozen=True)
class MetricChange:
    name: str
    before: float
    after: float
    absolute_delta: float
    percent_delta: float | None


@dataclass(frozen=True)
class PipelineStep:
    name: str
    image: np.ndarray
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineRun:
    pipeline_id: str
    display_name_ko: str
    display_name_en: str
    output_image: np.ndarray
    intermediate_images: tuple[PipelineStep, ...]
    processing_ms: float
    diagnostics_after: ImageDiagnostics | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoreComponent:
    name: str
    value: float
    weight: float
    weighted_value: float


@dataclass(frozen=True)
class ScoreBreakdown:
    total: float
    components: tuple[ScoreComponent, ...]
    reason_codes: tuple[str, ...] = ()
    warning_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Recommendation:
    pipeline_id: str
    suitability_score: float
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    reason_codes: tuple[str, ...]
    warning_codes: tuple[str, ...]
    score_components: tuple[ScoreComponent, ...]
    pipeline_run: PipelineRun
