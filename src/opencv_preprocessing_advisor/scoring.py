"""Transparent heuristic scoring for single-image preprocessing advice."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .models import (
    ImageDiagnostics,
    PipelineRun,
    Recommendation,
    ScoreBreakdown,
    ScoreComponent,
    TaskProfile,
)


@dataclass(frozen=True)
class ScoredPipeline:
    run: PipelineRun
    before: ImageDiagnostics
    after: ImageDiagnostics


PROFILE_WEIGHTS: dict[TaskProfile, dict[str, float]] = {
    TaskProfile.AUTO: {
        "local_contrast": 0.25,
        "entropy": 0.15,
        "noise_reduction": 0.20,
        "sharpness": 0.15,
        "edge_continuity": 0.15,
        "clipping_control": 0.10,
    },
    TaskProfile.SHAPE: {
        "local_contrast": 0.20,
        "edge_continuity": 0.25,
        "sharpness": 0.20,
        "noise_reduction": 0.10,
        "edge_density": 0.15,
        "clipping_control": 0.10,
    },
    TaskProfile.COLOR: {
        "color_preservation": 0.25,
        "saturation_separation": 0.20,
        "local_contrast": 0.15,
        "exposure_balance": 0.20,
        "clipping_control": 0.20,
    },
    TaskProfile.TEXTURE: {
        "local_contrast": 0.20,
        "sharpness": 0.20,
        "entropy": 0.15,
        "edge_density": 0.20,
        "noise_reduction": 0.15,
        "clipping_control": 0.10,
    },
}

SCORING_METRICS = frozenset(metric for weights in PROFILE_WEIGHTS.values() for metric in weights)


def load_profile_weights(path: Path | str) -> dict[TaskProfile, dict[str, float]]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("profiles"), dict):
        raise TypeError("scoring config must contain a profiles mapping")
    loaded: dict[TaskProfile, dict[str, float]] = {}
    for profile_name, raw_weights in payload["profiles"].items():
        try:
            profile = TaskProfile(profile_name)
        except ValueError as error:
            raise ValueError(f"unsupported scoring profile: {profile_name}") from error
        if not isinstance(raw_weights, dict) or not raw_weights:
            raise ValueError(f"scoring profile '{profile.value}' must define weights")
        weights: dict[str, float] = {}
        for metric, raw_weight in raw_weights.items():
            if metric not in SCORING_METRICS:
                raise ValueError(f"unsupported scoring metric: {metric}")
            if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
                raise TypeError(f"weight for '{metric}' must be numeric")
            weight = float(raw_weight)
            if not np.isfinite(weight) or weight < 0:
                raise ValueError(f"weight for '{metric}' must be finite and nonnegative")
            weights[metric] = weight
        if not np.isclose(sum(weights.values()), 1.0, atol=1e-9):
            raise ValueError(f"weights for profile '{profile.value}' must sum to 1.0")
        loaded[profile] = weights
    missing = set(TaskProfile) - set(loaded)
    if missing:
        names = ", ".join(sorted(profile.value for profile in missing))
        raise ValueError(f"scoring config is missing profiles: {names}")
    return loaded


REASON_TEXT = {
    "local_contrast": "국소 대비가 개선되었습니다.",
    "entropy": "명암 정보량이 개선되었습니다.",
    "noise_reduction": "고주파 노이즈 추정치가 감소했습니다.",
    "sharpness": "세부 구조의 선명도가 보존·향상되었습니다.",
    "edge_continuity": "연결된 경계 구조가 더 잘 보존되었습니다.",
    "edge_density": "형태·질감 경계가 더 드러났습니다.",
    "clipping_control": "암부·명부 정보 손실을 억제했습니다.",
    "color_preservation": "색상 구분 정보를 비교적 잘 보존했습니다.",
    "saturation_separation": "채도 분포의 구분력이 향상되었습니다.",
    "exposure_balance": "평균 노출이 중간 범위에 가까워졌습니다.",
}


WARNING_TEXT = {
    "clipping": "암부 또는 명부의 클리핑이 증가했습니다.",
    "excessive_edges": "엣지가 과도하게 증가해 노이즈도 강조될 수 있습니다.",
    "oversmoothing": "선명도가 크게 감소해 작은 구조가 사라질 수 있습니다.",
    "color_loss": "색상 정보가 크게 감소했습니다.",
}


def _reason_text(component: ScoreComponent) -> str:
    if component.value > 52.0:
        return REASON_TEXT[component.name]
    if component.value >= 48.0:
        return f"{component.name} 지표가 기준선과 비슷한 수준으로 유지되었습니다."
    return f"{component.name} 지표가 기준선보다 낮아져 강한 긍정 근거로 보기 어렵습니다."


def _bounded_improvement(before: float, after: float, scale: float = 50.0) -> float:
    denominator = max(abs(before), 1e-9)
    relative = (after - before) / denominator
    return float(np.clip(50.0 + relative * scale, 0.0, 100.0))


def _preservation(before: float, after: float) -> float:
    if abs(before) < 1e-9:
        return 100.0 if abs(after) < 1e-9 else 50.0
    relative_loss = abs(after - before) / abs(before)
    return float(np.clip(100.0 - relative_loss * 100.0, 0.0, 100.0))


def _component_values(
    before: ImageDiagnostics,
    after: ImageDiagnostics,
) -> dict[str, float]:
    before_clip = before.dark_clip_ratio + before.bright_clip_ratio
    after_clip = after.dark_clip_ratio + after.bright_clip_ratio
    clipping_control = float(np.clip(100.0 - after_clip * 500.0, 0.0, 100.0))
    exposure_balance = float(
        np.clip(100.0 - abs(after.mean_brightness - 127.5) / 127.5 * 100.0, 0.0, 100.0)
    )
    edge_score = _bounded_improvement(before.edge_density, after.edge_density, 35.0)
    if after.edge_density > max(0.25, before.edge_density * 2.5):
        edge_score *= 0.5
    return {
        "local_contrast": _bounded_improvement(
            before.local_contrast,
            after.local_contrast,
        ),
        "entropy": _bounded_improvement(before.entropy, after.entropy, 40.0),
        "noise_reduction": _bounded_improvement(
            before.noise_estimate,
            2 * before.noise_estimate - after.noise_estimate,
        ),
        "sharpness": _bounded_improvement(before.sharpness, after.sharpness, 35.0),
        "edge_continuity": _bounded_improvement(
            before.edge_continuity,
            after.edge_continuity,
        ),
        "edge_density": edge_score,
        "clipping_control": clipping_control,
        "color_preservation": _preservation(before.colorfulness, after.colorfulness),
        "saturation_separation": _bounded_improvement(
            before.saturation_spread,
            after.saturation_spread,
            35.0,
        ),
        "exposure_balance": exposure_balance,
        "_clip_growth": after_clip - before_clip,
    }


def score_pipeline(
    before: ImageDiagnostics,
    after: ImageDiagnostics,
    profile: TaskProfile,
    profile_weights: dict[TaskProfile, dict[str, float]] | None = None,
) -> ScoreBreakdown:
    values = _component_values(before, after)
    weights = profile_weights or PROFILE_WEIGHTS
    components = tuple(
        ScoreComponent(
            name=name,
            value=values[name],
            weight=weight,
            weighted_value=values[name] * weight,
        )
        for name, weight in weights[profile].items()
    )
    warnings: list[str] = []
    if values["_clip_growth"] > 0.01 or after.dark_clip_ratio + after.bright_clip_ratio > 0.05:
        warnings.append("clipping")
    if after.edge_density > max(0.25, before.edge_density * 2.5):
        warnings.append("excessive_edges")
    if before.sharpness > 1e-9 and after.sharpness < before.sharpness * 0.5:
        warnings.append("oversmoothing")
    if before.colorfulness > 1e-9 and after.colorfulness < before.colorfulness * 0.5:
        warnings.append("color_loss")
    reason_codes = tuple(
        component.name
        for component in sorted(components, key=lambda item: (-item.value, item.name))[:2]
    )
    total = float(np.clip(sum(item.weighted_value for item in components), 0.0, 100.0))
    return ScoreBreakdown(
        total=total,
        components=components,
        reason_codes=reason_codes,
        warning_codes=tuple(warnings),
    )


def rank_recommendations(
    candidates: list[ScoredPipeline],
    profile: TaskProfile,
    limit: int = 3,
    profile_weights: dict[TaskProfile, dict[str, float]] | None = None,
) -> list[Recommendation]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    recommendations: list[Recommendation] = []
    for candidate in candidates:
        breakdown = score_pipeline(
            candidate.before,
            candidate.after,
            profile,
            profile_weights,
        )
        components_by_name = {component.name: component for component in breakdown.components}
        recommendations.append(
            Recommendation(
                pipeline_id=candidate.run.pipeline_id,
                suitability_score=breakdown.total,
                reasons=tuple(
                    _reason_text(components_by_name[code]) for code in breakdown.reason_codes
                ),
                warnings=tuple(WARNING_TEXT[code] for code in breakdown.warning_codes),
                reason_codes=breakdown.reason_codes,
                warning_codes=breakdown.warning_codes,
                score_components=breakdown.components,
                pipeline_run=candidate.run,
            )
        )
    recommendations.sort(
        key=lambda item: (
            -item.suitability_score,
            len(item.warning_codes),
            item.pipeline_run.processing_ms,
            item.pipeline_id,
        )
    )
    return recommendations[:limit]
