import numpy as np
import pytest

from opencv_preprocessing_advisor.models import (
    ImageDiagnostics,
    PipelineRun,
    TaskProfile,
)
from opencv_preprocessing_advisor.scoring import (
    ScoredPipeline,
    load_profile_weights,
    rank_recommendations,
)


def diagnostics(**overrides) -> ImageDiagnostics:
    values = {
        "mean_brightness": 120.0,
        "dark_clip_ratio": 0.0,
        "bright_clip_ratio": 0.0,
        "global_contrast": 25.0,
        "local_contrast": 12.0,
        "entropy": 5.0,
        "sharpness": 100.0,
        "noise_estimate": 10.0,
        "illumination_nonuniformity": 0.1,
        "edge_density": 0.08,
        "edge_continuity": 0.5,
        "colorfulness": 30.0,
        "saturation_spread": 20.0,
    }
    values.update(overrides)
    return ImageDiagnostics(**values)


def scored_pipeline(pipeline_id: str, after: ImageDiagnostics) -> ScoredPipeline:
    image = np.zeros((8, 8, 3), np.uint8)
    run = PipelineRun(
        pipeline_id=pipeline_id,
        display_name_ko=pipeline_id,
        display_name_en=pipeline_id,
        output_image=image,
        intermediate_images=(),
        processing_ms=10.0,
    )
    return ScoredPipeline(run=run, before=diagnostics(), after=after)


def test_rank_returns_exactly_three_unique_recommendations():
    candidates = [
        scored_pipeline(f"pipeline-{index}", diagnostics(local_contrast=12.0 + index))
        for index in range(5)
    ]

    ranked = rank_recommendations(candidates, TaskProfile.AUTO, limit=3)

    assert len(ranked) == 3
    assert len({item.pipeline_id for item in ranked}) == 3
    assert ranked[0].suitability_score >= ranked[1].suitability_score


def test_excessive_clipping_creates_warning():
    candidate = scored_pipeline(
        "clipped",
        diagnostics(dark_clip_ratio=0.1, bright_clip_ratio=0.1),
    )

    ranked = rank_recommendations([candidate], TaskProfile.AUTO, limit=3)

    assert "clipping" in ranked[0].warning_codes


def test_score_is_finite_and_bounded():
    candidates = [
        scored_pipeline("a", diagnostics(local_contrast=30, sharpness=180)),
        scored_pipeline("b", diagnostics(noise_estimate=2, edge_continuity=0.8)),
        scored_pipeline("c", diagnostics(colorfulness=45, saturation_spread=35)),
    ]

    for recommendation in rank_recommendations(candidates, TaskProfile.TEXTURE, limit=3):
        assert np.isfinite(recommendation.suitability_score)
        assert 0.0 <= recommendation.suitability_score <= 100.0


def test_ranking_is_deterministic_for_equal_scores():
    candidates = [
        scored_pipeline("z-last", diagnostics()),
        scored_pipeline("a-first", diagnostics()),
    ]

    ranked = rank_recommendations(candidates, TaskProfile.AUTO, limit=2)

    assert [item.pipeline_id for item in ranked] == ["a-first", "z-last"]


def test_unchanged_metrics_do_not_claim_an_improvement():
    recommendation = rank_recommendations(
        [scored_pipeline("unchanged", diagnostics())],
        TaskProfile.AUTO,
        limit=1,
    )[0]

    assert all("개선" not in reason and "향상" not in reason for reason in recommendation.reasons)


def test_scoring_config_rejects_weights_that_do_not_sum_to_one(tmp_path):
    path = tmp_path / "scoring.yaml"
    path.write_text(
        "profiles:\n  auto:\n    sharpness: 0.5\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sum to 1.0"):
        load_profile_weights(path)
