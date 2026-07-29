"""Configurable, deterministic preprocessing pipeline catalog."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import yaml

from .io import validate_bgr_image
from .models import PipelineRun, PipelineStep, TaskProfile
from .transforms import TRANSFORMS, validate_transform_config


@dataclass(frozen=True)
class StepDefinition:
    transform: str
    params: dict[str, Any]


@dataclass(frozen=True)
class PipelineDefinition:
    pipeline_id: str
    display_name_ko: str
    display_name_en: str
    profiles: tuple[TaskProfile, ...]
    steps: tuple[StepDefinition, ...]
    rationale_codes: tuple[str, ...]
    warning_codes: tuple[str, ...]


class PipelineCatalog:
    def __init__(self, definitions: tuple[PipelineDefinition, ...]) -> None:
        identifiers = [definition.pipeline_id for definition in definitions]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("pipeline IDs must be unique")
        self._definitions = {definition.pipeline_id: definition for definition in definitions}

    @property
    def pipeline_ids(self) -> tuple[str, ...]:
        return tuple(self._definitions)

    @property
    def definitions(self) -> tuple[PipelineDefinition, ...]:
        return tuple(self._definitions.values())

    @classmethod
    def from_yaml(cls, path: Path | str) -> PipelineCatalog:
        config_path = Path(path)
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("configuration root must be a mapping")
        if not isinstance(payload.get("pipelines"), list):
            raise TypeError("configuration must contain a pipelines list")
        definitions: list[PipelineDefinition] = []
        for raw in payload["pipelines"]:
            steps: list[StepDefinition] = []
            for raw_step in raw.get("steps", []):
                name = str(raw_step["transform"])
                params = dict(raw_step.get("params", {}))
                validate_transform_config(name, params)
                steps.append(StepDefinition(name, params))
            if not steps:
                raise ValueError(f"pipeline {raw.get('id')} must contain at least one step")
            definitions.append(
                PipelineDefinition(
                    pipeline_id=str(raw["id"]),
                    display_name_ko=str(raw["display_name_ko"]),
                    display_name_en=str(raw["display_name_en"]),
                    profiles=tuple(TaskProfile(profile) for profile in raw["profiles"]),
                    steps=tuple(steps),
                    rationale_codes=tuple(raw.get("rationale_codes", [])),
                    warning_codes=tuple(raw.get("warning_codes", [])),
                )
            )
        return cls(tuple(definitions))

    def run(self, pipeline_id: str, image: np.ndarray) -> PipelineRun:
        validate_bgr_image(image)
        try:
            definition = self._definitions[pipeline_id]
        except KeyError as error:
            raise KeyError(f"unknown pipeline: {pipeline_id}") from error
        current = image.copy()
        intermediate: list[PipelineStep] = []
        started = perf_counter()
        for step in definition.steps:
            current = TRANSFORMS[step.transform](current, **step.params)
            validate_bgr_image(current)
            intermediate.append(PipelineStep(step.transform, current.copy(), dict(step.params)))
        processing_ms = (perf_counter() - started) * 1000.0
        return PipelineRun(
            pipeline_id=definition.pipeline_id,
            display_name_ko=definition.display_name_ko,
            display_name_en=definition.display_name_en,
            output_image=current,
            intermediate_images=tuple(intermediate),
            processing_ms=processing_ms,
            metadata={
                "profiles": [profile.value for profile in definition.profiles],
                "rationale_codes": list(definition.rationale_codes),
                "warning_codes": list(definition.warning_codes),
            },
        )
