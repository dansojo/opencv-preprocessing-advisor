"""Command-line interface for reproducible advice and benchmarks."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .datasets import discover_dataset
from .io import decode_image, encode_png
from .models import TaskProfile
from .reports import ReportWriter
from .services import (
    BenchmarkConfig,
    BenchmarkService,
    ImageAdvisorService,
)


def _csv_values(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("provide at least one comma-separated value")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="opencv-prep",
        description="Explainable OpenCV preprocessing recommendations and benchmarks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    image_parser = subparsers.add_parser("analyze-image", help="analyze one image")
    image_parser.add_argument("--image", type=Path, required=True)
    image_parser.add_argument(
        "--profile",
        choices=[profile.value for profile in TaskProfile],
        default=TaskProfile.AUTO.value,
    )
    image_parser.add_argument("--output", type=Path, default=Path("outputs"))

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="benchmark a class-folder dataset",
    )
    benchmark_parser.add_argument("--dataset", type=Path, required=True)
    benchmark_parser.add_argument("--folds", type=int, default=5)
    benchmark_parser.add_argument(
        "--pipelines",
        type=_csv_values,
        default=("original", "lab-clahe", "clahe-bilateral"),
    )
    benchmark_parser.add_argument(
        "--features",
        type=_csv_values,
        default=("combined",),
    )
    benchmark_parser.add_argument(
        "--classifiers",
        type=_csv_values,
        default=("svm", "knn", "rtrees"),
    )
    benchmark_parser.add_argument("--seed", type=int, default=42)
    benchmark_parser.add_argument("--output", type=Path, default=Path("outputs"))

    check_parser = subparsers.add_parser("self-check", help="run synthetic end-to-end checks")
    check_parser.add_argument("--output", type=Path)
    return parser


def _analyze_image(args: argparse.Namespace) -> int:
    image = decode_image(args.image)
    result = ImageAdvisorService().analyze(image, TaskProfile(args.profile))
    report = ReportWriter(args.output).write_image_advice(result)
    print(f"Image advice report: {report}")
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    manifest = discover_dataset(args.dataset)
    config = BenchmarkConfig(
        pipeline_ids=tuple(args.pipelines),
        feature_profiles=tuple(args.features),
        classifier_names=tuple(args.classifiers),
        folds=args.folds,
        seed=args.seed,
    )
    result = BenchmarkService().run(manifest, config)
    report = ReportWriter(args.output).write_benchmark(result)
    print(f"Benchmark report: {report}")
    print(
        f"Best: {result.top_entries[0].pipeline_id} / "
        f"{result.top_entries[0].feature_profile} / "
        f"{result.top_entries[0].classifier_name} "
        f"(macro F1={result.top_entries[0].cross_validation.mean_macro_f1:.3f})"
    )
    return 0


def _synthetic_fixture_data(root: Path) -> tuple[Path, Path]:
    fixtures = root / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    gradient = np.tile(np.linspace(105, 135, 128, dtype=np.uint8), (128, 1))
    low_contrast = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
    image_path = fixtures / "low_contrast.png"
    image_path.write_bytes(encode_png(low_contrast))

    dataset_root = fixtures / "classification"
    for class_name in ("circle", "square"):
        class_dir = dataset_root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for index in range(6):
            image = np.full((96, 96, 3), 35 + index, np.uint8)
            if class_name == "circle":
                cv2.circle(image, (48, 48), 18 + index, (225, 225, 225), -1)
            else:
                cv2.rectangle(
                    image,
                    (27 - index, 27 - index),
                    (69 + index, 69 + index),
                    (225, 225, 225),
                    -1,
                )
            (class_dir / f"{index:02}.png").write_bytes(encode_png(image))
    return image_path, dataset_root


def _run_self_check(args: argparse.Namespace) -> int:
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.output is None:
        temporary = tempfile.TemporaryDirectory(prefix="opencv-prep-self-check-")
        root = Path(temporary.name)
    else:
        root = args.output
        root.mkdir(parents=True, exist_ok=True)
    try:
        image_path, dataset_root = _synthetic_fixture_data(root)
        image_result = ImageAdvisorService().analyze(
            decode_image(image_path),
            TaskProfile.AUTO,
        )
        if len(image_result.recommendations) != 3:
            raise RuntimeError("single-image advice did not return three recommendations")
        ReportWriter(root / "reports").write_image_advice(image_result)

        benchmark_result = BenchmarkService().run(
            discover_dataset(dataset_root),
            BenchmarkConfig(
                pipeline_ids=("original", "lab-clahe"),
                feature_profiles=("shape",),
                classifier_names=("svm",),
                folds=3,
                seed=42,
            ),
        )
        if not benchmark_result.entries:
            raise RuntimeError("synthetic benchmark produced no entries")
        ReportWriter(root / "reports").write_benchmark(benchmark_result)
        print("SELF-CHECK PASSED")
        print(f"Artifacts: {root}")
        return 0
    finally:
        if temporary is not None:
            temporary.cleanup()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "analyze-image":
            return _analyze_image(args)
        if args.command == "benchmark":
            return _benchmark(args)
        if args.command == "self-check":
            return _run_self_check(args)
        parser.error(f"unknown command: {args.command}")
    except (FileNotFoundError, ValueError, KeyError) as error:
        print(f"Input error: {error}", file=sys.stderr)
        return 2
    except Exception as error:  # noqa: BLE001 - CLI boundary maps failures to exit code 1.
        print(f"Processing failed: {error}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
