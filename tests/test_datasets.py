from pathlib import Path

import numpy as np
import pytest

from opencv_preprocessing_advisor.datasets import discover_dataset, stratified_folds
from opencv_preprocessing_advisor.io import encode_png


def make_dataset(root: Path, classes: tuple[str, ...], count: int = 10) -> Path:
    for class_index, class_name in enumerate(classes):
        class_dir = root / class_name
        class_dir.mkdir(parents=True)
        for image_index in range(count):
            image = np.full(
                (16, 16, 3),
                30 + class_index * 100 + image_index,
                np.uint8,
            )
            (class_dir / f"{image_index:03}.png").write_bytes(encode_png(image))
    return root


def test_discovers_class_directories_in_sorted_order(tmp_path):
    dataset = make_dataset(tmp_path / "dataset", ("square", "circle"))

    manifest = discover_dataset(dataset)

    assert manifest.class_names == ("circle", "square")
    assert len(manifest.samples) == 20
    assert all(sample.checksum for sample in manifest.samples)


def test_stratified_folds_are_disjoint_and_complete():
    labels = np.array([0] * 10 + [1] * 10)

    folds = stratified_folds(labels, n_splits=5, seed=42)

    seen_test: set[int] = set()
    for fold in folds:
        assert set(fold.train_indices).isdisjoint(fold.test_indices)
        assert set(labels[fold.test_indices]) == {0, 1}
        seen_test.update(fold.test_indices.tolist())
    assert seen_test == set(range(20))


def test_dataset_rejects_fewer_than_two_classes(tmp_path):
    dataset = make_dataset(tmp_path / "dataset", ("only",))

    with pytest.raises(ValueError, match="at least two classes"):
        discover_dataset(dataset)


def test_dataset_rejects_too_few_images_per_class(tmp_path):
    dataset = make_dataset(tmp_path / "dataset", ("a", "b"), count=4)

    with pytest.raises(ValueError, match="five valid images"):
        discover_dataset(dataset)


def test_unreadable_image_is_reported_and_skipped(tmp_path):
    dataset = make_dataset(tmp_path / "dataset", ("a", "b"), count=6)
    (dataset / "a" / "broken.png").write_text("not an image", encoding="utf-8")

    manifest = discover_dataset(dataset)

    assert len(manifest.samples) == 12
    assert len(manifest.skipped_files) == 1
    assert manifest.skipped_files[0].path.name == "broken.png"
