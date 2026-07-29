"""Classification dataset discovery and deterministic stratified folds."""

from __future__ import annotations

import shutil
import stat
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

import numpy as np

from .io import decode_image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_ZIP_FILES = 10_000
MAX_ZIP_UNCOMPRESSED_BYTES = 1_000_000_000


@dataclass(frozen=True)
class DatasetSample:
    path: Path
    class_name: str
    class_index: int
    width: int
    height: int
    checksum: str


@dataclass(frozen=True)
class SkippedFile:
    path: Path
    reason: str


@dataclass(frozen=True)
class DatasetManifest:
    root: Path
    class_names: tuple[str, ...]
    samples: tuple[DatasetSample, ...]
    skipped_files: tuple[SkippedFile, ...]

    @property
    def labels(self) -> np.ndarray:
        return np.asarray([sample.class_index for sample in self.samples], dtype=np.int32)


@dataclass(frozen=True)
class Fold:
    train_indices: np.ndarray
    test_indices: np.ndarray


def extract_dataset_zip(data: bytes, destination: Path | str) -> Path:
    extraction_root = Path(destination)
    if extraction_root.exists():
        raise FileExistsError(f"ZIP destination already exists: {extraction_root}")
    try:
        archive = ZipFile(BytesIO(data))
    except BadZipFile as error:
        raise ValueError("uploaded file is not a valid ZIP archive") from error
    with archive:
        members = archive.infolist()
        files = [member for member in members if not member.is_dir()]
        if not files:
            raise ValueError("ZIP archive contains no files")
        if len(files) > MAX_ZIP_FILES:
            raise ValueError(f"ZIP archive exceeds {MAX_ZIP_FILES} files")
        if sum(member.file_size for member in files) > MAX_ZIP_UNCOMPRESSED_BYTES:
            raise ValueError("ZIP archive exceeds the uncompressed size limit")
        validated: list[tuple[object, PurePosixPath]] = []
        for member in members:
            path = PurePosixPath(member.filename.replace("\\", "/"))
            if path.is_absolute() or ".." in path.parts or not path.parts or ":" in path.parts[0]:
                raise ValueError(f"unsafe ZIP path: {member.filename}")
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"ZIP symbolic links are not supported: {member.filename}")
            if member.flag_bits & 0x1:
                raise ValueError("encrypted ZIP members are not supported")
            validated.append((member, path))

        extraction_root.mkdir(parents=True)
        for member, path in validated:
            target = extraction_root.joinpath(*path.parts)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)

    top_level = {path.parts[0] for _, path in validated if path.parts}
    if len(top_level) == 1:
        candidate = extraction_root / next(iter(top_level))
        if candidate.is_dir():
            return candidate
    return extraction_root


def discover_dataset(root: Path | str) -> DatasetManifest:
    dataset_root = Path(root)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {dataset_root}")
    class_dirs = sorted(
        (
            path
            for path in dataset_root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ),
        key=lambda path: path.name.casefold(),
    )
    if len(class_dirs) < 2:
        raise ValueError("dataset must contain at least two classes")

    samples: list[DatasetSample] = []
    skipped: list[SkippedFile] = []
    class_names = tuple(path.name for path in class_dirs)
    for class_index, class_dir in enumerate(class_dirs):
        class_samples: list[DatasetSample] = []
        candidates = sorted(
            (
                path
                for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
            ),
            key=lambda path: str(path).casefold(),
        )
        for path in candidates:
            try:
                image = decode_image(path)
            except (FileNotFoundError, ValueError) as error:
                skipped.append(SkippedFile(path, str(error)))
                continue
            class_samples.append(
                DatasetSample(
                    path=path,
                    class_name=class_dir.name,
                    class_index=class_index,
                    width=int(image.shape[1]),
                    height=int(image.shape[0]),
                    checksum=sha256(path.read_bytes()).hexdigest(),
                )
            )
        if len(class_samples) < 5:
            raise ValueError(
                f"class '{class_dir.name}' must contain at least five valid images; "
                f"found {len(class_samples)}"
            )
        samples.extend(class_samples)

    return DatasetManifest(
        root=dataset_root,
        class_names=class_names,
        samples=tuple(samples),
        skipped_files=tuple(skipped),
    )


def stratified_folds(
    labels: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> list[Fold]:
    label_array = np.asarray(labels)
    if label_array.ndim != 1 or label_array.size == 0:
        raise ValueError("labels must be a nonempty one-dimensional array")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    classes, counts = np.unique(label_array, return_counts=True)
    if classes.size < 2:
        raise ValueError("labels must contain at least two classes")
    actual_splits = min(n_splits, int(counts.min()))
    if actual_splits < 2:
        raise ValueError("each class must have at least two samples")

    generator = np.random.default_rng(seed)
    class_chunks: dict[int, list[np.ndarray]] = {}
    for class_value in classes:
        indices = np.flatnonzero(label_array == class_value)
        shuffled = generator.permutation(indices)
        class_chunks[int(class_value)] = list(np.array_split(shuffled, actual_splits))

    all_indices = np.arange(label_array.size, dtype=np.int64)
    folds: list[Fold] = []
    for fold_index in range(actual_splits):
        test_indices = np.sort(
            np.concatenate([class_chunks[int(value)][fold_index] for value in classes])
        ).astype(np.int64)
        train_indices = np.setdiff1d(all_indices, test_indices, assume_unique=True)
        folds.append(Fold(train_indices=train_indices, test_indices=test_indices))
    return folds
