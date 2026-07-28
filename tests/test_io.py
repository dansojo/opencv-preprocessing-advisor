from pathlib import Path

import numpy as np
import pytest

from opencv_preprocessing_advisor.io import (
    decode_image,
    decode_image_bytes,
    encode_png,
    validate_bgr_image,
)


def test_unicode_path_round_trip(tmp_path: Path):
    image = np.full((16, 20, 3), 127, np.uint8)
    path = tmp_path / "표면_이미지.png"
    path.write_bytes(encode_png(image))

    loaded = decode_image(path)

    assert loaded.shape == image.shape
    assert np.array_equal(loaded, image)


def test_validate_rejects_float_image():
    with pytest.raises(ValueError, match="uint8"):
        validate_bgr_image(np.zeros((8, 8, 3), np.float32))


def test_decode_rejects_missing_path(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        decode_image(tmp_path / "missing.png")


def test_decode_image_bytes_round_trip():
    image = np.full((12, 14, 3), 91, np.uint8)

    loaded = decode_image_bytes(encode_png(image))

    assert np.array_equal(loaded, image)
