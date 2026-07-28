"""Unicode-safe OpenCV image input and output."""

from pathlib import Path

import cv2
import numpy as np


def validate_bgr_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")
    if image.dtype != np.uint8:
        raise ValueError("image dtype must be uint8")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have BGR shape (height, width, 3)")
    if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image must not be empty")


def decode_image(path: Path | str) -> np.ndarray:
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    data = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"unable to decode image: {image_path}")
    validate_bgr_image(image)
    return image


def encode_png(image: np.ndarray) -> bytes:
    validate_bgr_image(image)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("OpenCV failed to encode image as PNG")
    return encoded.tobytes()
