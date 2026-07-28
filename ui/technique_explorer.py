"""Interactive OpenCV technique explorer."""

import cv2
import numpy as np
import streamlit as st

from opencv_preprocessing_advisor.io import decode_image_bytes
from opencv_preprocessing_advisor.transforms import (
    apply_bilateral,
    apply_gaussian,
    apply_lab_clahe,
    apply_median,
    apply_unsharp,
)


def _demo_image() -> np.ndarray:
    image = np.full((320, 480, 3), 75, np.uint8)
    cv2.rectangle(image, (40, 55), (210, 265), (185, 115, 45), -1)
    cv2.circle(image, (340, 160), 90, (55, 170, 215), -1)
    cv2.line(image, (20, 300), (455, 25), (245, 245, 245), 5)
    return image


def _apply(operation: str, image: np.ndarray, kernel: int) -> tuple[np.ndarray, str]:
    if operation == "CLAHE":
        return apply_lab_clahe(image), "cv2.createCLAHE + cv2.cvtColor(LAB)"
    if operation == "Gaussian Blur":
        return apply_gaussian(image, kernel), "cv2.GaussianBlur"
    if operation == "Median Blur":
        return apply_median(image, kernel), "cv2.medianBlur"
    if operation == "Bilateral Filter":
        return apply_bilateral(image), "cv2.bilateralFilter"
    if operation == "Unsharp Mask":
        return apply_unsharp(image, kernel_size=kernel), "cv2.GaussianBlur + cv2.addWeighted"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if operation == "Canny":
        result = cv2.Canny(gray, 50, 150)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR), "cv2.Canny"
    if operation == "Sobel":
        result = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=kernel))
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR), "cv2.Sobel"
    result = cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_32F, ksize=kernel))
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR), "cv2.Laplacian"


def render() -> None:
    st.title("OpenCV 기술 탐색기")
    uploaded = st.file_uploader(
        "실험 이미지 (없으면 합성 예제 사용)",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        key="explorer-upload",
    )
    image = decode_image_bytes(uploaded.getvalue()) if uploaded else _demo_image()
    operation = st.selectbox(
        "기술",
        [
            "CLAHE",
            "Gaussian Blur",
            "Median Blur",
            "Bilateral Filter",
            "Unsharp Mask",
            "Canny",
            "Sobel",
            "Laplacian",
        ],
    )
    kernel = st.select_slider("Kernel size", options=[3, 5, 7, 9, 11], value=5)
    result, function_name = _apply(operation, image, kernel)
    left, right = st.columns(2)
    left.image(image, channels="BGR", caption="원본", use_container_width=True)
    right.image(result, channels="BGR", caption=operation, use_container_width=True)
    st.code(function_name)
    st.info(
        "한 기술의 시각적 효과를 관찰하는 실험 화면입니다. "
        "분류 성능 효과는 데이터셋 벤치마크에서 별도로 확인해야 합니다."
    )
