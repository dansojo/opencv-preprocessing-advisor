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

TECHNIQUE_GUIDANCE = {
    "CLAHE": ("국소 대비가 낮거나 조명이 불균일할 때", "노이즈가 강한 미세 질감"),
    "HSV Saturation": ("색상 분리가 중요한 분류", "무채색 형태만 중요한 작업"),
    "Histogram Equalization": ("회색조 전역 대비가 낮을 때", "색 보존이 필요한 영상"),
    "Gaussian Blur": ("Gaussian 계열 노이즈 완화", "작은 경계가 핵심일 때"),
    "Median Blur": ("salt-and-pepper 노이즈", "얇은 선·작은 점이 특징일 때"),
    "Bilateral Filter": ("경계를 보존하며 평활화", "처리 시간이 매우 제한적일 때"),
    "Unsharp Mask": ("흐린 경계의 제어된 강화", "고주파 노이즈가 강할 때"),
    "Canny": ("명확한 경계 지도", "texture 강도 자체가 중요할 때"),
    "Sobel": ("수평·수직 1차 기울기", "방향 무관 구조만 필요할 때"),
    "Scharr": ("작은 kernel의 정밀 기울기", "강한 노이즈를 먼저 줄이지 않았을 때"),
    "Laplacian": ("방향 무관 2차 경계", "노이즈가 많은 원본"),
    "Gabor": ("방향성·주기성 texture", "형태나 색상만 중요한 데이터"),
    "Morphological Opening": ("작은 밝은 잡음 제거", "작은 밝은 객체가 실제 특징일 때"),
    "Blackhat": ("밝은 배경의 어두운 국소 결함", "자연색 출력이 필요할 때"),
    "Otsu Threshold": ("두 봉우리의 명암 분포", "조명이 심하게 불균일할 때"),
    "Contours": ("이진 객체의 외곽 형상", "객체 분리가 불안정한 texture 영상"),
    "Connected Components": ("분리된 이진 객체 수·크기", "붙어 있는 객체가 많은 영상"),
}


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
    if operation == "HSV Saturation":
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return (
            cv2.cvtColor(hsv[:, :, 1], cv2.COLOR_GRAY2BGR),
            "cv2.cvtColor(BGR2HSV) + cv2.split",
        )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if operation == "Histogram Equalization":
        result = cv2.equalizeHist(gray)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR), "cv2.equalizeHist"
    if operation == "Canny":
        result = cv2.Canny(gray, 50, 150)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR), "cv2.Canny"
    if operation == "Sobel":
        result = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=kernel))
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR), "cv2.Sobel"
    if operation == "Scharr":
        result = cv2.convertScaleAbs(cv2.Scharr(gray, cv2.CV_32F, 1, 0))
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR), "cv2.Scharr"
    if operation == "Laplacian":
        result = cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_32F, ksize=kernel))
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR), "cv2.Laplacian"
    if operation == "Gabor":
        gabor = cv2.getGaborKernel(
            (kernel, kernel),
            max(1.0, kernel / 5),
            np.pi / 4,
            max(2.0, kernel / 2),
            0.5,
            0,
        )
        result = cv2.convertScaleAbs(cv2.filter2D(gray, cv2.CV_32F, gabor))
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR), "cv2.getGaborKernel + cv2.filter2D"
    morphology_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel, kernel),
    )
    if operation == "Morphological Opening":
        result = cv2.morphologyEx(gray, cv2.MORPH_OPEN, morphology_kernel)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR), "cv2.morphologyEx(MORPH_OPEN)"
    if operation == "Blackhat":
        result = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, morphology_kernel)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR), "cv2.morphologyEx(MORPH_BLACKHAT)"
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if operation == "Otsu Threshold":
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), "cv2.threshold(THRESH_OTSU)"
    if operation == "Contours":
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = image.copy()
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
        return result, "cv2.findContours + cv2.drawContours"
    count, labels = cv2.connectedComponents(binary)
    scale = 255 / max(count - 1, 1)
    colored = cv2.applyColorMap(
        np.uint8(labels * scale),
        cv2.COLORMAP_TURBO,
    )
    colored[labels == 0] = 0
    return colored, "cv2.connectedComponents + cv2.applyColorMap"


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
        list(TECHNIQUE_GUIDANCE),
    )
    kernel = st.select_slider("Kernel size", options=[3, 5, 7, 9, 11], value=5)
    result, function_name = _apply(operation, image, kernel)
    left, right = st.columns(2)
    left.image(image, channels="BGR", caption="원본", use_container_width=True)
    right.image(result, channels="BGR", caption=operation, use_container_width=True)
    st.code(function_name)
    use_when, avoid_when = TECHNIQUE_GUIDANCE[operation]
    guidance_left, guidance_right = st.columns(2)
    guidance_left.success(f"사용할 때: {use_when}")
    guidance_right.warning(f"피할 때: {avoid_when}")
    st.info(
        "한 기술의 시각적 효과를 관찰하는 실험 화면입니다. "
        "분류 성능 효과는 데이터셋 벤치마크에서 별도로 확인해야 합니다."
    )
