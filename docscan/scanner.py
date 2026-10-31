"""Find a document in a photo and flatten it (OpenCV).

Steps: resize → grayscale → blur → Canny edges → close gaps → largest 4-sided
contour → perspective warp → contrast enhancement.
"""

from __future__ import annotations

import cv2
import numpy as np

WORK_WIDTH = 500


def order_points(pts: np.ndarray) -> np.ndarray:
    """Return corners as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def detect_corners(image: np.ndarray, min_area_ratio: float = 0.15) -> np.ndarray | None:
    """Corners of the largest quadrilateral, in original-image pixels, or None."""
    h, w = image.shape[:2]
    scale = WORK_WIDTH / w
    small = cv2.resize(image, (WORK_WIDTH, int(h * scale)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(cv2.dilate(edges, kernel, iterations=1), cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = min_area_ratio * small.shape[0] * small.shape[1]
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        if cv2.contourArea(contour) < min_area:
            break
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return order_points(approx) / scale
    return None


def default_corners(image: np.ndarray, margin: float = 0.05) -> np.ndarray:
    """Fallback when no document is found: a frame just inside the image edges."""
    h, w = image.shape[:2]
    mx, my = w * margin, h * margin
    return np.array([[mx, my], [w - mx, my], [w - mx, h - my], [mx, h - my]], dtype=np.float32)


def warp(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Perspective-correct the area inside ``corners``."""
    tl, tr, br, bl = order_points(corners)
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl]), dst)
    return cv2.warpPerspective(image, matrix, (width, height))


def enhance(image: np.ndarray) -> np.ndarray:
    """Even out lighting with CLAHE on the lightness channel (keeps colour)."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
