import cv2
import numpy as np
from pathlib import Path

def preprocess_face_array(face_bgr_or_gray, size=(25, 25)):
    """Preprocesa un arreglo de imagen facial."""
    if len(face_bgr_or_gray.shape) == 3:
        gray = cv2.cvtColor(face_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_bgr_or_gray.copy()

    gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    gray = cv2.equalizeHist(gray)

    h, w = gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    axes = (int(w * 0.38), int(h * 0.48))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

    gray_masked = cv2.bitwise_and(gray, gray, mask=mask)
    gray_norm = gray_masked.astype(np.float32) / 255.0

    return gray_norm.flatten(), gray_norm

def preprocess_image_path(img_path: str | Path, size=(25, 25)):
    """Lee una imagen desde el disco y la preprocesa."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {img_path}")
    return preprocess_face_array(img, size=size)