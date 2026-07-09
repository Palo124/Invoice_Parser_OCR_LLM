import math

import cv2
import numpy as np
from PIL import Image


def _to_bgr(image) -> np.ndarray:
    """Normalize PIL or NumPy images to a 3-channel BGR array for OpenCV."""
    if hasattr(image, "mode"):
        return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

    array = np.asarray(image)
    if array.ndim == 2:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    if array.ndim == 3 and array.shape[2] == 1:
        return cv2.cvtColor(array[:, :, 0], cv2.COLOR_GRAY2BGR)
    if array.ndim == 3 and array.shape[2] == 4:
        return cv2.cvtColor(array, cv2.COLOR_RGBA2BGR)
    if array.ndim == 3 and array.shape[2] == 3:
        return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)

    raise ValueError(f"Unsupported image shape: {array.shape}")


class ImageDeskewer:
    def __init__(self, image):
        if image is None or (isinstance(image, np.ndarray) and image.size == 0):
            raise ValueError("Error: Provided image is None or empty")

        self.image = _to_bgr(image)

    def _preprocess(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.Canny(blur, 50, 200, apertureSize=3)

    def _detect_lines(self, edges):
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=100,
            minLineLength=100,
            maxLineGap=10,
        )
        if lines is None:
            raise ValueError("No lines detected - unable to deskew")
        return lines

    def _calculate_median_angle(self, lines):
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if abs(angle) < 45:
                angles.append(angle)
        if not angles:
            raise ValueError("No valid angles found - unable to deskew")
        return np.median(angles)

    def deskew(self):
        edges = self._preprocess()
        lines = self._detect_lines(edges)
        median_angle = self._calculate_median_angle(lines)

        height, width = self.image.shape[:2]
        center = (width // 2, height // 2)
        matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated_bgr = cv2.warpAffine(
            self.image,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        rotated_rgb = cv2.cvtColor(rotated_bgr, cv2.COLOR_BGR2RGB)
        return rotated_rgb, median_angle

    def save_image(self, output_path, image):
        if isinstance(image, np.ndarray):
            Image.fromarray(image).save(output_path)
            return
        cv2.imwrite(output_path, image)
