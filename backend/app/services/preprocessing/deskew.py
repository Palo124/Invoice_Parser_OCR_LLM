# picture_rotation.py
import cv2
import numpy as np
import math

class ImageDeskewer:
    def __init__(self, image):
        """
        Initialize with an image, which can be either a NumPy array or a PIL Image.
        If a PIL Image is provided, it is converted to a NumPy array in BGR format.
        """
        # Check if the image is a PIL Image by looking for the 'mode' attribute.
        if hasattr(image, "mode"):
            # Convert PIL Image (RGB) to a NumPy array in BGR color space.
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        elif not isinstance(image, np.ndarray):
            # Attempt to convert to a NumPy array if not already.
            image = np.array(image)

        if image is None or image.size == 0:
            raise ValueError("Error: Provided image is None or empty")
        self.image = image

    def _preprocess(self):
        """
        Convert the image to grayscale, apply Gaussian blur, and detect edges.
        """
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 200, apertureSize=3)
        return edges

    def _detect_lines(self, edges):
        """
        Use the Hough Transform to detect lines in the edge map.
        """
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=100,
            minLineLength=100, maxLineGap=10
        )
        if lines is None:
            raise ValueError("No lines detected - unable to deskew")
        return lines

    def _calculate_median_angle(self, lines):
        """
        Calculate the median angle from the detected lines.
        Only near-horizontal lines (absolute angle less than 45 degrees) are considered.
        """
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if abs(angle) < 45:
                angles.append(angle)
        if not angles:
            raise ValueError("No valid angles found - unable to deskew")
        median_angle = np.median(angles)
        return median_angle

    def deskew(self):
        """
        Deskews the image by rotating it based on the median angle of the detected lines.
        Returns the deskewed image and the median angle used.
        """
        edges = self._preprocess()
        lines = self._detect_lines(edges)
        median_angle = self._calculate_median_angle(lines)

        (h, w) = self.image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(
            self.image, M, (w, h),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        return rotated, median_angle

    def save_image(self, output_path, image):
        """
        Save the processed image to the specified output path.
        """
        cv2.imwrite(output_path, image)
