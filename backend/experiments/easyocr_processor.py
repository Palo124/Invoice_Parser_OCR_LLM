"""
Optional EasyOCR experiment — not used by the core pipeline.
Install easyocr and torch manually if you want to run this module.
"""

import easyocr
import numpy as np
from PIL import Image


class EasyOCRProcessor:
    def __init__(self, languages, gpu):
        self.reader = easyocr.Reader(languages, gpu=gpu)

    def image_to_text_layout(self, image, detail: int = 1, threshold: int = 30) -> str:
        if isinstance(image, str):
            image = Image.open(image)
        image_np = np.array(image)
        results = self.reader.readtext(image_np, detail=detail)
        return self._reconstruct_layout(results, threshold)

    def _reconstruct_layout(self, results, threshold) -> str:
        items = []
        for bbox, text, _conf in results:
            xs = [point[0] for point in bbox]
            ys = [point[1] for point in bbox]
            items.append({"text": text, "x_min": min(xs), "y_min": min(ys)})

        items.sort(key=lambda item: item["y_min"])

        lines = []
        current_line = []
        current_y = None
        for item in items:
            if current_y is None:
                current_y = item["y_min"]
                current_line.append(item)
            elif abs(item["y_min"] - current_y) <= threshold:
                current_line.append(item)
            else:
                lines.append(current_line)
                current_line = [item]
                current_y = item["y_min"]
        if current_line:
            lines.append(current_line)

        output_lines = []
        for line in lines:
            line.sort(key=lambda item: item["x_min"])
            line_text = ""
            prev_x = None
            for item in line:
                if prev_x is not None:
                    gap = int((item["x_min"] - prev_x) / 50)
                    line_text += " " * gap
                line_text += item["text"]
                prev_x = item["x_min"] + len(item["text"]) * 7
            output_lines.append(line_text)
        return "\n".join(output_lines)
