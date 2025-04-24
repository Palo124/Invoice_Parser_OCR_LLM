# paddle_ocr_processor.py
import numpy as np
from paddleocr import PaddleOCR

class PaddleOCRProcessor:
    """
    A simple wrapper around PaddleOCR to extract text from images.
    """
    def __init__(self, lang='en', use_angle_cls=False, show_log=False, use_gpu=False):
        """
        :param lang: Language for OCR (e.g., 'en', 'ch', 'cs', etc.).
        :param use_angle_cls: Whether to enable angle classification.
        :param show_log: Whether PaddleOCR prints logs.
        :param use_gpu: Whether to use GPU.
                        Note: Ensure PaddlePaddle GPU version is installed if True.
        """
        self.ocr = PaddleOCR(
            lang=lang,
            use_angle_cls=use_angle_cls,
            show_log=show_log,
            use_gpu=use_gpu
        )

    def extract_text_from_pil(self, pil_image):
        """
        Extracts text from a PIL image using PaddleOCR (returns plain text only).
        
        :param pil_image: A PIL Image object.
        :return: A single string with all recognized text lines joined by newlines.
        """
        np_image = np.array(pil_image)
        results = self.ocr.ocr(np_image, det=True, rec=True)

        text_lines = []
        for result in results:
            for box, (text, confidence) in result:
                text_lines.append(text)

        return "\n".join(text_lines)

    def extract_text_layout_from_pil(self, pil_image, threshold):
        """
        Extract text while preserving approximate layout, based on bounding box coordinates.
        
        :param pil_image: A PIL Image object.
        :param threshold: Vertical distance threshold (in pixels) for grouping words on the same line.
        :return: A multi-line string approximating the original text layout.
        """
        np_image = np.array(pil_image)
        # PaddleOCR returns a list of lists: each sub-list has [ [ [x1, y1], [x2, y2], ... ], (text, confidence) ]
        results = self.ocr.ocr(np_image, det=True, rec=True)

        # Build a list of items, each with text + bounding box extremes
        items = []
        for result in results:
            for box, (text, confidence) in result:
                text = text.strip()
                if not text:
                    continue
                # box is something like [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                x_coords = [pt[0] for pt in box]
                y_coords = [pt[1] for pt in box]
                x_min = min(x_coords)
                y_min = min(y_coords)
                items.append({
                    'text': text,
                    'x_min': x_min,
                    'y_min': y_min
                })

        # Sort items top-to-bottom
        items.sort(key=lambda it: it['y_min'])

        # Group items into lines by vertical threshold
        lines = []
        current_line = []
        current_y = None

        for item in items:
            if current_y is None:
                current_line = [item]
                current_y = item['y_min']
            else:
                if abs(item['y_min'] - current_y) <= threshold:
                    current_line.append(item)
                else:
                    lines.append(current_line)
                    current_line = [item]
                    current_y = item['y_min']
        if current_line:
            lines.append(current_line)

        # Within each line, sort items left-to-right and insert spaces based on horizontal gaps
        output_lines = []
        for line in lines:
            line.sort(key=lambda it: it['x_min'])
            line_text = ""
            prev_x = None

            for item in line:
                if prev_x is not None:
                    # Calculate the gap to determine how many spaces to insert
                    gap = max(0, int((item['x_min'] - prev_x) / 10))
                    line_text += " " * gap
                line_text += item['text']
                # Roughly approximate the end of the current word
                prev_x = item['x_min'] + len(item['text']) * 7

            output_lines.append(line_text)

        return "\n".join(output_lines)
