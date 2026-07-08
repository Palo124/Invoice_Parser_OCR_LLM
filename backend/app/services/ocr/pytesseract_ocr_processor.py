# pytesseract_ocr_processor.py
import pytesseract
from pytesseract import Output
from PIL import Image

class PytesseractOCRProcessor:
    def __init__(self, tesseract_cmd=None, lang='ces'):
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.lang = lang

    def extract_text(self, image_path):
        """Extract plain text (no layout preservation) given a file path."""
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image, lang=self.lang)
        return text

    def extract_text_from_pil(self, pil_image):
        """Extract plain text (no layout preservation) from a PIL image."""
        text = pytesseract.image_to_string(pil_image, lang=self.lang)
        return text

    def extract_text_layout_from_pil(self, pil_image, threshold) -> str:
        """
        Extract text from a PIL image while preserving approximate layout.
        
        Args:
            pil_image (PIL.Image.Image): A PIL Image object.
            threshold (int): Vertical distance threshold (pixels) used to 
                             decide if words belong to the same line.
        
        Returns:
            str: A multi-line string that approximates the original text layout.
        """
        # Get bounding boxes and recognized text from Tesseract.
        data = pytesseract.image_to_data(pil_image, lang=self.lang, output_type=Output.DICT)

        # Collect bounding box info for each recognized word.
        items = []
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            text = data['text'][i].strip()
            # Skip empty results
            if not text:
                continue
            x_min = data['left'][i]
            y_min = data['top'][i]
            items.append({
                'text': text,
                'x_min': x_min,
                'y_min': y_min
            })
        
        # Sort items top-to-bottom by y_min.
        items.sort(key=lambda item: item['y_min'])

        # Group items into lines using threshold on y-coordinate.
        lines = []
        current_line = []
        current_y = None
        for item in items:
            if current_y is None:
                # First item of the first line
                current_y = item['y_min']
                current_line.append(item)
            else:
                if abs(item['y_min'] - current_y) <= threshold:
                    # Same line
                    current_line.append(item)
                else:
                    # New line
                    lines.append(current_line)
                    current_line = [item]
                    current_y = item['y_min']
        # Append the last line if not empty
        if current_line:
            lines.append(current_line)

        # For each line, sort left-to-right and insert spacing based on the gap
        output_lines = []
        for line in lines:
            line.sort(key=lambda item: item['x_min'])
            line_text = ""
            prev_x = None
            for item in line:
                # If there's a previous word, compute the horizontal gap
                if prev_x is not None:
                    gap = max(0, (item['x_min'] - prev_x) // 60)
                    line_text += " " * gap
                line_text += item['text']
                # Update prev_x to approximate the end of the current text
                # (Adjust character width factor as needed: 6, 7, 8, etc.)
                prev_x = item['x_min'] + len(item['text']) * 4
            output_lines.append(line_text)

        # Join reconstructed lines with newlines
        return "\n".join(output_lines)
