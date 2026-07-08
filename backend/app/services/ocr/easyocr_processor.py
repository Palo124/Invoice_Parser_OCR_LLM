import easyocr
from PIL import Image
import numpy as np

class EasyOCRProcessor:
    """
    A class for performing OCR on images using EasyOCR and reconstructing
    the text layout similar to the original image.
    
    Attributes:
        reader (easyocr.Reader): The EasyOCR reader initialized with the provided languages.
    
    Example:
        processor = EasyOCRProcessor(languages=['en'], gpu=False)
        layout_text = processor.image_to_text_layout('sample_image.png')
        print(layout_text)
    """
    def __init__(self, languages, gpu):
        # Initialize EasyOCR reader with the specified languages.
        self.reader = easyocr.Reader(languages, gpu=gpu)
    
    def image_to_text_layout(self, image, detail: int = 1, threshold: int = 30) -> str:
        """
        Extract text from an image using EasyOCR and reconstruct the text 
        preserving the approximate layout from the image.
        
        Args:
            image (PIL.Image.Image or str): A PIL Image object or a path to the image file.
            detail (int, optional): Must be 1 to retrieve detailed OCR results including bounding boxes.
            threshold (int, optional): Vertical distance threshold for grouping lines.
            
        Returns:
            str: A multi-line string with text arranged in an approximate layout.
        """
        # Open the image if a file path is provided.
        if isinstance(image, str):
            image = Image.open(image)
        image_np = np.array(image)
        # Get OCR results with bounding box details.
        results = self.reader.readtext(image_np, detail=detail)
        # Reconstruct layout, passing threshold
        layout_text = self._reconstruct_layout(results, threshold)
        return layout_text
    
    def _reconstruct_layout(self, results, threshold) -> str:
        """
        Reconstructs a textual layout by grouping OCR results into lines based on their y-coordinate.
        
        Args:
            results (list): OCR results as returned by EasyOCR in detailed mode.
            
        Returns:
            str: A reconstructed multi-line string.
        """
        # Convert OCR results into a list of dictionaries containing text and coordinates.
        items = []
        for bbox, text, conf in results:
            xs = [point[0] for point in bbox]
            ys = [point[1] for point in bbox]
            x_min = min(xs)
            y_min = min(ys)
            items.append({'text': text, 'x_min': x_min, 'y_min': y_min})
        
        # Sort items top-to-bottom.
        items.sort(key=lambda item: item['y_min'])
        
        # Group items into lines using a threshold on the y-coordinate.
        lines = []
        current_line = []
        current_y = None
        for item in items:
            if current_y is None:
                current_y = item['y_min']
                current_line.append(item)
            elif abs(item['y_min'] - current_y) <= threshold:
                current_line.append(item)
            else:
                lines.append(current_line)
                current_line = [item]
                current_y = item['y_min']
        if current_line:
            lines.append(current_line)
        
        # For each line, sort items left-to-right and reconstruct the line with spacing.
        output_lines = []
        for line in lines:
            line.sort(key=lambda item: item['x_min'])
            line_text = ""
            prev_x = None
            for item in line:
                if prev_x is not None:
                    # Calculate spacing based on horizontal gap.
                    gap = int((item['x_min'] - prev_x) / 50)  # scaling factor; adjust as needed.
                    line_text += " " * gap
                line_text += item['text']
                # Approximate the width of the text (this factor might need adjustment).
                prev_x = item['x_min'] + len(item['text']) * 7
            output_lines.append(line_text)
        return "\n".join(output_lines)
