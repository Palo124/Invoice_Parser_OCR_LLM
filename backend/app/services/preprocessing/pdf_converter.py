from pdf2image import convert_from_path

class PDFToImageConverter:
    """
    A class for converting PDF pages to images.

    Attributes:
        poppler_path (str, optional): The path to the Poppler binaries.
            This is required on some platforms (e.g., Windows). Default is None.
    """
    def __init__(self, poppler_path: str = None):
        self.poppler_path = poppler_path

    def convert_pdf_to_images(self, pdf_path: str, output_folder: str = None,
                                dpi: int = 200, fmt: str = 'png',
                                first_page: int = None, last_page: int = None):
        """
        Convert a PDF file into a list of images.

        Args:
            pdf_path (str): Path to the PDF file.
            output_folder (str, optional): If provided, saves the images to the folder.
            dpi (int, optional): Resolution for conversion. Default is 200.
            fmt (str, optional): Image format (e.g., 'png', 'jpeg'). Default is 'png'.
            first_page (int, optional): First page to convert. Default is None (start from the beginning).
            last_page (int, optional): Last page to convert. Default is None (convert until the end).

        Returns:
            list: A list of PIL.Image objects corresponding to PDF pages.
        """
        images = convert_from_path(pdf_path, dpi=dpi, output_folder=output_folder,
                                   fmt=fmt, first_page=first_page, last_page=last_page,
                                   poppler_path=self.poppler_path)
        return images
