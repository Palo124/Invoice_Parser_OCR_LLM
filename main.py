import torch
import json
import os
from pdf_converter import PDFToImageConverter
from easyocr_processor import EasyOCRProcessor
from deepInfra_deepseek import DeepSeekInterface
from deepInfra_second_model import SecondInterface
from deepInfra_third_model import ThirdInterface
from invoice_db import InvoiceDB  # Database insertion removed for now
from dotenv import load_dotenv
from json_extractor import JSONExtractor
from prompt import get_prompt
from picture_rotation import ImageDeskewer
from pytesseract_ocr_processor import PytesseractOCRProcessor
from PIL import Image
from paddle_ocr_processor import PaddleOCRProcessor
import numpy as np
from tmr import triple_modular_redundancy

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def main():
    # Specify your input file path (either a PDF or PNG file)
    file_name = "Test_invoce_34.pdf"
    #file_name = "CAREFUL_INTERNET_TEST.png"
    input_file = f'/home/pavol/Documents/Python_Codes/OCR_Invoices/Test_invoces/OneDrive_2025-03-24/Test_in_test/{file_name}'  # For PDF
    #input_file = f'/home/pavol/Documents/Python_Codes/OCR_Invoices/Test_invoces/OneDrive_2025-03-24/{file_name}'  # For normal image

    # Determine the file extension to choose the processing method.
    file_ext = os.path.splitext(input_file)[1].lower()
    images = []

    if file_ext == ".pdf":
        # Convert PDF pages to images.
        converter = PDFToImageConverter(poppler_path=None)
        images = converter.convert_pdf_to_images(input_file, dpi=300)
    elif file_ext == ".png":
        # Load the PNG image directly.
        pil_image = Image.open(input_file)
        # Convert the PIL image to a NumPy array (to be compatible with later processing).
        images = [np.array(pil_image)]
    else:
        print("Unsupported file format. Please provide a PDF or PNG file.")
        return

    # Initialize the OCR processors.
    ocr_processor = EasyOCRProcessor(languages=['cs'], gpu=True)
    paddle_processor = PaddleOCRProcessor(lang='cs', use_gpu=True)
    pytesseract_processor = PytesseractOCRProcessor(
        tesseract_cmd='/usr/bin/tesseract',
        lang='ces'
    )

    all_layout_texts_pytesseractOCR = []
    all_layout_texts_paddleOCR = []
    all_layout_texts_easyOCR = []

    # Process each image (whether from PDF pages or a single PNG image).
    for idx, image in enumerate(images):
        print(f"Processing image {idx}...")
        
        try:
            deskewer = ImageDeskewer(image)
            rotated_image, median_angle = deskewer.deskew()
            output_path = f"deskewed_image_{idx}.jpg"
            deskewer.save_image(output_path, rotated_image)
            print(f"Image {idx} rotated by {-median_angle:.2f} degrees and saved to {output_path}")
        except ValueError as e:
            print(f"Deskewing error on image {idx}: {e}")
            rotated_image = image  # use original if deskew fails

        # Convert the NumPy array to a PIL image.
        pil_image = Image.fromarray(rotated_image)

        # Run OCR with Pytesseract.
        layout_text_pytesseract = pytesseract_processor.extract_text_layout_from_pil(pil_image, threshold=15)
        print("-" * 40)
        print(f"\nApproximate layout (Pytesseract) for image {idx}:\n{layout_text_pytesseract}")
        print("-" * 40)

        # Run OCR with PaddleOCR.
        layout_text_paddleOCR = paddle_processor.extract_text_layout_from_pil(pil_image, threshold=15)
        print(f"\nApproximate layout (PaddleOCR) for image {idx}:\n{layout_text_paddleOCR}")
        print("-" * 40)

        # Run OCR with EasyOCR.
        layout_text_easyOCR = ocr_processor.image_to_text_layout(pil_image, threshold=30)
        print(f"OCR Text (EasyOCR) for image {idx}:\n{layout_text_easyOCR}")
        print("-" * 40)

        all_layout_texts_pytesseractOCR.append(layout_text_pytesseract)
        all_layout_texts_paddleOCR.append(layout_text_paddleOCR)
        all_layout_texts_easyOCR.append(layout_text_easyOCR)

    combined_layout_text_pytesseractOCR = "\n".join(all_layout_texts_pytesseractOCR)
    combined_layout_text_paddleOCR = "\n".join(all_layout_texts_paddleOCR)
    combined_layout_text_easyOCR = "\n".join(all_layout_texts_easyOCR)

    # Set up interfaces for DeepSeek, Llama, and Mistral.
    load_dotenv("deepinfra_api_key.env")
    interface_deepseek = DeepSeekInterface(api_key=os.getenv("DEEPINFRA_API_KEY")) #Deepseek
    second_model_llm = SecondInterface(api_key=os.getenv("DEEPINFRA_API_KEY")) #llama
    third_model_llm = ThirdInterface(api_key=os.getenv("DEEPINFRA_API_KEY")) #Qwen/QwQ-32B

    # Create prompts for each model based on their respective OCR outputs.
    prompt_content_deepseek = get_prompt(file_name, combined_layout_text_pytesseractOCR)
    prompt_deepseek = [{"role": "user", "content": prompt_content_deepseek, "temperature": 0.0}]

    prompt_content_second_model = get_prompt(file_name, combined_layout_text_paddleOCR)
    prompt_second_model = [{"role": "user", "content": prompt_content_second_model, "temperature": 0.0}]

    prompt_content_third_model = get_prompt(file_name, combined_layout_text_easyOCR)
    prompt_third_model = [{"role": "user", "content": prompt_content_third_model, "temperature": 0.0}]

    # Get responses from each model.
    response_deepseek = interface_deepseek.get_chat_completion(prompt_deepseek)
    deepseek_output = response_deepseek.choices[0].message.content

    response_second_model = second_model_llm.get_chat_completion(prompt_second_model)
    second_model_output = response_second_model.choices[0].message.content

    response_third_model = third_model_llm.get_chat_completion(prompt_third_model)
    third_model_output = response_third_model.choices[0].message.content

    print("\nDeepSeek Response Output:\n", deepseek_output)
    print("\nSecond Model (LLM) Response Output:\n", second_model_output)
    print("\nThird Model (LLM) Response Output:\n", third_model_output)

    # --- Calculate cumulative token usage and cost for each model ---

    # Calculate tokens for DeepSeek
    deepseek_prompt_tokens = response_deepseek.usage.prompt_tokens
    deepseek_completion_tokens = response_deepseek.usage.completion_tokens
    deepseek_total_tokens = deepseek_prompt_tokens + deepseek_completion_tokens

    # Calculate tokens for Second Model (LLM)
    second_prompt_tokens = response_second_model.usage.prompt_tokens
    second_completion_tokens = response_second_model.usage.completion_tokens
    second_total_tokens = second_prompt_tokens + second_completion_tokens

    # Calculate tokens for Third Model (LLM)
    third_prompt_tokens = response_third_model.usage.prompt_tokens
    third_completion_tokens = response_third_model.usage.completion_tokens
    third_total_tokens = third_prompt_tokens + third_completion_tokens

    # Cumulative totals
    cumulative_prompt_tokens = deepseek_prompt_tokens + second_prompt_tokens + third_prompt_tokens
    cumulative_completion_tokens = deepseek_completion_tokens + second_completion_tokens + third_completion_tokens
    cumulative_total_tokens = deepseek_total_tokens + second_total_tokens + third_total_tokens

    print("\n--- Token Usage Summary ---")
    print(f"DeepSeek Tokens: {deepseek_total_tokens} (Prompt: {deepseek_prompt_tokens}, Completion: {deepseek_completion_tokens})")
    print(f"Second Model Tokens: {second_total_tokens} (Prompt: {second_prompt_tokens}, Completion: {second_completion_tokens})")
    print(f"Third Model Tokens: {third_total_tokens} (Prompt: {third_prompt_tokens}, Completion: {third_completion_tokens})")
    print("-" * 40)
    print(f"Cumulative Tokens -> Prompt: {cumulative_prompt_tokens} | Completion: {cumulative_completion_tokens} | Total: {cumulative_total_tokens}")

    # Calculate cost for each model
    deepseek_cost = response_deepseek.usage.estimated_cost
    second_cost = response_second_model.usage.estimated_cost
    third_cost = response_third_model.usage.estimated_cost

    # Cumulative estimated cost
    cumulative_cost = deepseek_cost + second_cost + third_cost

    print("\n--- Cost Summary ---")
    print(f"DeepSeek Estimated Cost: ${deepseek_cost:.8f}")
    print(f"Second Model Estimated Cost: ${second_cost:.8f}")
    print(f"Third Model Estimated Cost: ${third_cost:.8f}")
    print("-" * 40)
    print(f"Cumulative Estimated Cost: ${cumulative_cost:.8f}")
    print(f"Approximately: {cumulative_cost * 0.91:.8f} €")  # USD to EUR conversion


    # --- Extracting JSON data from each model's output ---
    try:
        invoice_data_deepseek = JSONExtractor.extract_json(deepseek_output)
        print("\nExtracted JSON from DeepSeek:")
        print(json.dumps(invoice_data_deepseek, indent=4, ensure_ascii=False))
    except ValueError as e:
        print("Failed to extract JSON from DeepSeek output:", e)

    try:
        invoice_data_second_model = JSONExtractor.extract_json(second_model_output)
        print("\nExtracted JSON from Second Model (LLM):")
        print(json.dumps(invoice_data_second_model, indent=4, ensure_ascii=False))
    except ValueError as e:
        print("Failed to extract JSON from Second Model output:", e)

    try:
        invoice_data_third_model = JSONExtractor.extract_json(third_model_output)
        print("\nExtracted JSON from Third Model (LLM):")
        print(json.dumps(invoice_data_third_model, indent=4, ensure_ascii=False))
    except ValueError as e:
        print("Failed to extract JSON from Third Model output:", e)

    # --- Apply Triple Modular Redundancy ---
    try:
        final_invoice_data = triple_modular_redundancy(invoice_data_deepseek, invoice_data_second_model, invoice_data_third_model)
        print("\nFinal Invoice Data after Triple Modular Redundancy:")
        print(json.dumps(final_invoice_data, indent=4, ensure_ascii=False))
    except Exception as e:
        print("Failed to merge invoice data using triple modular redundancy:", e)

    # --- Insert the final invoice data into the database ---
    try:
        db = InvoiceDB()
        invoice_id = db.insert_invoice(final_invoice_data)
        print("Inserted invoice with ID:", invoice_id)
        db.close()
    except Exception as e:
        print("Failed to insert invoice into database:", e)

if __name__ == '__main__':
    main()
