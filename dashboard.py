import streamlit as st
import torch
import json
import os
import numpy as np
import tempfile
import base64

from pdf_converter import PDFToImageConverter
from easyocr_processor import EasyOCRProcessor
from deepInfra_deepseek import DeepSeekInterface
from deepInfra_second_model import SecondInterface
from deepInfra_third_model import ThirdInterface
from invoice_db import InvoiceDB  # Database insertion removed for now
from dotenv import load_dotenv
from json_extractor import JSONExtractor
from prompt import get_deepseek_prompt
from picture_rotation import ImageDeskewer
from pytesseract_ocr_processor import PytesseractOCRProcessor
from PIL import Image
from paddle_ocr_processor import PaddleOCRProcessor
from tmr import triple_modular_redundancy

# Set page configuration to use the full width of the screen.
st.set_page_config(page_title="Invoice Dashboard", layout="wide")

# Clear GPU cache and configure PyTorch.
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def process_invoice(file_path, file_ext, file_name):
    """Processes an invoice file (PDF or PNG) and returns a dictionary of results."""
    images = []

    # Convert PDF pages to images or load the image directly.
    if file_ext == ".pdf":
        converter = PDFToImageConverter(poppler_path=None)
        images = converter.convert_pdf_to_images(file_path, dpi=300)
    elif file_ext == ".png":
        pil_image = Image.open(file_path)
        images = [np.array(pil_image)]
    else:
        st.error("Unsupported file format. Please provide a PDF or PNG file.")
        return {}

    # Initialize OCR processors.
    ocr_processor = EasyOCRProcessor(languages=['cs'], gpu=True)
    paddle_processor = PaddleOCRProcessor(lang='cs', use_gpu=False)
    pytesseract_processor = PytesseractOCRProcessor(
        tesseract_cmd='/usr/bin/tesseract',
        lang='ces'
    )

    all_layout_texts_pytesseract = []
    all_layout_texts_paddle = []
    all_layout_texts_easy = []

    # Process every image (from PDF pages or a single PNG).
    for idx, image in enumerate(images):
        st.write(f"Processing image {idx}...")
        try:
            deskewer = ImageDeskewer(image)
            rotated_image, median_angle = deskewer.deskew()
            temp_out = f"deskewed_image_{idx}.jpg"
            deskewer.save_image(temp_out, rotated_image)
            st.write(f"Image {idx} rotated by {-median_angle:.2f}° (saved as {temp_out}).")
        except ValueError as e:
            st.warning(f"Deskewing error on image {idx}: {e}")
            rotated_image = image

        pil_image = Image.fromarray(rotated_image)
        layout_text_pytesseract = pytesseract_processor.extract_text_layout_from_pil(pil_image, threshold=15)
        layout_text_paddle = paddle_processor.extract_text_layout_from_pil(pil_image, threshold=15)
        layout_text_easy = ocr_processor.image_to_text_layout(pil_image, threshold=30)

        st.markdown(f"**Pytesseract OCR (Image {idx}):**")
        st.text(layout_text_pytesseract)
        st.markdown(f"**PaddleOCR (Image {idx}):**")
        st.text(layout_text_paddle)
        st.markdown(f"**EasyOCR (Image {idx}):**")
        st.text(layout_text_easy)

        all_layout_texts_pytesseract.append(layout_text_pytesseract)
        all_layout_texts_paddle.append(layout_text_paddle)
        all_layout_texts_easy.append(layout_text_easy)

    # Combine OCR texts from all pages.
    combined_pytesseract = "\n".join(all_layout_texts_pytesseract)
    combined_paddle = "\n".join(all_layout_texts_paddle)
    combined_easy = "\n".join(all_layout_texts_easy)

    # --- Set up model interfaces ---
    load_dotenv("deepinfra_api_key.env")
    deepseek_api_key = os.getenv("DEEPINFRA_API_KEY")
    if not deepseek_api_key:
        st.error("DeepInfra API key not found in environment file.")
        return {}
    interface_deepseek = DeepSeekInterface(api_key=deepseek_api_key)
    second_model_llm = SecondInterface(api_key=deepseek_api_key)
    third_model_llm = ThirdInterface(api_key=deepseek_api_key)

    # Create prompts for each model.
    prompt_deepseek = [{"role": "user", "content": get_deepseek_prompt(file_name, combined_pytesseract), "temperature": 0.0}]
    prompt_second = [{"role": "user", "content": get_deepseek_prompt(file_name, combined_paddle), "temperature": 0.0}]
    prompt_third = [{"role": "user", "content": get_deepseek_prompt(file_name, combined_easy), "temperature": 0.0}]

    # Get responses from each model.
    response_deepseek = interface_deepseek.get_chat_completion(prompt_deepseek)
    deepseek_output = response_deepseek.choices[0].message.content

    response_second = second_model_llm.get_chat_completion(prompt_second)
    second_output = response_second.choices[0].message.content

    response_third = third_model_llm.get_chat_completion(prompt_third)
    third_output = response_third.choices[0].message.content

    # --- Calculate token usage ---
    deepseek_prompt_tokens = response_deepseek.usage.prompt_tokens
    deepseek_completion_tokens = response_deepseek.usage.completion_tokens
    deepseek_total_tokens = deepseek_prompt_tokens + deepseek_completion_tokens

    second_prompt_tokens = response_second.usage.prompt_tokens
    second_completion_tokens = response_second.usage.completion_tokens
    second_total_tokens = second_prompt_tokens + second_completion_tokens

    third_prompt_tokens = response_third.usage.prompt_tokens
    third_completion_tokens = response_third.usage.completion_tokens
    third_total_tokens = third_prompt_tokens + third_completion_tokens

    cumulative_prompt_tokens = deepseek_prompt_tokens + second_prompt_tokens + third_prompt_tokens
    cumulative_completion_tokens = deepseek_completion_tokens + second_completion_tokens + third_completion_tokens
    cumulative_total_tokens = deepseek_total_tokens + second_total_tokens + third_total_tokens

    # --- Calculate estimated costs ---
    deepseek_cost = response_deepseek.usage.estimated_cost
    second_cost = response_second.usage.estimated_cost
    third_cost = response_third.usage.estimated_cost
    cumulative_cost = deepseek_cost + second_cost + third_cost

    # --- Extract JSON data from model outputs ---
    try:
        invoice_data_deepseek = JSONExtractor.extract_json(deepseek_output)
    except ValueError as e:
        st.error(f"JSON extraction failed for DeepSeek output: {e}")
        invoice_data_deepseek = {}

    try:
        invoice_data_second = JSONExtractor.extract_json(second_output)
    except ValueError as e:
        st.error(f"JSON extraction failed for Second Model output: {e}")
        invoice_data_second = {}

    try:
        invoice_data_third = JSONExtractor.extract_json(third_output)
    except ValueError as e:
        st.error(f"JSON extraction failed for Third Model output: {e}")
        invoice_data_third = {}

    # --- Apply Triple Modular Redundancy ---
    try:
        final_invoice_data = triple_modular_redundancy(invoice_data_deepseek, invoice_data_second, invoice_data_third)
    except Exception as e:
        st.error(f"Triple modular redundancy merge failed: {e}")
        final_invoice_data = {}

    # --- Insert invoice data into the database ---
    try:
        db = InvoiceDB()
        invoice_id = db.insert_invoice(final_invoice_data)
        db.close()
    except Exception as e:
        st.error(f"Database insertion failed: {e}")
        invoice_id = None

    # Bundle all results.
    results = {
        "deepseek_output": deepseek_output,
        "second_output": second_output,
        "third_output": third_output,
        "tokens": {
            "deepseek": {"prompt": deepseek_prompt_tokens, "completion": deepseek_completion_tokens, "total": deepseek_total_tokens},
            "second": {"prompt": second_prompt_tokens, "completion": second_completion_tokens, "total": second_total_tokens},
            "third": {"prompt": third_prompt_tokens, "completion": third_completion_tokens, "total": third_total_tokens},
            "cumulative": {"prompt": cumulative_prompt_tokens, "completion": cumulative_completion_tokens, "total": cumulative_total_tokens},
        },
        "cost": {
            "deepseek": deepseek_cost,
            "second": second_cost,
            "third": third_cost,
            "cumulative": cumulative_cost,
        },
        "final_invoice_data": final_invoice_data,
        "invoice_id": invoice_id,
        "ocr_results": {
            "pytesseract": combined_pytesseract,
            "paddle": combined_paddle,
            "easy": combined_easy,
        }
    }

    return results

def display_results(results):
    """Displays the processing results on the dashboard."""
    st.header("Model Responses")
    st.subheader("DeepSeek Output")
    st.text_area("DeepSeek:", results.get("deepseek_output", ""), height=150)
    st.subheader("Second Model (LLM) Output")
    st.text_area("Second Model:", results.get("second_output", ""), height=150)
    st.subheader("Third Model (LLM) Output")
    st.text_area("Third Model:", results.get("third_output", ""), height=150)

    tokens = results.get("tokens", {})
    st.header("Token Usage Summary")
    if tokens:
        st.markdown(f"""
        **DeepSeek Tokens:** Total: {tokens['deepseek']['total']} (Prompt: {tokens['deepseek']['prompt']}, Completion: {tokens['deepseek']['completion']})  
        **Second Model Tokens:** Total: {tokens['second']['total']} (Prompt: {tokens['second']['prompt']}, Completion: {tokens['second']['completion']})  
        **Third Model Tokens:** Total: {tokens['third']['total']} (Prompt: {tokens['third']['prompt']}, Completion: {tokens['third']['completion']})  
        **Cumulative Tokens:** Total: {tokens['cumulative']['total']} (Prompt: {tokens['cumulative']['prompt']}, Completion: {tokens['cumulative']['completion']})
        """)
    else:
        st.write("Token usage data not available.")

    cost = results.get("cost", {})
    st.header("Estimated Cost Summary")
    if cost:
        st.markdown(f"""
        **DeepSeek Estimated Cost:** ${cost.get('deepseek', 0):.8f}  
        **Second Model Estimated Cost:** ${cost.get('second', 0):.8f}  
        **Third Model Estimated Cost:** ${cost.get('third', 0):.8f}  
        **Cumulative Estimated Cost:** ${cost.get('cumulative', 0):.8f} (≈ €{cost.get('cumulative', 0)*0.91:.8f})
        """)
    else:
        st.write("Cost data not available.")

    st.header("Final Invoice Data (JSON)")
    if results.get("final_invoice_data"):
        st.json(results["final_invoice_data"])
    else:
        st.write("No invoice data was extracted.")

    st.header("Database Insertion")
    if results.get("invoice_id"):
        st.success(f"Invoice inserted with ID: {results['invoice_id']}")
    else:
        st.error("Invoice was not inserted into the database.")

def main():
    st.title("Invoice OCR and Processing Dashboard")
    st.sidebar.header("Invoice File Upload")
    uploaded_file = st.sidebar.file_uploader("Choose a PDF or PNG invoice file", type=["pdf", "png"])

    if uploaded_file is not None:
        file_name = uploaded_file.name
        file_ext = os.path.splitext(file_name)[1].lower()

        # Save uploaded file to a temporary location.
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        st.sidebar.success(f"File '{file_name}' uploaded successfully.")

        if file_ext == ".pdf":
            # Reset file pointer and get PDF bytes.
            uploaded_file.seek(0)
            pdf_bytes = uploaded_file.read()
            base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" frameborder="0"></iframe>'

            # Custom CSS for responsive layout.
            st.markdown("""
            <style>
            .pdf-container {
              width: 100%;
              height: calc(100vh - 150px);
            }
            .output-container {
              width: 100%;
              overflow-y: auto;
              height: calc(100vh - 150px);
            }
            </style>
            """, unsafe_allow_html=True)

            # Use a 2:1 column layout.
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("### Original PDF")
                st.markdown(f'<div class="pdf-container">{pdf_display}</div>', unsafe_allow_html=True)
            with col2:
                st.markdown("### Processing Options and Output")
                if st.button("Process Invoice"):
                    with st.spinner("Processing invoice. Please wait..."):
                        results = process_invoice(tmp_file_path, file_ext, file_name)
                    display_results(results)
        else:
            # For non-PDF formats simply process and show output.
            if st.sidebar.button("Process Invoice"):
                with st.spinner("Processing invoice. Please wait..."):
                    results = process_invoice(tmp_file_path, file_ext, file_name)
                display_results(results)
    else:
        st.info("Please upload an invoice file from the sidebar.")

if __name__ == '__main__':
    main()
