#  Batch Invoice Processor

This project automates the extraction of information from scanned paper invoices using a pipeline of OCR engines, large language models (LLMs), and database storage. It's designed for efficient, large-scale batch processing of Czech-language invoices in PDF or PNG formats.

##  Features

-  Converts PDF or image invoices into text using multiple OCR engines:
  - EasyOCR
  - PaddleOCR
  - Tesseract (optional)
-  Extracts structured data from OCR text using 3 different LLMs:
  - DeepSeek
  - LLaMA (SecondInterface)
  - Qwen (ThirdInterface)
-  Applies Triple Modular Redundancy (TMR) to merge LLM results
-  Inserts clean, validated invoice data into a PostgreSQL database
-  Tracks runtime, token usage, and cost estimates
-  Moves processed files to an archive folder to avoid re-processing

## Tech Stack

- **Python 3.10+**
- **Torch** – GPU memory handling
- **OCR** – EasyOCR, PaddleOCR, Tesseract
- **LLMs via DeepInfra API** – For structured JSON extraction
- **PostgreSQL** – Invoice data storage
- **Dotenv** – For secure API key handling

## Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt

## This project is still under Construction.
