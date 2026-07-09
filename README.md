# Invoice Parser

Full-stack app for extracting structured data from Czech invoices using OCR + LLMs.

## Stack

- **Backend:** Python, FastAPI, SQLite
- **Frontend:** React (Vite)
- **Pipeline (default):** PyMuPDF → OCRmyPDF/Tesseract + PaddleOCR → DeepSeek (DeepInfra)
- **Legacy mode (`LEGACY_PIPELINE=true`):** Tesseract + PaddleOCR → 2× LLM → TMR

## Project layout

```
backend/          # FastAPI API + OCR/LLM pipeline
frontend/         # React UI
```

## Setup

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add DEEPINFRA_API_KEY
uvicorn app.main:app --reload --app-dir .
```

API runs at `http://127.0.0.1:8000`

### Frontend

```bash
cd frontend
npm install
npm run dev
```

UI runs at `http://localhost:5173`

## API

- `GET /api/health`
- `GET /api/invoices`
- `GET /api/invoices/{id}`
- `POST /api/invoices/upload` (multipart file)

SQLite database is created automatically at `backend/data/app.db`.
