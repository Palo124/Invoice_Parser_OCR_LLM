# Invoice Parser

Full-stack prototype for extracting structured data from Czech invoices (PDF/PNG/JPG). Built for thesis experiments and evaluation.

## Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI, SQLite |
| Frontend | React (Vite) |
| Text extraction | PyMuPDF, OCRmyPDF, Tesseract, PaddleOCR |
| LLM (primary) | DeepSeek-V4-Flash via DeepInfra |
| Vision fallback | Qwen3-VL-30B via DeepInfra |
| Escalation | DeepSeek-V4-Pro via DeepInfra |

## Project layout

```
backend/          # FastAPI API + processing pipeline
frontend/         # React UI (upload, list, review)
backend/configs/  # Pipeline thresholds and model names
```

## Setup

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set DEEPINFRA_API_KEY
uvicorn app.main:app --reload --app-dir .
```

API: `http://127.0.0.1:8000`

System dependencies: Tesseract (`/usr/bin/tesseract` by default), Poppler (for `pdf2image`).

### Frontend

```bash
cd frontend
npm install
npm run dev
```

UI: `http://localhost:5173`

## Pipeline decision tree

The orchestrator runs a single modern pipeline with conditional branches. Digital PDFs skip OCR and vision unless validation triggers escalation.

```
Upload (PDF / PNG / JPG)
│
├─ TEXT EXTRACTION
│   ├─ PDF with usable PyMuPDF text layer
│   │     → branch: digital_pdf
│   │     → path: modern:pymupdf_digital
│   │     → OCR skipped
│   │
│   ├─ PDF without usable text (scan)
│   │     → branch: ocr_scan
│   │     → OCRmyPDF/Tesseract + PaddleOCR, compare, pick longer text
│   │     → path: modern:ocr_tesseract_paddle
│   │
│   └─ Image file
│         → branch: ocr_image
│         → deskew → Tesseract + PaddleOCR, compare
│         → path: modern:ocr_tesseract_paddle
│
├─ PRIMARY LLM (DeepSeek-V4-Flash)
│     → structured JSON extraction from text
│
├─ VISION FALLBACK (only for OCR branches, if triggered)
│   Triggers (any):
│     • OCR engines disagree (low agreement)
│     • Required fields missing in LLM output
│     • Low text density per page
│     • Multi-page document
│     • Dense numeric table layout
│   → Qwen3-VL on page images
│   → Merge: vision wins items/totals; text wins headers/parties
│
├─ VALIDATION
│     • Schema (Pydantic)
│     • Business rules (IČO checksum, dates, totals arithmetic)
│     • LLM self-check (numbers present in source text)
│     • OCR conflict flag
│     → confidence: high | medium | low | failed
│     → needs_review when confidence low or critical flags
│
├─ ESCALATION (if triggered, max 1 retry)
│   Triggers (any):
│     • confidence low
│     • totals / invoice ID validation failed
│     • text vs vision disagreement on key fields
│   → DeepSeek-V4-Pro re-extracts flagged fields only
│   → Re-validate
│
└─ PERSIST + optional HUMAN REVIEW (Phase 6)
      → User corrects flagged fields, approves
      → Audit: corrected_fields_json, reviewed_at
```

### Progress stages (UI)

```
text:pymupdf → text:ocr_compare → llm:deepseek → llm:vision → llm:escalation → validation
```

(`text:ocr_compare` is skipped for digital PDFs; vision/escalation stages run only when triggered.)

## Configuration

Edit `backend/configs/default.yaml`:

- `text_extraction` — PyMuPDF quality thresholds, OCR agreement
- `ocr` — Tesseract/Paddle languages, DPI
- `llm.primary_model` — primary extractor
- `vision` — enable/disable, model, page limits, layout triggers
- `escalation` — enable/disable, model, max retries
- `validation.totals_tolerance` — CZK tolerance for totals check

Environment (`.env`):

- `DEEPINFRA_API_KEY` — required for all LLM calls
- `TESSERACT_CMD` — path to tesseract binary

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Health check |
| GET | `/api/invoices` | List invoices (`?filter=needs_review\|approved`) |
| GET | `/api/invoices/{id}` | Invoice detail + validation flags |
| GET | `/api/invoices/{id}/file` | Original uploaded PDF/image |
| GET | `/api/invoices/{id}/html` | Generated HTML preview |
| POST | `/api/invoices/upload` | Upload and process (multipart) |
| PATCH | `/api/invoices/{id}` | Save user corrections |
| POST | `/api/invoices/{id}/approve` | Approve after review |
| POST | `/api/invoices/{id}/cancel` | Cancel in-flight processing |
| POST | `/api/invoices/{id}/redo` | Re-run extraction |
| DELETE | `/api/invoices/{id}` | Delete record + uploaded file |
| GET | `/api/pipeline/stages` | Display stage labels for UI |

SQLite database: `backend/data/app.db` (created on startup).

## Dependencies

`backend/requirements.txt` lists only packages used by the modern pipeline:

- **API:** fastapi, uvicorn, sqlalchemy, python-multipart, pyyaml, jinja2
- **LLM:** openai (DeepInfra-compatible client)
- **PDF/text:** pymupdf, ocrmypdf, pytesseract, paddleocr, paddlepaddle
- **Images:** Pillow, numpy, opencv-python-headless, pdf2image

## Evaluation notes (thesis)

- **Digital vs scan path:** compare accuracy/latency on born-digital PDFs vs scanned invoices.
- **OCR redundancy:** Tesseract + PaddleOCR agreement correlates with downstream confidence.
- **Vision cost/benefit:** triggered only on OCR branches; measure recall on line items and totals.
- **Escalation:** Pro model used sparingly on flagged fields; compare error rate before/after approval.
- **Human-in-the-loop:** `needs_review` queue + corrections provide ground-truth labels for error analysis.
