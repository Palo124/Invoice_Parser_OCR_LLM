import json
import threading
from pathlib import Path

from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.models.invoice import Invoice
from app.services.exceptions import ProcessingCancelled
from app.services.pipeline import InvoicePipeline

STAGE_LABELS = {
    "queued": "Waiting to start",
    "preprocessing": "Preparing pages",
    "ocr:tesseract": "Running Tesseract OCR",
    "ocr:paddleocr": "Running PaddleOCR",
    "ocr:easyocr": "Running EasyOCR",
    "llm:deepseek": "Extracting with DeepSeek",
    "llm:llama": "Extracting with Llama",
    "llm:maverick": "Extracting with Scout",
    "tmr": "Merging results (TMR)",
    "complete": "Finished",
    "cancelled": "Cancelled",
}

PROGRESS_STAGE_ORDER = list(STAGE_LABELS.keys())
HIDDEN_PROGRESS_STAGES = {"queued", "complete", "cancelled"}


_cancel_lock = threading.Lock()
_cancel_requested: set[int] = set()


def request_cancel(invoice_id: int) -> None:
    with _cancel_lock:
        _cancel_requested.add(invoice_id)


def clear_cancel(invoice_id: int) -> None:
    with _cancel_lock:
        _cancel_requested.discard(invoice_id)


def is_cancel_requested(invoice_id: int) -> bool:
    with _cancel_lock:
        return invoice_id in _cancel_requested


def get_display_stages() -> list[dict[str, str]]:
    return [
        {"id": stage, "label": STAGE_LABELS[stage]}
        for stage in PROGRESS_STAGE_ORDER
        if stage not in HIDDEN_PROGRESS_STAGES
    ]


def begin_processing(invoice: Invoice) -> None:
    invoice.status = "processing"
    invoice.metadata_json = json.dumps(
        {
            "pipeline_mode": "legacy",
            "progress": {"stage": "queued", "label": STAGE_LABELS["queued"]},
            "steps": {},
            "flags": [],
        },
        ensure_ascii=False,
    )


def reset_for_reprocess(invoice: Invoice) -> None:
    invoice.invoice_number = None
    invoice.supplier_name = None
    invoice.data_json = None
    invoice.error_message = None
    invoice.extraction_path = None
    invoice.confidence = None
    invoice.needs_review = False
    begin_processing(invoice)


def apply_pipeline_result(invoice: Invoice, result) -> None:
    supplier = result.data.get("supplier") or {}
    invoice.status = "completed"
    invoice.invoice_number = result.data.get("invoice_number")
    invoice.supplier_name = supplier.get("name")
    invoice.data_json = json.dumps(result.data, ensure_ascii=False)
    invoice.extraction_path = result.extraction_path
    invoice.confidence = result.confidence
    invoice.needs_review = result.needs_review
    invoice.metadata_json = json.dumps(
        {
            **result.metadata,
            "progress": {"stage": "complete", "label": STAGE_LABELS["complete"]},
            "flags": result.flags,
        },
        ensure_ascii=False,
    )


def _save_progress(db: Session, invoice: Invoice, stage: str, steps: dict) -> None:
    metadata = {
        "pipeline_mode": "legacy",
        "progress": {"stage": stage, "label": STAGE_LABELS.get(stage, stage)},
        "steps": steps,
        "flags": [],
    }
    invoice.metadata_json = json.dumps(metadata, ensure_ascii=False)
    db.commit()


def mark_invoice_failed(db: Session, invoice_id: int, message: str) -> None:
    invoice = db.query(Invoice).filter(Invoice.id == invoice_id).first()
    if not invoice:
        return
    invoice.status = "failed"
    invoice.error_message = message
    invoice.confidence = "failed"
    invoice.needs_review = True
    db.commit()


def mark_invoice_cancelled(
    db: Session,
    invoice_id: int,
    message: str = "Processing cancelled by user",
) -> Invoice | None:
    invoice = db.query(Invoice).filter(Invoice.id == invoice_id).first()
    if not invoice:
        return None

    metadata = {}
    if invoice.metadata_json:
        metadata = json.loads(invoice.metadata_json)
    metadata["progress"] = {
        "stage": "cancelled",
        "label": STAGE_LABELS["cancelled"],
    }

    invoice.status = "cancelled"
    invoice.error_message = message
    invoice.confidence = None
    invoice.metadata_json = json.dumps(metadata, ensure_ascii=False)
    db.commit()
    db.refresh(invoice)
    return invoice


def cancel_invoice_processing(db: Session, invoice_id: int) -> Invoice | None:
    invoice = db.query(Invoice).filter(Invoice.id == invoice_id).first()
    if not invoice:
        return None
    if invoice.status != "processing":
        return invoice

    request_cancel(invoice_id)
    return mark_invoice_cancelled(db, invoice_id)


def run_invoice_job(invoice_id: int, file_path: Path) -> None:
    db = SessionLocal()
    clear_cancel(invoice_id)
    try:
        invoice = db.query(Invoice).filter(Invoice.id == invoice_id).first()
        if not invoice or invoice.status != "processing":
            return

        def should_cancel() -> bool:
            if is_cancel_requested(invoice_id):
                return True
            db.refresh(invoice)
            return invoice.status != "processing"

        def on_progress(stage: str, steps: dict) -> None:
            if should_cancel():
                raise ProcessingCancelled()
            _save_progress(db, invoice, stage, steps)

        result = InvoicePipeline().process_file(
            file_path,
            on_progress=on_progress,
            should_cancel=should_cancel,
        )
        if should_cancel():
            raise ProcessingCancelled()

        db.refresh(invoice)
        apply_pipeline_result(invoice, result)
        db.commit()
    except ProcessingCancelled:
        mark_invoice_cancelled(db, invoice_id)
    except Exception as exc:
        mark_invoice_failed(db, invoice_id, str(exc))
    finally:
        clear_cancel(invoice_id)
        db.close()
