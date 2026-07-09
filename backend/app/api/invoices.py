import json
import shutil
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.models.invoice import Invoice
from app.schemas.invoice import InvoiceDetail, InvoiceSummary, PipelineMetadata
from app.services.generation.html_renderer import render_invoice_html
from app.services.invoice_processing import (
    begin_processing,
    cancel_invoice_processing,
    reset_for_reprocess,
    run_invoice_job,
)
from app.services.validation import flagged_fields_from_errors
from app.services.validation.types import ValidationError

router = APIRouter(prefix="/invoices", tags=["invoices"])


def _parse_json_list(raw: str | None) -> list:
    if not raw:
        return []
    return json.loads(raw)


def _validation_errors_for(invoice: Invoice) -> list[dict]:
    return _parse_json_list(invoice.validation_errors_json)


def _flagged_fields_for(invoice: Invoice) -> list[dict]:
    return flagged_fields_from_errors(
        [ValidationError(**item) for item in _validation_errors_for(invoice)]
    )


def _build_metadata(invoice: Invoice) -> PipelineMetadata | None:
    if not invoice.metadata_json:
        return None

    payload = json.loads(invoice.metadata_json)
    return PipelineMetadata.model_validate(payload)


def _to_summary(invoice: Invoice) -> InvoiceSummary:
    return InvoiceSummary(
        id=invoice.id,
        original_filename=invoice.original_filename,
        status=invoice.status,
        invoice_number=invoice.invoice_number,
        supplier_name=invoice.supplier_name,
        extraction_path=invoice.extraction_path,
        confidence=invoice.confidence,
        needs_review=invoice.needs_review,
        flags=_parse_json_list(invoice.flags_json),
        review_status=invoice.review_status,
        created_at=invoice.created_at,
    )


def _to_detail(invoice: Invoice) -> InvoiceDetail:
    data = json.loads(invoice.data_json) if invoice.data_json else None
    metadata = _build_metadata(invoice)
    if metadata and invoice.extraction_path:
        metadata.pipeline_mode = metadata.pipeline_mode or invoice.extraction_path

    validation_errors = _validation_errors_for(invoice)

    return InvoiceDetail(
        **_to_summary(invoice).model_dump(),
        data=data,
        error_message=invoice.error_message,
        metadata=metadata,
        raw_text=invoice.raw_text,
        llm_raw_json=invoice.llm_raw_json,
        model_used=invoice.model_used,
        flagged_fields=_flagged_fields_for(invoice),
        validation_errors=validation_errors,
    )


def _get_invoice_or_404(invoice_id: int, db: Session) -> Invoice:
    invoice = db.query(Invoice).filter(Invoice.id == invoice_id).first()
    if not invoice:
        raise HTTPException(status_code=404, detail="Invoice not found")
    return invoice


def _upload_path_for(invoice: Invoice) -> Path:
    exact = settings.upload_dir / f"{invoice.id}_{invoice.original_filename}"
    if exact.exists():
        return exact

    matches = list(settings.upload_dir.glob(f"{invoice.id}_*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Original file not found for this invoice")
    if len(matches) == 1:
        return matches[0]

    for path in matches:
        if path.name == exact.name:
            return path
    return matches[0]


@router.get("", response_model=list[InvoiceSummary])
def list_invoices(db: Session = Depends(get_db)):
    invoices = db.query(Invoice).order_by(Invoice.created_at.desc()).all()
    return [_to_summary(invoice) for invoice in invoices]


@router.get("/{invoice_id}/html", response_class=HTMLResponse)
def get_invoice_html(invoice_id: int, db: Session = Depends(get_db)):
    invoice = _get_invoice_or_404(invoice_id, db)
    if not invoice.data_json:
        raise HTTPException(status_code=400, detail="Invoice has no extracted data")

    data = json.loads(invoice.data_json)
    data.setdefault("original_filename", invoice.original_filename)
    return HTMLResponse(content=render_invoice_html(data))


@router.get("/{invoice_id}", response_model=InvoiceDetail)
def get_invoice(invoice_id: int, db: Session = Depends(get_db)):
    return _to_detail(_get_invoice_or_404(invoice_id, db))


@router.post("/{invoice_id}/cancel", response_model=InvoiceDetail)
def cancel_invoice(invoice_id: int, db: Session = Depends(get_db)):
    invoice = _get_invoice_or_404(invoice_id, db)
    if invoice.status != "processing":
        raise HTTPException(status_code=409, detail="Invoice is not processing")

    cancelled = cancel_invoice_processing(db, invoice_id)
    if not cancelled:
        raise HTTPException(status_code=404, detail="Invoice not found")
    return _to_detail(cancelled)


@router.post("/{invoice_id}/redo", response_model=InvoiceDetail)
def redo_invoice(
    invoice_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    invoice = _get_invoice_or_404(invoice_id, db)
    if invoice.status == "processing":
        raise HTTPException(status_code=409, detail="Invoice is already processing")

    saved_path = _upload_path_for(invoice)
    reset_for_reprocess(invoice)
    db.commit()
    db.refresh(invoice)

    background_tasks.add_task(run_invoice_job, invoice.id, saved_path)
    return _to_detail(invoice)


@router.post("/upload", response_model=InvoiceDetail)
async def upload_invoice(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    ext = Path(file.filename).suffix.lower()
    if ext not in {".pdf", ".png", ".jpg", ".jpeg"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    invoice = Invoice(original_filename=file.filename)
    db.add(invoice)
    db.commit()
    db.refresh(invoice)

    saved_path = settings.upload_dir / f"{invoice.id}_{file.filename}"
    with saved_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    begin_processing(invoice)
    db.commit()
    db.refresh(invoice)

    background_tasks.add_task(run_invoice_job, invoice.id, saved_path)
    return _to_detail(invoice)
