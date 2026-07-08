import json
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.models.invoice import Invoice
from app.schemas.invoice import InvoiceDetail, InvoiceSummary
from app.services.pipeline import InvoicePipeline

router = APIRouter(prefix="/invoices", tags=["invoices"])


def _to_summary(invoice: Invoice) -> InvoiceSummary:
    return InvoiceSummary.model_validate(invoice)


def _to_detail(invoice: Invoice) -> InvoiceDetail:
    data = json.loads(invoice.data_json) if invoice.data_json else None
    return InvoiceDetail(
        id=invoice.id,
        original_filename=invoice.original_filename,
        status=invoice.status,
        invoice_number=invoice.invoice_number,
        supplier_name=invoice.supplier_name,
        created_at=invoice.created_at,
        data=data,
        error_message=invoice.error_message,
    )


@router.get("", response_model=list[InvoiceSummary])
def list_invoices(db: Session = Depends(get_db)):
    invoices = db.query(Invoice).order_by(Invoice.created_at.desc()).all()
    return [_to_summary(invoice) for invoice in invoices]


@router.get("/{invoice_id}", response_model=InvoiceDetail)
def get_invoice(invoice_id: int, db: Session = Depends(get_db)):
    invoice = db.query(Invoice).filter(Invoice.id == invoice_id).first()
    if not invoice:
        raise HTTPException(status_code=404, detail="Invoice not found")
    return _to_detail(invoice)


@router.post("/upload", response_model=InvoiceDetail)
async def upload_invoice(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    ext = Path(file.filename).suffix.lower()
    if ext not in {".pdf", ".png", ".jpg", ".jpeg"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    invoice = Invoice(
        original_filename=file.filename,
        status="processing",
    )
    db.add(invoice)
    db.commit()
    db.refresh(invoice)

    saved_path = settings.upload_dir / f"{invoice.id}_{file.filename}"
    with saved_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = InvoicePipeline().process_file(saved_path)
        supplier = result.data.get("supplier") or {}
        invoice.status = "completed"
        invoice.invoice_number = result.data.get("invoice_number")
        invoice.supplier_name = supplier.get("name")
        invoice.data_json = json.dumps(result.data, ensure_ascii=False)
    except Exception as exc:
        invoice.status = "failed"
        invoice.error_message = str(exc)

    db.commit()
    db.refresh(invoice)
    return _to_detail(invoice)
