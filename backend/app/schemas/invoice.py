from datetime import datetime
from typing import Any

from pydantic import BaseModel


class InvoiceSummary(BaseModel):
    id: int
    original_filename: str
    status: str
    invoice_number: str | None
    supplier_name: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class InvoiceDetail(InvoiceSummary):
    data: dict[str, Any] | None = None
    error_message: str | None = None
