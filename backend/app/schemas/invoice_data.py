from pydantic import BaseModel, Field


class Party(BaseModel):
    name: str | None = None
    ico: str | None = None
    dic: str | None = None
    address: str | None = None


class Supplier(Party):
    platce_dph: bool | None = None


class InvoiceItem(BaseModel):
    index: int | None = None
    description: str | None = None
    quantity: float | None = None
    unit: str | None = None
    unit_price: float | None = None
    tax_rate: float | None = None
    net_amount: float | None = None
    tax_amount: float | None = None
    gross_amount: float | None = None


class InvoiceData(BaseModel):
    id: str | None = None
    original_filename: str | None = None
    supplier: Supplier | None = None
    customer: Party | None = None
    invoice_number: str | None = None
    invoice_date: str | None = None
    tax_date: str | None = None
    due_date: str | None = None
    variable_symbol: str | None = None
    specific_symbol: str | None = None
    iban: str | None = None
    swift: str | None = None
    items: list[InvoiceItem] = Field(default_factory=list)
    net_total: float | None = None
    tax_total: float | None = None
    gross_total: float | None = None
    notes: str | None = None
    contact: str | None = None
    payment_method: str | None = None
    source: str | None = None

    def to_storage_dict(self) -> dict:
        return self.model_dump(mode="json", exclude_none=False)
