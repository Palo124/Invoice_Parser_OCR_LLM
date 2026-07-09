from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class PipelineStepOCR(BaseModel):
    engine: str
    text: str
    char_count: int
    preview: str | None = None


class PipelineStepLLM(BaseModel):
    model: str
    ocr_engine: str
    raw_output: str
    parsed_json: dict[str, Any] | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    structured_output: bool | None = None
    duration_seconds: float | None = None
    estimated_cost: float | None = None


class StepMetricItem(BaseModel):
    stage: str
    duration_seconds: float
    estimated_cost: float
    model: str | None = None


class ValidationErrorItem(BaseModel):
    field: str
    code: str
    message: str
    severity: str


class FlaggedFieldItem(BaseModel):
    field: str
    flag: str
    message: str


class PipelineSteps(BaseModel):
    preprocessing: dict[str, Any] | None = None
    text_extraction: dict[str, Any] | None = None
    ocr: list[PipelineStepOCR] = Field(default_factory=list)
    llm: list[PipelineStepLLM] = Field(default_factory=list)
    tmr: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None
    vision_trigger: dict[str, Any] | None = None
    vision: dict[str, Any] | None = None
    vision_merge: dict[str, Any] | None = None
    escalation_trigger: dict[str, Any] | None = None
    escalation: dict[str, Any] | None = None
    escalation_merge: dict[str, Any] | None = None


class PipelineMetadata(BaseModel):
    pipeline_mode: str | None = None
    progress: dict[str, Any] | None = None
    token_usage: dict[str, Any] | None = None
    total_duration_seconds: float | None = None
    estimated_cost: float | None = None
    step_metrics: list[StepMetricItem] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)
    vision_used: bool | None = None
    vision_triggers: list[str] = Field(default_factory=list)
    escalation_used: bool | None = None
    escalation_triggers: list[str] = Field(default_factory=list)
    steps: PipelineSteps | None = None


class InvoiceSummary(BaseModel):
    id: int
    original_filename: str
    status: str
    invoice_number: str | None
    supplier_name: str | None
    extraction_path: str | None = None
    confidence: str | None = None
    needs_review: bool = False
    flags: list[str] = Field(default_factory=list)
    review_status: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class CorrectedFieldItem(BaseModel):
    from_value: Any = Field(alias="from")
    to: Any
    corrected_at: str | None = None

    model_config = {"populate_by_name": True}


class InvoiceUpdateRequest(BaseModel):
    data: dict[str, Any]


class InvoiceDetail(InvoiceSummary):
    data: dict[str, Any] | None = None
    error_message: str | None = None
    metadata: PipelineMetadata | None = None
    raw_text: str | None = None
    llm_raw_json: str | None = None
    model_used: str | None = None
    flagged_fields: list[FlaggedFieldItem] = Field(default_factory=list)
    validation_errors: list[ValidationErrorItem] = Field(default_factory=list)
    corrected_fields: dict[str, CorrectedFieldItem] = Field(default_factory=dict)
    reviewed_at: datetime | None = None
