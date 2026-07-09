from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidationError:
    field: str
    code: str
    message: str
    severity: str = "error"

    def to_dict(self) -> dict[str, str]:
        return {
            "field": self.field,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
        }


def flagged_fields_from_errors(errors: list[ValidationError]) -> list[dict[str, str]]:
    return [
        {
            "field": error.field,
            "flag": error.code,
            "message": error.message,
        }
        for error in errors
    ]


def collect_flags(errors: list[ValidationError]) -> list[str]:
    codes = {error.code for error in errors}
    flags: list[str] = []
    if "totals_mismatch" in codes:
        flags.append("totals_mismatch")
    if "required_field" in codes:
        flags.append("required_field_missing")
    if codes.intersection({"invalid_ico", "invalid_ico_checksum"}):
        flags.append("invalid_party_id")
    if "ocr_conflict" in codes:
        flags.append("ocr_conflict")
    if "tmr_disagreement" in codes:
        flags.append("tmr_disagreement")
    if "llm_number_mismatch" in codes:
        flags.append("llm_number_mismatch")
    if "schema_invalid" in codes:
        flags.append("schema_invalid")
    return flags


@dataclass
class ValidationResult:
    confidence: str
    needs_review: bool
    errors: list[ValidationError] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    review_status: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "confidence": self.confidence,
            "needs_review": self.needs_review,
            "flags": self.flags,
            "flagged_fields": flagged_fields_from_errors(self.errors),
            "validation_errors": [error.to_dict() for error in self.errors],
            "review_status": self.review_status,
            "metadata": self.metadata,
        }
