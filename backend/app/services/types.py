from dataclasses import dataclass, field
from typing import Any, Callable


ProgressCallback = Callable[[str, dict[str, Any]], None]
CancelCheck = Callable[[], bool]


@dataclass
class TextExtractionResult:
    text: str
    source: str
    confidence: float


@dataclass
class ExtractionResult:
    data: dict
    model: str
    confidence: str = "high"
    warnings: list[str] = field(default_factory=list)
    raw_output: str = ""
    ocr_engine: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass
class PipelineResult:
    data: dict
    extraction_path: str
    confidence: str
    needs_review: bool
    flags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    text_extraction: TextExtractionResult | None = None
    extractions: list[ExtractionResult] = field(default_factory=list)
