from pathlib import Path

from app.services.orchestrator import InvoiceOrchestrator
from app.services.types import PipelineResult


class InvoicePipeline:
    """Backward-compatible facade for the invoice orchestrator."""

    def process_file(self, file_path: Path) -> PipelineResult:
        return InvoiceOrchestrator().process_file(file_path)
