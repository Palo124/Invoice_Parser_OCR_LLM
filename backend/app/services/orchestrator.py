from pathlib import Path

from app.config import settings
from app.services.legacy_pipeline import LegacyPipeline
from app.services.types import PipelineResult


class InvoiceOrchestrator:
    """Decision-tree entry point for invoice processing."""

    def process_file(self, file_path: Path) -> PipelineResult:
        if settings.legacy_pipeline:
            return self._run_legacy(file_path)
        return self._run_modern(file_path)

    def _run_legacy(self, file_path: Path) -> PipelineResult:
        return LegacyPipeline().process_file(file_path)

    def _run_modern(self, file_path: Path) -> PipelineResult:
        raise NotImplementedError(
            "Modern pipeline is not implemented yet. Set LEGACY_PIPELINE=true "
            "to use the legacy 3x OCR + 3x LLM path."
        )
