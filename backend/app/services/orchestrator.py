from pathlib import Path

from app.config import settings
from app.services.legacy_pipeline import LegacyPipeline
from app.services.types import PipelineResult, ProgressCallback, CancelCheck


class InvoiceOrchestrator:
    """Decision-tree entry point for invoice processing."""

    def process_file(
        self,
        file_path: Path,
        on_progress: ProgressCallback | None = None,
        should_cancel: CancelCheck | None = None,
    ) -> PipelineResult:
        if settings.legacy_pipeline:
            return self._run_legacy(file_path, on_progress, should_cancel)
        return self._run_modern(file_path)

    def _run_legacy(
        self,
        file_path: Path,
        on_progress: ProgressCallback | None = None,
        should_cancel: CancelCheck | None = None,
    ) -> PipelineResult:
        return LegacyPipeline().process_file(
            file_path,
            on_progress=on_progress,
            should_cancel=should_cancel,
        )

    def _run_modern(self, file_path: Path) -> PipelineResult:
        raise NotImplementedError(
            "Modern pipeline is not implemented yet. Set LEGACY_PIPELINE=true "
            "to use the legacy 3x OCR + 3x LLM path."
        )
