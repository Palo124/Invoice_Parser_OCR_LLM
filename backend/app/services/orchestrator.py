from pathlib import Path

from app.config import settings
from app.services.legacy_pipeline import LegacyPipeline
from app.services.modern_pipeline import ModernPipeline
from app.services.types import CancelCheck, PipelineResult, ProgressCallback


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
        return self._run_modern(file_path, on_progress, should_cancel)

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

    def _run_modern(
        self,
        file_path: Path,
        on_progress: ProgressCallback | None = None,
        should_cancel: CancelCheck | None = None,
    ) -> PipelineResult:
        return ModernPipeline().process_file(
            file_path,
            on_progress=on_progress,
            should_cancel=should_cancel,
        )
