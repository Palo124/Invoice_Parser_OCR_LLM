from pathlib import Path

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
        return ModernPipeline().process_file(
            file_path,
            on_progress=on_progress,
            should_cancel=should_cancel,
        )
