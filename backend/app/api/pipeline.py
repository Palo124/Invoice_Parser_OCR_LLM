from fastapi import APIRouter

from app.services.invoice_processing import get_display_stages

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


@router.get("/stages")
def list_pipeline_stages():
    return {"stages": get_display_stages()}
