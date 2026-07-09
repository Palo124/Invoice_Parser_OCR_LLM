from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.invoices import router as invoices_router
from app.api.pipeline import router as pipeline_router
from app.config import settings
from app.db import init_db

app = FastAPI(title="Invoice Parser API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(invoices_router, prefix="/api")
app.include_router(pipeline_router, prefix="/api")


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/api/health")
def health():
    return {"status": "ok"}
