import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "app.db"
CONFIG_PATH = BASE_DIR / "configs" / "default.yaml"

load_dotenv(BASE_DIR / ".env")


def _load_yaml_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


_yaml = _load_yaml_config()
_pipeline = _yaml.get("pipeline", {})
_ocr = _yaml.get("ocr", {})
_llm = _yaml.get("llm", {})


class Settings:
    deepinfra_api_key: str = os.getenv("DEEPINFRA_API_KEY", "")
    database_url: str = f"sqlite:///{DB_PATH}"
    upload_dir: Path = UPLOAD_DIR
    tesseract_cmd: str = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    legacy_pipeline: bool = _env_bool("LEGACY_PIPELINE", True)

    legacy_extraction_path: str = _pipeline.get(
        "legacy_extraction_path",
        "legacy:triple_ocr_triple_llm_tmr",
    )
    default_confidence: str = _pipeline.get("default_confidence", "high")

    ocr_easyocr_lang: str = _ocr.get("easyocr_lang", "cs")
    ocr_paddle_lang: str = _ocr.get("paddle_lang", "cs")
    ocr_tesseract_lang: str = _ocr.get("tesseract_lang", "ces")
    ocr_easyocr_gpu: bool = _ocr.get("easyocr_gpu", True)
    ocr_paddle_gpu: bool = _ocr.get("paddle_gpu", True)
    ocr_tesseract_threshold: int = _ocr.get("tesseract_threshold", 15)
    ocr_paddle_threshold: int = _ocr.get("paddle_threshold", 15)
    ocr_easyocr_threshold: int = _ocr.get("easyocr_threshold", 30)
    ocr_pdf_dpi: int = _ocr.get("pdf_dpi", 300)

    llm_deepseek_model: str = _llm.get("deepseek_model", "deepseek-ai/DeepSeek-R1")
    llm_llama_model: str = _llm.get("llama_model", "meta-llama/Llama-3.3-70B-Instruct-Turbo")
    llm_maverick_model: str = _llm.get(
        "maverick_model",
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    )


settings = Settings()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
