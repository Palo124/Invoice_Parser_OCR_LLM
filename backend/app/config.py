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
_text = _yaml.get("text_extraction", {})

_validation = _yaml.get("validation", {})
_vision = _yaml.get("vision", {})
_escalation = _yaml.get("escalation", {})


class Settings:
    deepinfra_api_key: str = os.getenv("DEEPINFRA_API_KEY", "")
    database_url: str = f"sqlite:///{DB_PATH}"
    upload_dir: Path = UPLOAD_DIR
    tesseract_cmd: str = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    legacy_pipeline: bool = _env_bool("LEGACY_PIPELINE", False)

    legacy_extraction_path: str = _pipeline.get(
        "legacy_extraction_path",
        "legacy:triple_ocr_triple_llm_tmr",
    )
    default_confidence: str = _pipeline.get("default_confidence", "high")

    text_min_chars: int = _text.get("min_chars", 80)
    text_max_garbage_ratio: float = _text.get("max_garbage_ratio", 0.35)
    text_require_czech_signal: bool = _text.get("require_czech_signal", True)
    ocr_agreement_threshold: float = _text.get("ocr_agreement_threshold", 0.75)
    ocrmypdf_language: str = _text.get("ocrmypdf_language", "ces+eng")

    ocr_paddle_lang: str = _ocr.get("paddle_lang", "cs")
    ocr_tesseract_lang: str = _ocr.get("tesseract_lang", "ces")
    ocr_paddle_gpu: bool = _ocr.get("paddle_gpu", True)
    ocr_tesseract_threshold: int = _ocr.get("tesseract_threshold", 15)
    ocr_paddle_threshold: int = _ocr.get("paddle_threshold", 15)
    ocr_pdf_dpi: int = _ocr.get("pdf_dpi", 300)

    llm_primary_model: str = _llm.get("primary_model", _llm.get("deepseek_model", "deepseek-ai/DeepSeek-V4-Flash"))
    llm_use_structured_output: bool = _llm.get("use_structured_output", True)
    llm_temperature: float = float(_llm.get("temperature", 0.0))

    llm_deepseek_model: str = _llm.get("deepseek_model", llm_primary_model)
    llm_llama_model: str = _llm.get(
        "llama_model",
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    )
    llm_maverick_model: str = _llm.get(
        "maverick_model",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    )

    validation_totals_tolerance: float = float(_validation.get("totals_tolerance", 1.0))

    vision_enabled: bool = _vision.get("enabled", True)
    llm_vision_model: str = _vision.get(
        "model",
        "Qwen/Qwen3-VL-30B-A3B-Instruct",
    )
    vision_max_pages: int = int(_vision.get("max_pages", 4))
    vision_max_tokens: int = int(_vision.get("max_tokens", 4096))
    vision_min_chars_per_page: int = int(_vision.get("min_chars_per_page", 120))
    vision_min_numeric_columns: int = int(_vision.get("min_numeric_columns", 3))
    vision_min_numeric_lines: int = int(_vision.get("min_numeric_lines", 4))

    escalation_enabled: bool = _escalation.get("enabled", True)
    llm_escalation_model: str = _escalation.get(
        "model",
        "deepseek-ai/DeepSeek-V4-Pro",
    )
    escalation_max_retries: int = int(_escalation.get("max_retries", 1))


settings = Settings()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
