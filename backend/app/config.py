import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "app.db"

load_dotenv(BASE_DIR / ".env")


class Settings:
    deepinfra_api_key: str = os.getenv("DEEPINFRA_API_KEY", "")
    database_url: str = f"sqlite:///{DB_PATH}"
    upload_dir: Path = UPLOAD_DIR
    tesseract_cmd: str = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]


settings = Settings()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
