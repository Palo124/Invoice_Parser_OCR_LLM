from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    pass


engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _migrate_invoices_table() -> None:
    inspector = inspect(engine)
    if "invoices" not in inspector.get_table_names():
        return

    existing = {column["name"] for column in inspector.get_columns("invoices")}
    migrations = {
        "extraction_path": "ALTER TABLE invoices ADD COLUMN extraction_path VARCHAR(128)",
        "confidence": "ALTER TABLE invoices ADD COLUMN confidence VARCHAR(32)",
        "needs_review": "ALTER TABLE invoices ADD COLUMN needs_review BOOLEAN DEFAULT 0",
        "metadata_json": "ALTER TABLE invoices ADD COLUMN metadata_json TEXT",
        "raw_text": "ALTER TABLE invoices ADD COLUMN raw_text TEXT",
        "llm_raw_json": "ALTER TABLE invoices ADD COLUMN llm_raw_json TEXT",
        "model_used": "ALTER TABLE invoices ADD COLUMN model_used VARCHAR(128)",
    }

    with engine.begin() as connection:
        for column_name, statement in migrations.items():
            if column_name not in existing:
                connection.execute(text(statement))


def init_db():
    from app.models import invoice  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _migrate_invoices_table()
