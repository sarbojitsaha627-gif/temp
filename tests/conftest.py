import os
import sqlalchemy
import sqlalchemy.dialects.postgresql

# --- Override DATABASE_URL before app imports ---
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"

# --- Monkeypatch Postgres types for SQLite testing ---
sqlalchemy.dialects.postgresql.JSONB = sqlalchemy.JSON
sqlalchemy.dialects.postgresql.UUID = sqlalchemy.Uuid
# -----------------------------------------------------
