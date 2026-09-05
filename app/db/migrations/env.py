"""
Alembic env.py — configured for Finance Controller models.

Uses synchronous psycopg2 driver for migrations (Alembic doesn't need async).
Imports all ORM models so autogenerate picks up every table.
"""
from __future__ import annotations

import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# ── Ensure project root is on sys.path ──────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

# ── Import Base and all models (so autogenerate sees them) ──
from app.db.engine import Base  # noqa: E402
from app.db import models  # noqa: E402, F401  — side-effect import registers tables

# this is the Alembic Config object
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# target_metadata is what autogenerate diffs against
target_metadata = Base.metadata

# Allow env-var override of sqlalchemy.url
database_url = os.getenv("DATABASE_URL")
if database_url:
    # Convert async URL to sync for Alembic
    sync_url = database_url.replace("+asyncpg", "+psycopg2")
    config.set_main_option("sqlalchemy.url", sync_url)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode — emit SQL to stdout."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode — connect to the database."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
