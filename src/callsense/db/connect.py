"""Database connection. Defaults to the local Postgres in docker-compose.yml."""

import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

DEFAULT_URL = "postgresql://callsense:callsense-local@127.0.0.1:5433/callsense"
SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def database_url() -> str:
    load_dotenv()
    return os.environ.get("CALLSENSE_DB_URL", DEFAULT_URL)


def connect(url: str | None = None) -> psycopg.Connection:
    """Open a connection with the schema applied and the vector type registered."""
    conn = psycopg.connect(url or database_url(), autocommit=True)
    conn.execute(SCHEMA_PATH.read_text())
    register_vector(conn)
    return conn
