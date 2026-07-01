"""FreeLLMAPI integration (OpenAI-compatible local proxy)."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path


DEFAULT_BASE_URL = "http://127.0.0.1:3001/v1"


def freellmapi_dir() -> Path | None:
    raw = os.environ.get("HBAC_FREELLMAPI_DIR", "").strip()
    if raw:
        path = Path(raw).expanduser()
        return path if path.is_dir() else None
    return None


def freellmapi_db_path(directory: Path | None = None) -> Path | None:
    root = directory or freellmapi_dir()
    if not root:
        return None
    db = root / "server" / "data" / "freeapi.db"
    return db if db.is_file() else None


def read_unified_api_key(directory: Path | None = None) -> str | None:
    db_path = freellmapi_db_path(directory)
    if not db_path:
        return None
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT value FROM settings WHERE key = 'unified_api_key'"
        ).fetchone()
    finally:
        conn.close()
    if not row or not row[0]:
        return None
    return str(row[0]).strip() or None


def freellmapi_configured() -> bool:
    return bool(
        os.environ.get("FREELLMAPI_BASE_URL", "").strip()
        and os.environ.get("FREELLMAPI_API_KEY", "").strip()
    )


def bootstrap_freellmapi_env() -> bool:
    """Populate FREELLMAPI_* from HBAC_FREELLMAPI_DIR when unset. Returns True if configured."""
    if freellmapi_configured():
        return True

    directory = freellmapi_dir()
    if not directory:
        return False

    key = read_unified_api_key(directory)
    if not key:
        return False

    os.environ.setdefault("FREELLMAPI_API_KEY", key)
    os.environ.setdefault(
        "FREELLMAPI_BASE_URL",
        os.environ.get("HBAC_FREELLMAPI_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL,
    )
    return True
