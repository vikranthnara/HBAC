"""Load `.env` from the repo root or current working directory."""

from __future__ import annotations

from pathlib import Path

from hbac.freellmapi_config import bootstrap_freellmapi_env


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_project_env() -> Path | None:
    """Load the first `.env` found (cwd, then repo root). Returns the path loaded."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        bootstrap_freellmapi_env()
        return None

    loaded: Path | None = None
    for root in (Path.cwd(), repo_root()):
        env_path = root / ".env"
        if env_path.is_file():
            load_dotenv(env_path, override=False)
            loaded = env_path
            break
    bootstrap_freellmapi_env()
    return loaded
