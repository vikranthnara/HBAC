"""Sync FreeLLMAPI credentials from a local .freellmapi install into HBAC .env."""

from __future__ import annotations

from pathlib import Path

import typer

from hbac.dotenv_loader import repo_root
from hbac.freellmapi_config import DEFAULT_BASE_URL, read_unified_api_key

app = typer.Typer(help="Sync FREELLMAPI_* vars into HBAC .env from SQLite")


def _upsert_env_line(lines: list[str], key: str, value: str) -> list[str]:
    prefix = f"{key}="
    out = [line for line in lines if not line.startswith(prefix)]
    if out and out[-1].strip():
        out.append("")
    out.append(f"{key}={value}")
    return out


@app.command()
def main(
    freellmapi_dir: Path = typer.Option(
        ...,
        "--dir",
        envvar="HBAC_FREELLMAPI_DIR",
        help="Path to paradocs/.freellmapi (or any FreeLLMAPI clone)",
    ),
    env_file: Path = typer.Option(
        repo_root() / ".env",
        "--env-file",
        help="HBAC .env file to update",
    ),
    base_url: str = typer.Option(
        DEFAULT_BASE_URL,
        "--base-url",
        help="FreeLLMAPI OpenAI-compatible base URL",
    ),
) -> None:
    key = read_unified_api_key(freellmapi_dir.expanduser())
    if not key:
        raise typer.Exit(
            "Could not read unified_api_key from "
            f"{freellmapi_dir}/server/data/freeapi.db. Start FreeLLMAPI once first."
        )

    env_path = env_file.expanduser()
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.is_file() else []
    lines = _upsert_env_line(lines, "HBAC_FREELLMAPI_DIR", str(freellmapi_dir.expanduser()))
    lines = _upsert_env_line(lines, "HBAC_LLM_PROVIDER", "freellmapi")
    lines = _upsert_env_line(lines, "HBAC_LLM_MODEL", "auto")
    lines = _upsert_env_line(lines, "FREELLMAPI_BASE_URL", base_url.rstrip("/"))
    lines = _upsert_env_line(lines, "FREELLMAPI_API_KEY", key)
    env_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    typer.echo(f"Updated {env_path}")
    typer.echo("Provider: freellmapi | Model: auto")
    typer.echo(f"Base URL: {base_url.rstrip('/')}")


if __name__ == "__main__":
    app()
