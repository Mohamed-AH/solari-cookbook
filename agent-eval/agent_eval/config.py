"""Configuration: API key resolution and service constants.

The key is read from the environment. A missing key is a hard error with a
message that says what to do about it — never a silent fallback to a run that
would look like it worked.
"""

from __future__ import annotations

import os
from pathlib import Path

# The standalone SandboxClient requires base_url; only the umbrella
# SolariClient in @solarisdk/sdk defaults it.
BASE_URL = "https://api.getsolari.com"

API_KEY_ENV = "SOLARI_API_KEY"


class MissingApiKey(RuntimeError):
    """Raised when no Solari API key is available."""


def load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines from a .env file into os.environ.

    Existing environment variables win, so CI secrets are never shadowed by a
    stale file on disk. Deliberately minimal: no interpolation, no export
    syntax, no dependency.
    """
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def api_key() -> str:
    """Return the Solari API key, or raise with an actionable message."""
    key = os.environ.get(API_KEY_ENV, "").strip()
    if not key:
        raise MissingApiKey(
            f"{API_KEY_ENV} is not set.\n"
            f"  export {API_KEY_ENV}=slr_live_...   # https://console.getsolari.com\n"
            f"  ...or copy .env.example to .env and fill it in."
        )
    return key
