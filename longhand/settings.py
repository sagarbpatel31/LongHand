"""Environment / configuration loading.

Loads `.env` at import time (via python-dotenv) so the API key is available
without any explicit wiring. The key is never committed — `.env` is gitignored.
"""

from __future__ import annotations

import os

try:  # dotenv is a runtime dep, but degrade gracefully if it is missing
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is declared in pyproject
    pass

API_KEY_ENV = "ASSEMBLYAI_API_KEY"


def get_api_key() -> str | None:
    """Return the raw AssemblyAI key, or None if unset.

    Note: this key is sent as the `Authorization` header value verbatim —
    there is NO `Bearer` prefix. See stt/client.py.
    """
    key = os.environ.get(API_KEY_ENV)
    return key.strip() if key else None
