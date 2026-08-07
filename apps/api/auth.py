from __future__ import annotations

import os
from typing import Optional

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

_VALID_KEYS: Optional[set] = None


def _load_keys() -> set:
    global _VALID_KEYS
    if _VALID_KEYS is None:
        raw = os.environ.get("KYC_API_KEYS", "")
        _VALID_KEYS = {k.strip() for k in raw.split(",") if k.strip()}
    return _VALID_KEYS


async def require_api_key(api_key: Optional[str] = Security(_api_key_header)) -> str:
    valid_keys = _load_keys()
    if not valid_keys:
        return "anonymous"
    if not api_key or api_key not in valid_keys:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key
