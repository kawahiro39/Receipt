"""Security helpers for request authentication and idempotency tracking."""
from __future__ import annotations

import base64
import hashlib
import hmac
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from fastapi import HTTPException, status

from .settings import Settings, get_settings


class SignatureVerificationError(HTTPException):
    """Raised when a request signature cannot be verified."""

    def __init__(self) -> None:
        super().__init__(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_signature")


@dataclass(frozen=True)
class IdempotentResponse:
    doc_id: str
    payload: Dict[str, object]


class IdempotencyStore:
    """In-memory TTL cache for Idempotency-Key tracking."""

    def __init__(self, ttl_seconds: int = 60 * 10) -> None:
        self._ttl = ttl_seconds
        self._store: Dict[str, Tuple[float, IdempotentResponse]] = {}
        self._lock = threading.Lock()

    def get(self, key: Optional[str]) -> Optional[IdempotentResponse]:
        if not key:
            return None
        now = time.time()
        with self._lock:
            record = self._store.get(key)
            if not record:
                return None
            expires_at, response = record
            if expires_at < now:
                self._store.pop(key, None)
                return None
            return response

    def remember(self, key: Optional[str], response: IdempotentResponse) -> None:
        if not key:
            return
        expires_at = time.time() + self._ttl
        with self._lock:
            self._store[key] = (expires_at, response)


def _extract_bearer(token_header: Optional[str]) -> Optional[str]:
    if not token_header:
        return None
    parts = token_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1]


def verify_admin_token(token_header: Optional[str], settings: Optional[Settings] = None) -> None:
    """Validate an incoming Authorization header for admin endpoints."""

    settings = settings or get_settings()
    token = _extract_bearer(token_header)
    if not token or token != settings.admin_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_admin_token")


def verify_signature(raw_body: bytes, signature_header: Optional[str], settings: Optional[Settings] = None) -> bool:
    """Verify Bubble webhook HMAC signatures.

    Bubble sends signatures in the form ``hmac=<base64>`` or ``sha256=<hex>``.
    The verification secret is configured via ``BUBBLE_SIGNATURE_SECRET``.  The
    function returns ``True`` if a signature was present and validated, ``False``
    if no signature header was provided, and raises ``SignatureVerificationError``
    if verification fails.
    """

    if not signature_header:
        return False

    settings = settings or get_settings()
    secret = settings.bubble_signature_secret
    if not secret:
        # Signature provided but secret missing is treated as configuration
        # error to avoid silently accepting unsigned requests.
        raise SignatureVerificationError()

    normalized = signature_header.strip()
    if "=" in normalized:
        prefix, value = normalized.split("=", 1)
        prefix = prefix.lower()
        if prefix == "hmac":
            try:
                provided = base64.b64decode(value, validate=True)
            except Exception as exc:  # pragma: no cover - defensive
                raise SignatureVerificationError() from exc
            digest = hmac.new(secret.encode(), raw_body, hashlib.sha256).digest()
            if not hmac.compare_digest(digest, provided):
                raise SignatureVerificationError()
            return True
        if prefix == "sha256":
            digest = hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
            if not hmac.compare_digest(digest, value.lower()):
                raise SignatureVerificationError()
            return True

    raise SignatureVerificationError()


__all__ = [
    "IdempotencyStore",
    "IdempotentResponse",
    "SignatureVerificationError",
    "verify_admin_token",
    "verify_signature",
]
