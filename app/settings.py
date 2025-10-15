"""Application settings management for the receipt ingestion service."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional


@dataclass(frozen=True)
class Settings:
    bubble_api_base: str
    bubble_api_key: str
    ocr_engine: str
    ocr_language: str
    admin_token: str
    timezone: Optional[str]
    bubble_signature_secret: Optional[str]

    @staticmethod
    def _require_env(name: str) -> str:
        value = os.getenv(name)
        if value is None or not value.strip():
            raise RuntimeError(f"Environment variable {name} is required")
        return value.strip()

    @classmethod
    def load(cls) -> "Settings":
        raw_base = cls._require_env("BUBBLE_API_BASE")
        if not raw_base.startswith("https://"):
            raise RuntimeError("BUBBLE_API_BASE must start with https://")
        # Allow callers to pass either the `/api/1.1` base or the `/api/1.1/obj`
        # collection root. The service always appends `/obj`, so we normalise the
        # environment value to avoid generating URLs like `/obj/obj/...` when the
        # caller already included it.
        base = raw_base.rstrip("/")
        if base.endswith("/obj"):
            base = base[: -len("/obj")]
        bubble_api_base = base.rstrip("/")

        api_key = cls._require_env("BUBBLE_API_KEY")
        ocr_engine = cls._require_env("OCR_ENGINE").lower()
        if ocr_engine != "local":
            raise RuntimeError("OCR_ENGINE currently supports only 'local'")
        ocr_language = cls._require_env("OCR_LANGUAGE")
        admin_token = cls._require_env("ADMIN_TOKEN")

        timezone = os.getenv("TZ")
        signature_secret = os.getenv("BUBBLE_SIGNATURE_SECRET")

        return cls(
            bubble_api_base=bubble_api_base,
            bubble_api_key=api_key,
            ocr_engine=ocr_engine,
            ocr_language=ocr_language,
            admin_token=admin_token,
            timezone=timezone.strip() if timezone else None,
            bubble_signature_secret=signature_secret.strip() if signature_secret else None,
        )


@lru_cache()
def get_settings() -> Settings:
    return Settings.load()


__all__ = ["Settings", "get_settings"]
