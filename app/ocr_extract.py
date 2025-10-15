"""Utility helpers for extracting receipt fields from images.

This module currently implements a lightweight placeholder extractor that
returns deterministic data for development and unit testing. The structure is
kept intentionally close to the expected production output so that swapping in a
real OCR engine later requires minimal changes.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, Optional, Union

ExtractionResult = Dict[str, Any]


def extract_all(image_input: Union[str, bytes], hint: Optional[Dict[str, Any]] = None) -> ExtractionResult:
    """Extract structured fields from an image URL or raw bytes.

    Parameters
    ----------
    image_input:
        Either a URL string pointing to the receipt image or raw image bytes.
    hint:
        Optional dict of hints (e.g. expected tax rate) that downstream
        extractors could leverage. The placeholder implementation only echoes a
        subset of hints back in the response for traceability.
    """

    vendor_hint = None
    tax_rate_hint = None
    if hint:
        vendor_hint = hint.get("known_vendor")
        tax_rate_hint = hint.get("tax_rate")

    return {
        "raw_text": "Sample receipt text for development",
        "vendor": {"value": vendor_hint or "Sample Store", "confidence": 0.5},
        "date": {"value": str(date.today()), "confidence": 0.5},
        "total": {"value": 1000, "confidence": 0.5},
        "tax": {"value": 91, "rate": tax_rate_hint or 0.1, "confidence": 0.5},
        "payment_method": {"value": "クレジット", "confidence": 0.5},
    }


__all__ = ["extract_all"]
