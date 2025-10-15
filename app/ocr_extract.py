"""Receipt OCR helpers.

The previous implementation returned hard-coded sample data.  This module now
provides a real OCR pipeline that downloads the requested image/PDF, runs it
through an OCR engine, and parses the recognised text into the structured
``ExtractionResult`` schema used by the API.

The OCR engine currently supports Google Cloud Vision's ``images:annotate``
endpoint via an API key supplied through ``GOOGLE_VISION_API_KEY``.  The
implementation is intentionally pluggable so that alternative providers can be
introduced by extending ``_perform_ocr``.
"""
from __future__ import annotations

import base64
import logging
import os
import re
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import requests
from pdfminer.high_level import extract_text

LOGGER = logging.getLogger(__name__)

ExtractionResult = Dict[str, Any]

OCR_ENDPOINT = os.getenv(
    "GOOGLE_VISION_ENDPOINT", "https://vision.googleapis.com/v1/images:annotate"
)
OCR_TIMEOUT = 30


class ImageFetchError(RuntimeError):
    """Raised when the input image/PDF cannot be retrieved."""


class OCRServiceError(RuntimeError):
    """Raised when the upstream OCR engine fails."""


class OCRDecodeError(RuntimeError):
    """Raised when OCR output cannot be interpreted."""


def extract_all(
    image_input: Union[str, bytes],
    hint: Optional[Dict[str, Any]] = None,
) -> ExtractionResult:
    """Extract structured receipt fields from ``image_input``.

    Parameters
    ----------
    image_input:
        Either a URL string pointing to the receipt file, a base64 encoded
        string, or raw image/PDF bytes.
    hint:
        Optional metadata about the receipt.  Hints are ignored for tax-rate
        inference to ensure the engine derives it purely from the OCR content.
    """

    del hint  # hints are currently unused but kept for signature compatibility

    binary, _ = _load_bytes(image_input)
    if _is_pdf(binary):
        raw_text = _extract_text_from_pdf(binary)
    else:
        raw_text = _perform_ocr(binary)

    if not raw_text or not raw_text.strip():
        raise OCRDecodeError("empty_ocr_text")

    return _parse_receipt_text(raw_text)


def _load_bytes(image_input: Union[str, bytes]) -> Tuple[bytes, str]:
    if isinstance(image_input, (bytes, bytearray)):
        return bytes(image_input), "bytes"

    if isinstance(image_input, str):
        trimmed = image_input.strip()
        if trimmed.startswith("http://") or trimmed.startswith("https://"):
            try:
                response = requests.get(trimmed, timeout=OCR_TIMEOUT)
                response.raise_for_status()
            except requests.RequestException as exc:  # pragma: no cover - network
                raise ImageFetchError("fetch_failed") from exc
            return response.content, trimmed

        # Treat as base64 payload
        try:
            return base64.b64decode(trimmed, validate=True), "base64"
        except Exception as exc:  # pragma: no cover - invalid input
            raise OCRDecodeError("invalid_base64") from exc

    raise OCRDecodeError("unsupported_input_type")


def _is_pdf(binary: bytes) -> bool:
    return binary.startswith(b"%PDF")


def _extract_text_from_pdf(binary: bytes) -> str:
    try:
        return extract_text(BytesIO(binary))
    except Exception as exc:  # pragma: no cover - pdfminer internal failures
        raise OCRDecodeError("pdf_text_extraction_failed") from exc


def _perform_ocr(binary: bytes) -> str:
    engine = os.getenv("OCR_ENGINE", "google").strip().lower() or "google"
    if engine == "google":
        return _ocr_google(binary)
    raise OCRServiceError(f"unknown_ocr_engine:{engine}")


def _ocr_google(binary: bytes) -> str:
    api_key = os.getenv("GOOGLE_VISION_API_KEY")
    if not api_key:
        raise OCRServiceError("missing_google_vision_api_key")

    payload = {
        "requests": [
            {
                "image": {"content": base64.b64encode(binary).decode("ascii")},
                "features": [{"type": "TEXT_DETECTION"}],
            }
        ]
    }

    try:
        response = requests.post(
            f"{OCR_ENDPOINT}?key={api_key}", json=payload, timeout=OCR_TIMEOUT
        )
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - network
        raise OCRServiceError("google_vision_request_failed") from exc

    data = response.json()
    try:
        annotations = data["responses"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise OCRServiceError("google_vision_unexpected_response") from exc

    full_text = (
        annotations.get("fullTextAnnotation", {}).get("text")
        or annotations.get("textAnnotations", [{}])[0].get("description")
        or ""
    )
    return str(full_text)


def _parse_receipt_text(raw_text: str) -> ExtractionResult:
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    vendor, vendor_conf = _guess_vendor(lines)
    amount, amount_conf = _extract_total(lines)
    date_value, date_conf = _extract_date(lines)
    tax_rate, tax_conf = _extract_tax_rate(raw_text)
    payment, payment_conf = _extract_payment_method(raw_text)
    invoice, invoice_conf = _extract_invoice(raw_text)
    address, address_conf = _extract_address(lines)

    result: ExtractionResult = {"raw_text": raw_text}
    result["vendor"] = {"value": vendor, "confidence": vendor_conf}
    result["amount"] = {
        "value": amount,
        "currency": "JPY" if amount is not None else None,
        "confidence": amount_conf,
    }
    result["paid_date"] = {"value": date_value, "confidence": date_conf}
    result["tax_rate"] = {"value": tax_rate, "confidence": tax_conf}
    result["payment_method"] = {"value": payment, "confidence": payment_conf}
    result["invoice_number"] = {"value": invoice, "confidence": invoice_conf}
    result["address"] = {"value": address, "confidence": address_conf}
    return result


def _guess_vendor(lines: Iterable[str]) -> Tuple[Optional[str], float]:
    for line in lines:
        if _looks_like_amount(line) or _contains_date(line):
            continue
        if len(line) < 2:
            continue
        return line, 0.8
    return None, 0.0


def _extract_total(lines: Iterable[str]) -> Tuple[Optional[int], float]:
    best_value: Optional[int] = None
    best_conf = 0.0
    amount_pattern = re.compile(r"(?:¥|\\u00a5)?\s*([0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)\s*(?:円)?")

    for line in lines:
        for match in amount_pattern.finditer(line):
            raw = match.group(1).replace(",", "")
            try:
                value = int(raw)
            except ValueError:
                continue
            if best_value is None or value > best_value:
                best_value = value
                best_conf = 0.9
    return best_value, best_conf


def _extract_date(lines: Iterable[str]) -> Tuple[Optional[str], float]:
    iso_date: Optional[str] = None
    confidence = 0.0
    date_patterns = [
        re.compile(r"(\d{4})[年/.-](\d{1,2})[月/.-](\d{1,2})日?"),
        re.compile(r"(\d{2})[/-](\d{1,2})[/-](\d{1,2})"),
    ]

    for line in lines:
        for pattern in date_patterns:
            match = pattern.search(line)
            if not match:
                continue
            groups = match.groups()
            try:
                if len(groups[0]) == 4:
                    year = int(groups[0])
                    month = int(groups[1])
                    day = int(groups[2])
                else:
                    year = int(groups[0])
                    if year < 50:
                        year += 2000
                    else:
                        year += 1900
                    month = int(groups[1])
                    day = int(groups[2])
                iso = datetime(year, month, day).date().isoformat()
            except ValueError:
                continue
            iso_date = iso
            confidence = 0.8
            return iso_date, confidence
    return iso_date, confidence


def _extract_tax_rate(text: str) -> Tuple[Optional[float], float]:
    lowered = text.lower()
    if re.search(r"標準税率|10\s*%|10\s*％|１０％|10%対象", text):
        return 0.10, 0.8
    if re.search(r"軽減税率|8\s*%|8\s*％|８％|8%対象", text):
        return 0.08, 0.7
    if "reduced" in lowered and "tax" in lowered:
        return 0.08, 0.6
    return None, 0.0


def _extract_payment_method(text: str) -> Tuple[Optional[str], float]:
    if re.search(r"クレジット|credit", text, re.IGNORECASE):
        return "クレジット", 0.7
    if re.search(r"現金|cash", text, re.IGNORECASE):
        return "現金", 0.7
    if re.search(r"電子マネー|ic|交通系", text, re.IGNORECASE):
        return "電子マネー", 0.6
    return None, 0.0


def _extract_invoice(text: str) -> Tuple[Optional[str], float]:
    match = re.search(r"T\d{13}", text)
    if match:
        return match.group(0), 0.85
    return None, 0.0


def _extract_address(lines: Iterable[str]) -> Tuple[Optional[str], float]:
    for line in lines:
        if re.search(r"[都道府県].*市|市.*区|区.*町|丁目", line):
            return line, 0.6
    return None, 0.0


def _looks_like_amount(text: str) -> bool:
    return bool(re.search(r"¥|円|\d{1,3}(?:,\d{3})+", text))


def _contains_date(text: str) -> bool:
    return bool(re.search(r"\d{4}[年/.-]\d{1,2}[月/.-]\d{1,2}", text))


__all__ = [
    "ExtractionResult",
    "ImageFetchError",
    "OCRDecodeError",
    "OCRServiceError",
    "extract_all",
]
