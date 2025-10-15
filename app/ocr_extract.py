"""Receipt OCR helpers.

The previous implementation returned hard-coded sample data.  This module now
provides a real OCR pipeline that downloads the requested image/PDF, runs it
through an OCR engine, and parses the recognised text into the structured
``ExtractionResult`` schema used by the API.

The OCR engine uses a local Tesseract backend by default.  Additional engines
can be introduced by extending ``_perform_ocr``.
"""
from __future__ import annotations

import base64
import importlib.util
import logging
import os
import re
import unicodedata
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import requests

_PIL_AVAILABLE = importlib.util.find_spec("PIL") is not None
if _PIL_AVAILABLE:
    from PIL import Image, UnidentifiedImageError
else:  # pragma: no cover - pillow missing in runtime environment
    Image = Any  # type: ignore[assignment]

    class UnidentifiedImageError(Exception):
        """Fallback error when Pillow is unavailable."""

_PYTESSERACT_AVAILABLE = importlib.util.find_spec("pytesseract") is not None
if _PYTESSERACT_AVAILABLE:
    import pytesseract
else:  # pragma: no cover - pytesseract missing in runtime environment
    pytesseract = None  # type: ignore[assignment]

_PDFMINER_AVAILABLE = importlib.util.find_spec("pdfminer") is not None
if _PDFMINER_AVAILABLE:
    from pdfminer.high_level import extract_text
else:  # pragma: no cover - pdfminer missing in runtime environment
    extract_text = None  # type: ignore[assignment]

_RAPIDOCR_AVAILABLE = importlib.util.find_spec("rapidocr_onnxruntime") is not None
if _RAPIDOCR_AVAILABLE:
    from rapidocr_onnxruntime import RapidOCR  # type: ignore
else:  # pragma: no cover - rapidocr missing in runtime environment
    RapidOCR = None  # type: ignore[assignment]

_NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None
if _NUMPY_AVAILABLE:
    import numpy as np
else:  # pragma: no cover - numpy missing in runtime environment
    np = None  # type: ignore[assignment]

_JACONV_AVAILABLE = importlib.util.find_spec("jaconv") is not None
if _JACONV_AVAILABLE:
    import jaconv
else:  # pragma: no cover - jaconv missing in runtime environment
    jaconv = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)

ExtractionResult = Dict[str, Any]

OCR_TIMEOUT = 30

_RAPIDOCR_ENGINE: Optional[RapidOCR] = None  # type: ignore[assignment]

_ERA_OFFSETS = {"令和": 2018, "平成": 1988, "昭和": 1925}

_VENDOR_SKIP_PATTERN = re.compile(
    r"(〒|tel|電話|fax|url|http|www|mail|メール|領収書|receipt|合計|total)",
    re.IGNORECASE,
)

_PREFECTURE_PATTERN = re.compile(
    "|".join(
        [
            "北海道",
            "青森県",
            "岩手県",
            "宮城県",
            "秋田県",
            "山形県",
            "福島県",
            "茨城県",
            "栃木県",
            "群馬県",
            "埼玉県",
            "千葉県",
            "東京都",
            "神奈川県",
            "新潟県",
            "富山県",
            "石川県",
            "福井県",
            "山梨県",
            "長野県",
            "岐阜県",
            "静岡県",
            "愛知県",
            "三重県",
            "滋賀県",
            "京都府",
            "大阪府",
            "兵庫県",
            "奈良県",
            "和歌山県",
            "鳥取県",
            "島根県",
            "岡山県",
            "広島県",
            "山口県",
            "徳島県",
            "香川県",
            "愛媛県",
            "高知県",
            "福岡県",
            "佐賀県",
            "長崎県",
            "熊本県",
            "大分県",
            "宮崎県",
            "鹿児島県",
            "沖縄県",
        ]
    )
)


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


def _normalise_text(text: str) -> str:
    cleaned = unicodedata.normalize("NFKC", text or "")
    if _JACONV_AVAILABLE and jaconv is not None:  # pragma: no branch - import guard
        cleaned = jaconv.z2h(cleaned, kana=False, digit=True, ascii=True)
    cleaned = cleaned.replace("\u3000", " ")
    cleaned = re.sub(r"[\t\f\r]+", " ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def _normalise_lines(raw_text: str) -> List[str]:
    return [line for line in (_normalise_text(line) for line in raw_text.splitlines()) if line]


def _is_japanese_char(char: str) -> bool:
    code = ord(char)
    return (
        0x3040 <= code <= 0x30FF  # Hiragana & Katakana
        or 0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
        or 0x3400 <= code <= 0x4DBF  # CJK Unified Ideographs Extension A
        or 0xFF66 <= code <= 0xFF9D  # Half-width Katakana
    )


def _score_vendor_line(line: str, index: int) -> float:
    score = 0.0
    if index == 0:
        score += 0.45
    elif index == 1:
        score += 0.35
    elif index <= 4:
        score += 0.25
    elif index <= 8:
        score += 0.15

    if re.search(r"株式会社|有限会社|合同会社|inc\.?|co\.?|ltd", line, re.IGNORECASE):
        score += 0.25

    japanese_chars = sum(1 for char in line if _is_japanese_char(char))
    ascii_letters = sum(1 for char in line if char.isalpha() and char.isascii())
    digits = sum(1 for char in line if char.isdigit())
    total = len(line)
    if total:
        score += min(japanese_chars / total, 0.6)
        score += min(ascii_letters / total, 0.3)
        score -= min(digits / total, 0.5)

    if re.search(r"店|堂|屋|社|センター|カフェ|サービス", line):
        score += 0.1

    return max(score, 0.0)


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
    if not _PDFMINER_AVAILABLE or extract_text is None:
        raise OCRServiceError("pdfminer_not_installed")
    try:
        return extract_text(BytesIO(binary))
    except Exception as exc:  # pragma: no cover - pdfminer internal failures
        raise OCRDecodeError("pdf_text_extraction_failed") from exc


def _perform_ocr(binary: bytes) -> str:
    engine = os.getenv("OCR_ENGINE", "rapidocr").strip().lower() or "rapidocr"
    if engine == "rapidocr":
        try:
            return _ocr_rapidocr(binary)
        except (OCRServiceError, OCRDecodeError) as exc:
            LOGGER.warning(
                "rapidocr_failed_falling_back: %s",
                exc,
                exc_info=LOGGER.isEnabledFor(logging.DEBUG),
            )
            return _ocr_local(binary)
    if engine == "local":
        return _ocr_local(binary)
    raise OCRServiceError(f"unknown_ocr_engine:{engine}")


def _ocr_local(binary: bytes) -> str:
    if not _PYTESSERACT_AVAILABLE:
        raise OCRServiceError("pytesseract_not_installed")
    image = _image_from_bytes(binary)
    language = os.getenv("OCR_LANGUAGE", "eng").strip() or "eng"
    try:
        return pytesseract.image_to_string(image, lang=language)
    except pytesseract.TesseractNotFoundError as exc:
        raise OCRServiceError("tesseract_not_found") from exc
    except pytesseract.TesseractError as exc:
        raise OCRServiceError(f"tesseract_error:{exc}") from exc
    except Exception as exc:  # pragma: no cover - unexpected pytesseract failure
        raise OCRServiceError("tesseract_unknown_error") from exc


def _ocr_rapidocr(binary: bytes) -> str:
    if not _RAPIDOCR_AVAILABLE or RapidOCR is None:
        raise OCRServiceError("rapidocr_not_installed")
    if not _NUMPY_AVAILABLE or np is None:
        raise OCRServiceError("rapidocr_numpy_missing")
    image = _image_from_bytes(binary)
    np_image = np.array(image)
    engine = _get_rapidocr()
    try:
        result, _ = engine(np_image)
    except Exception as exc:  # pragma: no cover - rapidocr runtime failure
        raise OCRServiceError("rapidocr_execution_failed") from exc
    if not result:
        raise OCRDecodeError("rapidocr_empty_result")
    texts: List[str] = []
    for entry in result:
        if not entry:
            continue
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            candidate = entry[1]
        else:
            candidate = entry
        if isinstance(candidate, (list, tuple)) and candidate:
            candidate = candidate[0]
        if not isinstance(candidate, str):
            continue
        normalised = _normalise_text(candidate)
        if normalised:
            texts.append(normalised)
    if not texts:
        raise OCRDecodeError("rapidocr_no_text")
    return "\n".join(texts)


def _get_rapidocr() -> RapidOCR:
    global _RAPIDOCR_ENGINE
    if _RAPIDOCR_ENGINE is None:
        _RAPIDOCR_ENGINE = RapidOCR(det_use_cuda=False, rec_use_cuda=False, cls_use_cuda=False)
    return _RAPIDOCR_ENGINE


def _image_from_bytes(binary: bytes) -> Image.Image:
    if not _PIL_AVAILABLE:
        raise OCRServiceError("pillow_not_installed")
    try:
        image = Image.open(BytesIO(binary))
        image.load()
    except UnidentifiedImageError as exc:
        raise OCRDecodeError("unsupported_image_format") from exc
    except Exception as exc:  # pragma: no cover - pillow internal failures
        raise OCRServiceError("image_open_failed") from exc
    return image.convert("RGB")


def _parse_receipt_text(raw_text: str) -> ExtractionResult:
    lines = _normalise_lines(raw_text)
    normalised_text = "\n".join(lines)
    vendor, vendor_conf = _guess_vendor(lines)
    amount, amount_conf = _extract_total(lines)
    date_value, date_conf = _extract_date(lines)
    tax_rate, tax_conf = _extract_tax_rate(normalised_text)
    payment, payment_conf = _extract_payment_method(normalised_text)
    invoice, invoice_conf = _extract_invoice(normalised_text)
    address, address_conf = _extract_address(lines)

    result: ExtractionResult = {"raw_text": normalised_text}
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
    best_line: Optional[str] = None
    best_score = 0.0
    for index, original in enumerate(list(lines)[:20]):
        if not original:
            continue
        if _looks_like_amount(original) or _contains_date(original):
            continue
        if _VENDOR_SKIP_PATTERN.search(original):
            continue
        candidate = original.strip("-＝=・:：～~")
        candidate = re.sub(r"[☆★※#]+", "", candidate)
        if len(candidate) < 2:
            continue
        score = _score_vendor_line(candidate, index)
        if score > best_score:
            best_score = score
            best_line = candidate
    if best_line:
        confidence = min(0.95, 0.55 + best_score / 2.0)
        return best_line, confidence
    return None, 0.0


def _extract_total(lines: Iterable[str]) -> Tuple[Optional[int], float]:
    best_value: Optional[int] = None
    best_conf = 0.0
    amount_pattern = re.compile(r"(?:¥|\\u00a5)?\s*([0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)\s*(?:円)?")

    for line in lines:
        for match in amount_pattern.finditer(line):
            raw = match.group(1).replace(",", "")
            start = match.start(1)
            if start > 0 and line[start - 1].isalnum():
                continue
            try:
                value = int(raw)
            except ValueError:
                continue
            if best_value is None or value > best_value:
                best_value = value
                if re.search(r"合計|総額|請求額|金額|total|税込", line, re.IGNORECASE):
                    best_conf = 0.97
                elif value >= 1000:
                    best_conf = 0.9
                else:
                    best_conf = 0.75
    return best_value, best_conf


def _extract_date(lines: Iterable[str]) -> Tuple[Optional[str], float]:
    iso_date: Optional[str] = None
    confidence = 0.0
    date_patterns = [
        re.compile(r"(?:発行日|領収日|支払日)[:：\s]*(\d{4})[年/.-]?(\d{1,2})[月/.-]?(\d{1,2})日?"),
        re.compile(r"(\d{4})[年/.・-](\d{1,2})[月/.・-](\d{1,2})日?"),
        re.compile(r"(令和|平成|昭和)(\d{1,2})年(\d{1,2})月(\d{1,2})日?"),
        re.compile(r"R(\d{1,2})[./-](\d{1,2})[./-](\d{1,2})", re.IGNORECASE),
        re.compile(r"(\d{4})/(\d{1,2})/(\d{1,2})"),
        re.compile(r"(\d{2})[/-](\d{1,2})[/-](\d{1,2})"),
    ]

    for line in lines:
        for idx, pattern in enumerate(date_patterns):
            match = pattern.search(line)
            if not match:
                continue
            groups = match.groups()
            try:
                if idx == 3:  # R01.01.01 style
                    year = int(groups[0]) + 2018
                    month = int(groups[1])
                    day = int(groups[2])
                elif groups and groups[0] in _ERA_OFFSETS:
                    year = _ERA_OFFSETS[groups[0]] + int(groups[1])
                    month = int(groups[2])
                    day = int(groups[3])
                elif len(groups[0]) == 4:
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
            except (ValueError, IndexError):
                continue
            iso_date = iso
            confidence = 0.85
            return iso_date, confidence
    return iso_date, confidence


def _extract_tax_rate(text: str) -> Tuple[Optional[float], float]:
    lowered = text.lower()
    match = re.search(r"税率[:：\s]*([0-9]+(?:\.[0-9]+)?)\s*%", text)
    if match:
        value = float(match.group(1)) / 100
        return round(value, 2), 0.85
    if re.search(r"標準税率|10\s*%|10\s*％|１０％|10%対象|tax10", text):
        return 0.10, 0.82
    if re.search(r"軽減税率|8\s*%|8\s*％|８％|8%対象|tax8", text):
        return 0.08, 0.75
    if "reduced" in lowered and "tax" in lowered:
        return 0.08, 0.6
    return None, 0.0


def _extract_payment_method(text: str) -> Tuple[Optional[str], float]:
    patterns = [
        (r"クレジット|カード|visa|mastercard|amex", "クレジット", 0.78),
        (r"現金|cash", "現金", 0.75),
        (r"電子マネー|ic|交通系|suica|pasmo", "電子マネー", 0.7),
        (r"paypay|line\s*pay|楽天ペイ|メルペイ|d払い|au\s*pay", "キャッシュレス決済", 0.7),
        (r"振込|銀行振込|transfer", "振込", 0.68),
        (r"請求書|後払い", "請求書払い", 0.6),
    ]
    for pattern, label, conf in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return label, conf
    return None, 0.0


def _extract_invoice(text: str) -> Tuple[Optional[str], float]:
    cleaned = text.replace("ー", "-")
    match = re.search(r"T[-\d]{13,}", cleaned)
    if not match:
        match = re.search(r"登録番号[:：\s]*([Tt]?[-\d]{13,})", cleaned)
    if match:
        token = match.group(0) if match.lastindex is None else match.group(match.lastindex)
        token = token.strip().upper()
        digits = "".join(ch for ch in token if ch.isdigit())
        if len(digits) >= 13:
            value = "T" + digits[:13]
            return value, 0.9
    return None, 0.0


def _extract_address(lines: Iterable[str]) -> Tuple[Optional[str], float]:
    lines_list = list(lines)
    postal_pattern = re.compile(r"〒?\s?\d{3}-?\d{4}")
    for idx, line in enumerate(lines_list):
        if re.search(r"T\d{6,}", line):
            continue
        if postal_pattern.search(line):
            segment = line
            if idx + 1 < len(lines_list):
                next_line = lines_list[idx + 1]
                if _PREFECTURE_PATTERN.search(next_line) or "丁目" in next_line:
                    segment = f"{segment} {next_line}"
            return segment.strip(), 0.75

    for line in lines_list:
        if re.search(r"T\d{6,}", line):
            continue
        if _PREFECTURE_PATTERN.search(line) or re.search(r"市.*区|区.*町|丁目", line):
            return line, 0.65
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
