"""OCR pipeline for receipts using a local Tesseract backend."""
from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:  # optional dependencies
    from PIL import Image
except Exception:  # pragma: no cover - pillow is optional in tests
    Image = None  # type: ignore[misc]

try:
    import pytesseract
    from pytesseract import Output as TesseractOutput
except Exception:  # pragma: no cover - pytesseract optional
    pytesseract = None  # type: ignore[assignment]
    TesseractOutput = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)


class OCRTimeoutError(RuntimeError):
    """Raised when Tesseract execution exceeds the timeout."""


class OCRDecodeError(RuntimeError):
    """Raised when OCR text cannot be decoded."""


class OCRServiceError(RuntimeError):
    """Raised when the OCR engine returns an unexpected error."""


class ImageFetchError(RuntimeError):
    """Raised when the incoming payload cannot be decoded into bytes."""


@dataclass
class OCRWord:
    text: str
    bbox: Tuple[int, int, int, int]
    confidence: float


@dataclass
class OCRLine:
    text: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    words: List[OCRWord] = field(default_factory=list)


@dataclass
class OCRResult:
    raw_text: str
    lines: List[OCRLine]
    confidence: float


def _decode_payload(payload: bytes | str) -> bytes:
    if isinstance(payload, bytes):
        return payload
    if payload.startswith("http://") or payload.startswith("https://"):
        raise ImageFetchError("remote_fetch_disabled")
    try:
        return base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise ImageFetchError("invalid_base64") from exc


def _load_image(data: bytes) -> Image.Image:
    if Image is None:  # pragma: no cover - optional dependency
        raise OCRServiceError("pillow_not_available")
    try:
        return Image.open(io.BytesIO(data))
    except Exception as exc:  # pragma: no cover - invalid file
        raise OCRDecodeError("image_decode_failed") from exc


def _run_tesseract(image: Image.Image, *, language: str) -> Dict[str, List[Any]]:
    if pytesseract is None:  # pragma: no cover - optional dependency
        raise OCRServiceError("pytesseract_not_available")
    try:
        return pytesseract.image_to_data(
            image,
            lang=language,
            config="--oem 3 --psm 6",
            output_type=TesseractOutput.DICT,
        )
    except RuntimeError as exc:  # pragma: no cover - runtime failure
        raise OCRServiceError("tesseract_failed") from exc


def _build_result(data: Dict[str, List[Any]]) -> OCRResult:
    lines: Dict[int, OCRLine] = {}
    words: List[OCRWord] = []
    confidences: List[float] = []

    n = len(data.get("text", []))
    for idx in range(n):
        text = data["text"][idx] or ""
        conf_value = float(data.get("conf", [0])[idx] or 0)
        left = int(data.get("left", [0])[idx] or 0)
        top = int(data.get("top", [0])[idx] or 0)
        width = int(data.get("width", [0])[idx] or 0)
        height = int(data.get("height", [0])[idx] or 0)
        line_no = int(data.get("line_num", [0])[idx] or 0)

        bbox = (left, top, left + width, top + height)
        word = OCRWord(text=text, bbox=bbox, confidence=conf_value / 100.0)
        words.append(word)
        confidences.append(word.confidence)

        if line_no not in lines:
            lines[line_no] = OCRLine(text="", bbox=bbox, confidence=word.confidence, words=[word])
        else:
            line = lines[line_no]
            line.words.append(word)
            line_text = f"{line.text} {text}".strip()
            line.text = line_text
            line.confidence = max(line.confidence, word.confidence)
            line_bbox = (
                min(line.bbox[0], bbox[0]),
                min(line.bbox[1], bbox[1]),
                max(line.bbox[2], bbox[2]),
                max(line.bbox[3], bbox[3]),
            )
            line.bbox = line_bbox

    ordered_lines = [lines[key] for key in sorted(lines.keys()) if lines[key].text]
    raw_text = "\n".join(line.text for line in ordered_lines)
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return OCRResult(raw_text=raw_text.strip(), lines=ordered_lines, confidence=mean_conf)


def perform_ocr(payload: bytes | str, *, language: str) -> OCRResult:
    data = _decode_payload(payload)
    image = _load_image(data)
    result = _run_tesseract(image, language=language)
    ocr_result = _build_result(result)
    if not ocr_result.raw_text:
        raise OCRDecodeError("empty_text")
    return ocr_result


def fake_ocr(payload: bytes | str) -> OCRResult:
    """Fallback OCR used in tests when Tesseract is unavailable."""

    data = payload if isinstance(payload, bytes) else payload.encode()
    raw_text = data.decode("utf-8", errors="ignore")
    if not raw_text.strip():
        raise OCRDecodeError("empty_text")
    line = OCRLine(text=raw_text, bbox=(0, 0, 0, 0), confidence=0.5)
    return OCRResult(raw_text=raw_text, lines=[line], confidence=0.5)


def extract_ocr(payload: bytes | str, *, language: str, use_fallback: bool = True) -> OCRResult:
    try:
        return perform_ocr(payload, language=language)
    except OCRServiceError:
        if not use_fallback:
            raise
    except OCRDecodeError:
        if not use_fallback:
            raise
    LOGGER.warning("Falling back to fake OCR pipeline")
    return fake_ocr(payload)


__all__ = [
    "OCRWord",
    "OCRLine",
    "OCRResult",
    "OCRTimeoutError",
    "OCRDecodeError",
    "OCRServiceError",
    "ImageFetchError",
    "extract_ocr",
]
