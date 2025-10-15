"""Rule-based amount extraction utilities."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional

from ..ocr_extract import OCRResult

AMOUNT_PATTERN = re.compile(r"(?<![\d])(\d{1,3}(?:[,\d]{0,3})*(?:\.\d{1,2})?)(?=\D)")


@dataclass
class AmountCandidate:
    value: Optional[float]
    raw_text: str
    confidence: float
    bbox: Optional[tuple[int, int, int, int]] = None


@dataclass
class AmountExtraction:
    best: Optional[AmountCandidate]
    candidates: List[AmountCandidate]


def _normalise_number(text: str) -> Optional[float]:
    cleaned = text.replace(",", "").replace("円", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _score_candidate(value: Optional[float], raw: str) -> float:
    if value is None:
        return 0.1
    score = 0.3
    if value >= 0:
        score += 0.2
    if "合計" in raw or "税込" in raw:
        score += 0.3
    if value >= 1000:
        score += 0.2
    return min(score, 0.99)


def _iter_texts(ocr: OCRResult) -> Iterable[tuple[str, Optional[tuple[int, int, int, int]]]]:
    for line in ocr.lines:
        yield line.text, line.bbox
        for word in line.words:
            yield word.text, word.bbox


def extract_amount(ocr: OCRResult, max_candidates: int = 5) -> AmountExtraction:
    """Extract candidate amount values from OCR output."""

    seen = []
    candidates: List[AmountCandidate] = []
    for text, bbox in _iter_texts(ocr):
        for match in AMOUNT_PATTERN.finditer(text + " "):
            raw = match.group(1)
            if raw in seen:
                continue
            seen.append(raw)
            value = _normalise_number(raw)
            confidence = _score_candidate(value, text)
            candidates.append(AmountCandidate(value=value, raw_text=raw, confidence=confidence, bbox=bbox))

    candidates.sort(key=lambda item: item.confidence, reverse=True)
    best = candidates[0] if candidates else None
    return AmountExtraction(best=best, candidates=candidates[:max_candidates])


__all__ = ["AmountCandidate", "AmountExtraction", "extract_amount"]
