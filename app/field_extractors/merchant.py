"""Merchant extraction using heuristics and fuzzy matching."""
from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import List, Optional

from ..ocr_extract import OCRResult

COMMON_MERCHANTS = [
    "スターバックス",
    "ファミリーマート",
    "セブンイレブン",
    "ローソン",
    "マクドナルド",
]


@dataclass
class MerchantCandidate:
    value: Optional[str]
    confidence: float


@dataclass
class MerchantExtraction:
    best: Optional[MerchantCandidate]
    candidates: List[MerchantCandidate]


def _score(text: str) -> float:
    score = 0.2
    if text.isupper() or text.islower():
        score += 0.1
    if len(text) > 5:
        score += 0.2
    if "店" in text or "株式会社" in text:
        score += 0.2
    return min(score, 0.95)


def extract_merchant(ocr: OCRResult, max_candidates: int = 5) -> MerchantExtraction:
    candidates: List[MerchantCandidate] = []
    for line in ocr.lines:
        text = line.text.strip()
        if not text:
            continue
        if any(char.isdigit() for char in text):
            continue
        score = _score(text)
        if score < 0.2:
            continue
        candidates.append(MerchantCandidate(value=text, confidence=score))

    for entry in COMMON_MERCHANTS:
        matches = difflib.get_close_matches(entry, [c.value for c in candidates if c.value], n=1, cutoff=0.8)
        if matches:
            candidates.insert(0, MerchantCandidate(value=entry, confidence=0.9))
            break

    candidates.sort(key=lambda item: item.confidence, reverse=True)
    deduped: List[MerchantCandidate] = []
    seen: List[str] = []
    for candidate in candidates:
        value = candidate.value or ""
        if value in seen:
            continue
        seen.append(value)
        deduped.append(candidate)

    best = deduped[0] if deduped else None
    return MerchantExtraction(best=best, candidates=deduped[:max_candidates])


__all__ = ["MerchantCandidate", "MerchantExtraction", "extract_merchant"]
