"""Date extraction helpers."""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import List, Optional

from ..ocr_extract import OCRResult

DATE_PATTERNS = [
    re.compile(r"(\d{4})[./年](\d{1,2})[./月](\d{1,2})[日]?"),
    re.compile(r"(\d{1,2})月(\d{1,2})日"),
    re.compile(r"(\d{2})/(\d{1,2})/(\d{1,2})"),
]


@dataclass
class DateCandidate:
    value: Optional[dt.date]
    raw_text: str
    confidence: float


@dataclass
class DateExtraction:
    best: Optional[DateCandidate]
    candidates: List[DateCandidate]


def _normalise(year: int, month: int, day: int) -> Optional[dt.date]:
    try:
        return dt.date(year, month, day)
    except ValueError:
        return None


def _score(candidate: Optional[dt.date]) -> float:
    if not candidate:
        return 0.1
    today = dt.date.today()
    if candidate > today + dt.timedelta(days=30):
        return 0.2
    if candidate < today - dt.timedelta(days=365 * 10):
        return 0.3
    return 0.8


def _search_dates(text: str) -> List[DateCandidate]:
    candidates: List[DateCandidate] = []
    for pattern in DATE_PATTERNS:
        for match in pattern.finditer(text):
            groups = match.groups()
            if len(groups) == 3:
                g1, g2, g3 = groups
                if pattern is DATE_PATTERNS[0]:
                    year = int(g1)
                    month = int(g2)
                    day = int(g3)
                elif pattern is DATE_PATTERNS[1]:
                    year = dt.date.today().year
                    month = int(g1)
                    day = int(g2)
                else:
                    year = int(g1)
                    month = int(g2)
                    day = int(g3)
                date_value = _normalise(year, month, day)
                candidates.append(
                    DateCandidate(value=date_value, raw_text=match.group(0), confidence=_score(date_value))
                )
    return candidates


def extract_date(ocr: OCRResult, max_candidates: int = 5) -> DateExtraction:
    seen: List[str] = []
    collected: List[DateCandidate] = []
    for line in ocr.lines:
        for candidate in _search_dates(line.text):
            if candidate.raw_text in seen:
                continue
            seen.append(candidate.raw_text)
            collected.append(candidate)
    collected.sort(key=lambda item: item.confidence, reverse=True)
    best = collected[0] if collected else None
    return DateExtraction(best=best, candidates=collected[:max_candidates])


__all__ = ["DateCandidate", "DateExtraction", "extract_date"]
