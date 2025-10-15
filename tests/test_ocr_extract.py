from __future__ import annotations

from typing import Any, Tuple

import pytest

from app import ocr_extract


@pytest.fixture
def sample_text() -> str:
    return "\n".join(
        [
            "デンキチ",
            "2025-10-10",
            "標準税率 10%",
            "合計 ¥36,990",
            "T1234567890123",
            "埼玉県川口市栄町3-2-1",
            "お支払い方法: クレジット",
        ]
    )


def test_extract_all_parses_text(monkeypatch: pytest.MonkeyPatch, sample_text: str) -> None:
    monkeypatch.setattr(ocr_extract, "_load_bytes", lambda _: (b"binary", "bytes"))
    monkeypatch.setattr(ocr_extract, "_is_pdf", lambda __: False)
    monkeypatch.setattr(ocr_extract, "_perform_ocr", lambda __: sample_text)

    result = ocr_extract.extract_all("dummy")

    assert result["vendor"]["value"] == "デンキチ"
    assert result["amount"]["value"] == 36990
    assert result["tax_rate"]["value"] == 0.10
    assert result["invoice_number"]["value"] == "T1234567890123"
    assert result["address"]["value"] == "埼玉県川口市栄町3-2-1"
    assert result["payment_method"]["value"] == "クレジット"
    assert result["paid_date"]["value"] == "2025-10-10"


def test_extract_all_uses_pdf_pipeline(monkeypatch: pytest.MonkeyPatch, sample_text: str) -> None:
    calls: dict[str, int] = {"pdf": 0}

    def fake_load(_: Any) -> Tuple[bytes, str]:
        return b"%PDF-1.4", "http://example.com/sample.pdf"

    def fake_pdf(_: bytes) -> str:
        calls["pdf"] += 1
        return sample_text

    monkeypatch.setattr(ocr_extract, "_load_bytes", fake_load)
    monkeypatch.setattr(ocr_extract, "_extract_text_from_pdf", fake_pdf)

    result = ocr_extract.extract_all("http://example.com/sample.pdf")

    assert calls["pdf"] == 1
    assert result["vendor"]["value"] == "デンキチ"
