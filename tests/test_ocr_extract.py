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


def test_perform_ocr_uses_local_engine(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    class FakeImage:
        mode = "L"

        def load(self) -> None:
            captured["loaded"] = True

        def convert(self, mode: str) -> "FakeImage":
            captured["converted_to"] = mode
            self.mode = mode
            return self

    class FakeImageModule:
        @staticmethod
        def open(_: Any) -> FakeImage:
            captured["opened"] = True
            return FakeImage()

    class FakeTesseractError(Exception):
        pass

    class FakePytesseract:
        TesseractError = FakeTesseractError
        TesseractNotFoundError = FakeTesseractError

        @staticmethod
        def image_to_string(image: FakeImage, lang: str) -> str:
            captured["lang"] = lang
            captured["image_mode"] = image.mode
            return "Vanlee"

    monkeypatch.setenv("OCR_ENGINE", "local")
    monkeypatch.setenv("OCR_LANGUAGE", "eng")
    monkeypatch.setattr(ocr_extract, "_PIL_AVAILABLE", True)
    monkeypatch.setattr(ocr_extract, "_PYTESSERACT_AVAILABLE", True)
    monkeypatch.setattr(ocr_extract, "Image", FakeImageModule)
    monkeypatch.setattr(ocr_extract, "pytesseract", FakePytesseract)

    text = ocr_extract._perform_ocr(b"fake-bytes")

    assert text == "Vanlee"
    assert captured["opened"] is True
    assert captured["loaded"] is True
    assert captured["converted_to"] == "RGB"
    assert captured["image_mode"] == "RGB"
    assert captured["lang"] == "eng"


def test_perform_ocr_falls_back_when_rapidocr_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OCR_ENGINE", "rapidocr")

    def boom(_: bytes) -> str:
        raise ocr_extract.OCRServiceError("boom")

    monkeypatch.setattr(ocr_extract, "_ocr_rapidocr", boom)

    calls: dict[str, int] = {"local": 0}

    def fake_local(_: bytes) -> str:
        calls["local"] += 1
        return "fallback"

    monkeypatch.setattr(ocr_extract, "_ocr_local", fake_local)

    text = ocr_extract._perform_ocr(b"fake")

    assert text == "fallback"
    assert calls["local"] == 1


def test_ocr_local_reports_tesseract_errors(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeImage:
        def load(self) -> None:  # pragma: no cover - simple stub
            pass

        def convert(self, mode: str) -> "FakeImage":  # pragma: no cover
            return self

    class FakeImageModule:
        @staticmethod
        def open(_: Any) -> FakeImage:  # pragma: no cover - simple stub
            return FakeImage()

    class FakeTesseractError(Exception):
        pass

    class FakeTesseractNotFoundError(FakeTesseractError):
        pass

    class FakePytesseract:
        TesseractError = FakeTesseractError
        TesseractNotFoundError = FakeTesseractNotFoundError

        @staticmethod
        def image_to_string(_: Any, lang: str) -> str:  # noqa: ARG001
            raise FakeTesseractError("ocr failed")

    monkeypatch.setattr(ocr_extract, "_PIL_AVAILABLE", True)
    monkeypatch.setattr(ocr_extract, "_PYTESSERACT_AVAILABLE", True)
    monkeypatch.setattr(ocr_extract, "Image", FakeImageModule)
    monkeypatch.setattr(ocr_extract, "pytesseract", FakePytesseract)

    with pytest.raises(ocr_extract.OCRServiceError) as excinfo:
        ocr_extract._ocr_local(b"fake")

    assert "tesseract_error" in str(excinfo.value)


def test_ocr_local_rejects_invalid_images() -> None:
    class FakeTesseractError(Exception):
        pass

    class FakePytesseract:
        TesseractError = FakeTesseractError
        TesseractNotFoundError = FakeTesseractError

    class FakeImageModule:
        @staticmethod
        def open(_: Any) -> None:
            raise ocr_extract.UnidentifiedImageError("bad image")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(ocr_extract, "_PIL_AVAILABLE", True)
    monkeypatch.setattr(ocr_extract, "_PYTESSERACT_AVAILABLE", True)
    monkeypatch.setattr(ocr_extract, "Image", FakeImageModule)
    monkeypatch.setattr(ocr_extract, "pytesseract", FakePytesseract)

    with pytest.raises(ocr_extract.OCRDecodeError):
        ocr_extract._ocr_local(b"bad")

    monkeypatch.undo()
