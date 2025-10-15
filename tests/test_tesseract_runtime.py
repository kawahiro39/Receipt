from __future__ import annotations

import os
import shutil

import pytest

Image = pytest.importorskip("PIL.Image").Image
ImageDraw = pytest.importorskip("PIL.ImageDraw")
pytesseract = pytest.importorskip("pytesseract")

pytestmark = pytest.mark.skipif(
    shutil.which("tesseract") is None,
    reason="tesseract binary not available",
)


def test_pytesseract_image_to_string_smoke() -> None:
    """Ensure pytesseract can call the native binary when available."""

    image = Image.new("L", (160, 60), color=255)
    draw = ImageDraw.Draw(image)
    draw.text((10, 15), "Vanlee", fill=0)

    text = pytesseract.image_to_string(image, lang="eng")

    assert "vanlee" in text.lower()


def test_pytesseract_image_to_string_japanese(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure pytesseract can switch to Japanese language data."""

    monkeypatch.setenv("OCR_LANGUAGE", "jpn")

    image = Image.new("L", (160, 60), color=255)
    draw = ImageDraw.Draw(image)
    draw.text((10, 15), "Vanlee", fill=0)

    text = pytesseract.image_to_string(image, lang=os.environ["OCR_LANGUAGE"])

    assert "vanlee" in text.lower()
