from __future__ import annotations

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
