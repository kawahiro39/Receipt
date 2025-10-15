from __future__ import annotations

import pytest

from app.settings import Settings


def _populate_env(monkeypatch):
    monkeypatch.setenv("BUBBLE_API_KEY", "dummy")
    monkeypatch.setenv("OCR_ENGINE", "local")
    monkeypatch.setenv("OCR_LANGUAGE", "jpn+eng")
    monkeypatch.setenv("ADMIN_TOKEN", "token")


@pytest.mark.parametrize(
    "env_value,expected",
    [
        ("https://example.com/version-test/api/1.1", "https://example.com/version-test/api/1.1"),
        ("https://example.com/version-test/api/1.1/", "https://example.com/version-test/api/1.1"),
        ("https://example.com/version-test/api/1.1/obj", "https://example.com/version-test/api/1.1"),
        ("https://example.com/version-test/api/1.1/obj/", "https://example.com/version-test/api/1.1"),
    ],
)

def test_bubble_base_normalised(monkeypatch, env_value, expected):
    _populate_env(monkeypatch)
    monkeypatch.setenv("BUBBLE_API_BASE", env_value)
    settings = Settings.load()
    assert settings.bubble_api_base == expected
