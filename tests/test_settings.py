from __future__ import annotations

import pytest

from app import settings as settings_module
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
    settings_module.reset_settings_state()
    settings = Settings.load()
    assert settings.bubble_api_base == expected


def test_loads_from_env_file(tmp_path, monkeypatch):
    env_content = (
        "BUBBLE_API_BASE=https://example.com/version-test/api/1.1/obj\n"
        "BUBBLE_API_KEY=test-key\n"
        "OCR_ENGINE=local\n"
        "OCR_LANGUAGE=jpn+eng\n"
        "ADMIN_TOKEN=test-admin\n"
    )
    env_file = tmp_path / ".env"
    env_file.write_text(env_content)

    for name in [
        "BUBBLE_API_BASE",
        "BUBBLE_API_KEY",
        "OCR_ENGINE",
        "OCR_LANGUAGE",
        "ADMIN_TOKEN",
    ]:
        monkeypatch.delenv(name, raising=False)

    monkeypatch.chdir(tmp_path)
    settings_module.reset_settings_state()

    settings = Settings.load()

    assert settings.bubble_api_base == "https://example.com/version-test/api/1.1"
    assert settings.bubble_api_key == "test-key"
