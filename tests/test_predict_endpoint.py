from __future__ import annotations

from typing import Any, Dict

import pytest

from app.main import HTTPException, PredictRequest, predict


@pytest.fixture(autouse=True)
def reset_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.main._generate_doc_id", lambda: "r_20251015_ab12cd")


@pytest.mark.asyncio
async def test_predict_creates_receipt(monkeypatch: pytest.MonkeyPatch) -> None:
    extracted: Dict[str, Any] = {
        "raw_text": "デンキチ\n2025-10-10",
        "vendor": {"value": "デンキチ", "confidence": 0.9},
        "amount": {"value": 36990, "confidence": 0.95, "currency": "JPY"},
        "tax_rate": {"value": 0.1, "confidence": 0.8},
        "invoice_number": {"value": "T1234567890123", "confidence": 0.9},
        "address": {"value": "埼玉県川口市栄町3-2-1", "confidence": 0.6},
        "paid_date": {"value": "2025-10-10", "confidence": 0.8},
        "payment_method": {"value": "クレジット", "confidence": 0.7},
    }

    monkeypatch.setattr("app.main.ocr_extract.extract_all", lambda *_, **__: extracted)
    monkeypatch.setattr("app.main.model_store.load_latest_model", lambda: (None, None))
    monkeypatch.setattr(
        "app.main.predict_category", lambda *_: ("事務用品費", 0.82, [("雑費", 0.4)])
    )

    calls: Dict[str, Dict[str, Any]] = {}

    def fake_search(type_name: str, **_: Any) -> Dict[str, Any]:
        assert type_name == "Receipt"
        return {"response": {"results": []}}

    def fake_create(type_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        assert type_name == "Receipt"
        calls["create"] = payload
        return {"id": "rec_123"}

    monkeypatch.setattr("app.main.bubble_client.bubble_search", fake_search)
    monkeypatch.setattr("app.main.bubble_client.bubble_create", fake_create)
    monkeypatch.setattr("app.main.bubble_client.bubble_update", lambda *_, **__: None)

    response = await predict(PredictRequest(image_url="https://example.com/receipt.jpg"))

    assert response.doc_id == "r_20251015_ab12cd"
    assert response.receipt_id == "rec_123"
    assert response.extracted["vendor"]["value"] == "デンキチ"
    assert response.category["pred"] == "事務用品費"

    payload = calls["create"]
    assert payload["doc_id"] == "r_20251015_ab12cd"
    assert payload["image_url"] == "https://example.com/receipt.jpg"
    assert payload["vendor"] == "デンキチ"
    assert payload["total"] == 36990
    assert payload["tax_rate"] == 0.1
    assert payload["invoice_number"] == "T1234567890123"
    assert payload["address"] == "埼玉県川口市栄町3-2-1"
    assert payload["paid_date"] == "2025-10-10"
    assert payload["payment_method"] == "クレジット"
    assert payload["pred_category"] == "事務用品費"
    assert payload["status"] == "predicted"


@pytest.mark.asyncio
async def test_predict_requires_image_url() -> None:
    with pytest.raises(HTTPException) as exc:
        await predict(PredictRequest())
    assert exc.value.status_code == 400
    assert exc.value.detail == "missing_image_url"
