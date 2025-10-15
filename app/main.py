"""FastAPI entrypoint for the Vanlee Receipt AI service."""
from __future__ import annotations

import logging
import random
import string
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import bubble_client, model_store, ocr_extract
from .classifier import ModelBundle, partial_train, predict_category

LOGGER = logging.getLogger(__name__)
app = FastAPI(title="Vanlee Receipt AI")


class PredictRequest(BaseModel):
    image_url: Optional[str] = None
    image_base64: Optional[str] = Field(default=None, description="Base64 encoded image")
    hint: Optional[Dict[str, Any]] = None

    def image_source(self) -> str:
        if self.image_url:
            return self.image_url
        if self.image_base64:
            return self.image_base64
        raise HTTPException(status_code=400, detail="missing_image_url")


class PredictResponse(BaseModel):
    doc_id: str
    extracted: Dict[str, Any]
    category: Dict[str, Any]
    model_version: Optional[str]
    receipt_id: Optional[str] = None


class FeedbackPayload(BaseModel):
    receipt_id: Optional[str] = None
    doc_id: Optional[str] = None
    correct: Dict[str, Any]
    reason: Optional[str] = None


class TrainRequest(BaseModel):
    since: Optional[str] = None
    min_samples: int = Field(default=1, ge=0)


def _generate_doc_id() -> str:
    today = datetime.utcnow().strftime("%Y%m%d")
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"r_{today}_{suffix}"


def _lookup_receipt_by_doc_id(doc_id: str) -> Optional[str]:
    constraints = [{"key": "doc_id", "constraint_type": "equals", "value": doc_id}]
    try:
        response = bubble_client.bubble_search("Receipt", constraints=constraints, limit=1)
    except Exception as exc:  # pragma: no cover - defensive network handling
        LOGGER.warning("Receipt lookup failed: %s", exc)
        return None

    container = response.get("response", {}) if isinstance(response.get("response"), dict) else response
    results = container.get("results") if isinstance(container, dict) else response.get("results")
    if isinstance(results, list) and results:
        first = results[0]
        return first.get("_id") or first.get("id")
    return None


def _prepare_receipt_payload(
    *,
    doc_id: str,
    extracted: Dict[str, Any],
    image_url: Optional[str],
    label: str,
    score: float,
    version_name: Optional[str],
) -> Dict[str, Any]:
    amount = extracted.get("amount") if isinstance(extracted.get("amount"), dict) else {}
    tax_rate = extracted.get("tax_rate") if isinstance(extracted.get("tax_rate"), dict) else {}
    paid_date = extracted.get("paid_date") if isinstance(extracted.get("paid_date"), dict) else {}
    vendor = extracted.get("vendor") if isinstance(extracted.get("vendor"), dict) else {}
    invoice = extracted.get("invoice_number") if isinstance(extracted.get("invoice_number"), dict) else {}
    address = extracted.get("address") if isinstance(extracted.get("address"), dict) else {}
    payment = extracted.get("payment_method") if isinstance(extracted.get("payment_method"), dict) else {}

    return {
        "doc_id": doc_id,
        "image_url": image_url,
        "vendor": vendor.get("value") if isinstance(vendor, dict) else vendor,
        "tax_rate": tax_rate.get("value") if isinstance(tax_rate, dict) else tax_rate,
        "invoice_number": invoice.get("value") if isinstance(invoice, dict) else invoice,
        "address": address.get("value") if isinstance(address, dict) else address,
        "total": amount.get("value") if isinstance(amount, dict) else amount,
        "paid_date": paid_date.get("value") if isinstance(paid_date, dict) else paid_date,
        "payment_method": payment.get("value") if isinstance(payment, dict) else payment,
        "pred_category": label,
        "pred_score": score,
        "status": "predicted",
        "model_version": version_name,
        "raw_text": extracted.get("raw_text"),
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest) -> PredictResponse:
    image_source = payload.image_source()

    try:
        extracted = ocr_extract.extract_all(image_source, payload.hint)
    except ocr_extract.ImageFetchError as exc:
        raise HTTPException(status_code=502, detail="fetch_failed") from exc
    except ocr_extract.OCRDecodeError as exc:
        raise HTTPException(status_code=422, detail="decode_failed") from exc
    except ocr_extract.OCRServiceError as exc:
        LOGGER.exception("OCR service failure: %s", exc)
        raise HTTPException(status_code=500, detail="internal_error") from exc
    except Exception as exc:  # pragma: no cover - unexpected
        LOGGER.exception("Unexpected OCR failure: %s", exc)
        raise HTTPException(status_code=500, detail="internal_error") from exc

    model_bundle: Optional[ModelBundle]
    model_bundle, version_name = model_store.load_latest_model()
    label, score, alternatives = predict_category(extracted, model_bundle)

    receipt_id: Optional[str] = None
    for _ in range(5):
        doc_id = _generate_doc_id()
        receipt_id = _lookup_receipt_by_doc_id(doc_id)
        if not receipt_id:
            break
    else:  # pragma: no cover - extremely unlikely collision loop
        raise HTTPException(status_code=500, detail="internal_error")

    receipt_payload = _prepare_receipt_payload(
        doc_id=doc_id,
        extracted=extracted,
        image_url=payload.image_url,
        label=label,
        score=score,
        version_name=version_name,
    )
    try:
        if receipt_id:
            bubble_client.bubble_update("Receipt", receipt_id, receipt_payload)
        else:
            response = bubble_client.bubble_create("Receipt", receipt_payload)
            receipt_id = (
                response.get("response", {}).get("id")
                if isinstance(response.get("response"), dict)
                else response.get("id")
            )
    except Exception as exc:  # pragma: no cover - network operations
        LOGGER.warning("Receipt upsert failed: %s", exc)

    category_payload = {
        "pred": label,
        "score": score,
        "alternatives": alternatives,
    }
    return PredictResponse(
        doc_id=doc_id,
        extracted=extracted,
        category=category_payload,
        model_version=version_name,
        receipt_id=receipt_id,
    )


@app.post("/feedback")
async def feedback(payload: FeedbackPayload) -> Dict[str, Any]:
    receipt_id = payload.receipt_id
    if not receipt_id and payload.doc_id:
        try:
            constraints = [{"key": "doc_id", "constraint_type": "equals", "value": payload.doc_id}]
            search = bubble_client.bubble_search("Receipt", constraints=constraints, limit=1)
            results = search.get("response", {}).get("results") if isinstance(search.get("response"), dict) else search.get("results")
            if isinstance(results, list) and results:
                receipt_id = results[0].get("_id") or results[0].get("id")
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Receipt lookup failed: %s", exc)

    feedback_payload = {
        "receipt": receipt_id,
        "category_correct": payload.correct.get("category"),
        "vendor_correct": payload.correct.get("vendor"),
        "date_correct": payload.correct.get("date"),
        "total_correct": payload.correct.get("total"),
        "reason": payload.reason,
    }
    try:
        bubble_client.bubble_create("Feedback", feedback_payload)
    except Exception as exc:  # pragma: no cover
        LOGGER.error("Failed to store feedback: %s", exc)
        raise HTTPException(status_code=502, detail="Bubble feedback create failed") from exc

    if receipt_id:
        try:
            bubble_client.bubble_update("Receipt", receipt_id, {"status": "corrected"})
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed to update receipt status: %s", exc)

    return {"ok": True}


@app.post("/train")
async def train(payload: TrainRequest) -> Dict[str, Any]:
    cursor: Optional[str] = None
    samples: List[Dict[str, Any]] = []
    since_dt: Optional[datetime] = None
    if payload.since:
        try:
            since_dt = datetime.fromisoformat(payload.since.replace("Z", "+00:00"))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid since timestamp") from exc

    while True:
        try:
            response = bubble_client.bubble_search("Feedback", limit=100, cursor=cursor)
        except Exception as exc:  # pragma: no cover
            LOGGER.error("Feedback fetch failed: %s", exc)
            break

        container = response.get("response", {}) if isinstance(response.get("response"), dict) else response
        results = container.get("results") if isinstance(container, dict) else response.get("results")
        if not isinstance(results, list) or not results:
            break

        for row in results:
            created_text = row.get("Created Date") or row.get("created_date")
            if since_dt and created_text:
                try:
                    created_dt = datetime.fromisoformat(str(created_text).replace("Z", "+00:00"))
                except ValueError:
                    created_dt = None
                if created_dt and created_dt < since_dt:
                    continue
            samples.append(
                {
                    "text": row.get("raw_text") or row.get("category_correct"),
                    "label": row.get("category_correct") or "未分類",
                    "vendor": row.get("vendor_correct"),
                    "total": row.get("total_correct"),
                    "payment_method": row.get("payment_method"),
                }
            )
        cursor = container.get("cursor") if isinstance(container, dict) else response.get("cursor")
        if not cursor:
            break

    if len(samples) < payload.min_samples:
        return {"ok": False, "reason": "not_enough_samples", "collected": len(samples)}

    existing_model, _ = model_store.load_latest_model()
    model_bundle, metrics = partial_train(samples, existing_model)
    if not model_bundle:
        return {"ok": False, "reason": "training_skipped"}

    version_name = model_store.generate_version_name()
    created_id = model_store.save_model(model_bundle, version_name, metrics)
    return {"ok": True, "model_version": version_name, "model_id": created_id, "metrics": metrics}


__all__ = ["app"]
