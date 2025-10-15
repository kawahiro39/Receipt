"""FastAPI entrypoint for the Vanlee Receipt AI service."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import bubble_client, model_store, ocr_extract
from .classifier import ModelBundle, partial_train, predict_category

LOGGER = logging.getLogger(__name__)
app = FastAPI(title="Vanlee Receipt AI")


class PredictRequest(BaseModel):
    doc_id: str
    image_url: Optional[str] = None
    image_base64: Optional[str] = Field(default=None, description="Base64 encoded image")
    hint: Optional[Dict[str, Any]] = None

    def image_source(self) -> str:
        if self.image_url:
            return self.image_url
        if self.image_base64:
            return self.image_base64
        raise ValueError("Either image_url or image_base64 must be provided")


class PredictResponse(BaseModel):
    doc_id: str
    extracted: Dict[str, Any]
    category: Dict[str, Any]
    model_version: Optional[str]


class FeedbackPayload(BaseModel):
    receipt_id: Optional[str] = None
    doc_id: Optional[str] = None
    correct: Dict[str, Any]
    reason: Optional[str] = None


class TrainRequest(BaseModel):
    since: Optional[str] = None
    min_samples: int = Field(default=1, ge=0)


@app.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest) -> PredictResponse:
    try:
        image_source = payload.image_source()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    extracted = ocr_extract.extract_all(image_source, payload.hint)
    model_bundle: Optional[ModelBundle]
    model_bundle, version_name = model_store.load_latest_model()
    label, score, alternatives = predict_category(extracted, model_bundle)

    receipt_payload = {
        "doc_id": payload.doc_id,
        "raw_text": extracted.get("raw_text"),
        "vendor": extracted.get("vendor", {}).get("value") if isinstance(extracted.get("vendor"), dict) else extracted.get("vendor"),
        "date": extracted.get("date", {}).get("value") if isinstance(extracted.get("date"), dict) else extracted.get("date"),
        "total": extracted.get("total", {}).get("value") if isinstance(extracted.get("total"), dict) else extracted.get("total"),
        "tax": extracted.get("tax", {}).get("value") if isinstance(extracted.get("tax"), dict) else extracted.get("tax"),
        "payment_method": extracted.get("payment_method", {}).get("value") if isinstance(extracted.get("payment_method"), dict) else extracted.get("payment_method"),
        "pred_category": label,
        "pred_score": score,
        "status": "predicted",
        "model_version": version_name,
    }

    try:
        constraints = [{"key": "doc_id", "constraint_type": "equals", "value": payload.doc_id}]
        existing = bubble_client.bubble_search("Receipt", constraints=constraints, limit=1)
        results = existing.get("response", {}).get("results") if isinstance(existing.get("response"), dict) else existing.get("results")
        thing_id = None
        if isinstance(results, list) and results:
            first = results[0]
            thing_id = first.get("_id") or first.get("id")
        if thing_id:
            bubble_client.bubble_update("Receipt", thing_id, receipt_payload)
        else:
            bubble_client.bubble_create("Receipt", receipt_payload)
    except Exception as exc:  # pragma: no cover - network operations
        LOGGER.warning("Receipt upsert failed: %s", exc)

    category_payload = {
        "pred": label,
        "score": score,
        "alternatives": alternatives,
    }
    return PredictResponse(doc_id=payload.doc_id, extracted=extracted, category=category_payload, model_version=version_name)


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
