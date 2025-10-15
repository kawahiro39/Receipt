"""FastAPI router definitions for the receipt automation service."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, Field

from . import bubble_client, model_store
from .classifier import ModelBundle, partial_train, predict_category
from .field_extractors import amount, date, merchant
from .ocr_extract import ImageFetchError, OCRDecodeError, OCRResult, OCRServiceError, extract_ocr
from .security import IdempotencyStore, IdempotentResponse, verify_admin_token, verify_signature
from .settings import Settings, get_settings

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Receipt Intelligence Service")

MAX_UPLOAD_SIZE = 15 * 1024 * 1024
ALLOWED_MIME_TYPES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/tiff",
}


class IngestResponse(BaseModel):
    doc_id: str
    extracted: Dict[str, Any]
    candidates: Dict[str, Any]


class FeedbackRequest(BaseModel):
    doc_id: str
    patch: Dict[str, Any]
    field_scope: Optional[List[str]] = None


class FeedbackResponse(BaseModel):
    status: str = "ok"
    feedback_ids: List[str]
    updated_doc_id: str


class TrainRequest(BaseModel):
    task: str = Field(default="all", pattern="^(all|category|amount|date|merchant)$")
    min_samples: int = Field(default=50, ge=1)
    test_ratio: float = Field(default=0.1, ge=0.0, le=0.5)


class TrainResponse(BaseModel):
    trained: Dict[str, int]
    metrics: Dict[str, Any]
    model_version_ids: Dict[str, Optional[str]]

    model_config = {"protected_namespaces": ()}


class ModelCache:
    def __init__(self) -> None:
        self._category: Optional[Tuple[Optional[ModelBundle], Optional[str]]] = None

    def category(self) -> Tuple[Optional[ModelBundle], Optional[str]]:
        if self._category is None:
            model, record_id, record = model_store.load_latest_model("category")
            bundle = model if isinstance(model, ModelBundle) else None
            version_id: Optional[str] = None
            if isinstance(record, dict):
                version_id = record.get("_id") or record.get("id") or record_id
            else:
                version_id = record_id
            self._category = (bundle, version_id)
        return self._category

    def refresh_category(self) -> Tuple[Optional[ModelBundle], Optional[str]]:
        self._category = None
        return self.category()


model_cache = ModelCache()
idempotency_store = IdempotencyStore()


def _extract_id(response: Dict[str, Any]) -> Optional[str]:
    container = response.get("response")
    if isinstance(container, dict):
        identifier = container.get("id") or container.get("_id")
        if isinstance(identifier, str):
            return identifier
    identifier = response.get("id") or response.get("_id")
    if isinstance(identifier, str):
        return identifier
    return None


def _normalise_amount(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _prepare_receipt_payload(
    *,
    ocr_result: OCRResult,
    extracted: Dict[str, Any],
    candidates: Dict[str, Any],
    model_version_id: Optional[str],
    image_url: Optional[str],
    source: str,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "image_url": image_url,
        "raw_text": ocr_result.raw_text,
        "date": extracted.get("date"),
        "amount": extracted.get("amount"),
        "merchant": extracted.get("merchant"),
        "category": extracted.get("category"),
        "status": "predicted",
        "candidates_json": json.dumps(candidates, ensure_ascii=False),
        "ocr_confidence": ocr_result.confidence,
        "model_version_id": model_version_id,
        "source": source,
    }
    return payload


async def _read_upload(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="empty_file")
    if len(data) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="file_too_large")
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="unsupported_mime")
    return data


def _format_amount_candidates(candidates: List[amount.AmountCandidate]) -> List[Dict[str, Any]]:
    formatted = []
    for candidate in candidates:
        formatted.append(
            {
                "value": candidate.value,
                "raw_text": candidate.raw_text,
                "confidence": candidate.confidence,
            }
        )
    return formatted


def _format_category_candidates(entries: List[Tuple[str, float]]) -> List[Dict[str, Any]]:
    return [
        {"label": label, "confidence": float(score)}
        for label, score in entries
    ]


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    request: Request,
    settings: Settings = Depends(get_settings),
    file: UploadFile = File(...),
    image_url: Optional[str] = Form(None),
    source: str = Form("bubble-ui"),
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
    signature_header: Optional[str] = Header(default=None, alias="X-Bubble-Signature"),
) -> IngestResponse:
    cached = idempotency_store.get(idempotency_key)
    if cached:
        LOGGER.info("Returning cached response for idempotency key %s", idempotency_key)
        return IngestResponse(**cached.payload)

    if signature_header:
        raw_body = await request.body()
        verify_signature(raw_body, signature_header, settings)

    data = await _read_upload(file)

    try:
        ocr_result = extract_ocr(data, language=settings.ocr_language)
    except ImageFetchError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="image_fetch_failed") from exc
    except OCRDecodeError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="ocr_decode_failed") from exc
    except OCRServiceError as exc:
        LOGGER.exception("OCR service error: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="ocr_service_error") from exc

    amount_info = amount.extract_amount(ocr_result)
    date_info = date.extract_date(ocr_result)
    merchant_info = merchant.extract_merchant(ocr_result)

    model_bundle, version_id = model_cache.category()
    category_label, category_score, category_candidates = predict_category(
        {
            "raw_text": ocr_result.raw_text,
            "amount": amount_info.best.value if amount_info.best else None,
            "merchant": merchant_info.best.value if merchant_info.best else None,
        },
        model_bundle,
    )

    extracted = {
        "date": date_info.best.value.isoformat() if date_info.best and date_info.best.value else None,
        "amount": _normalise_amount(amount_info.best.value if amount_info.best else None),
        "merchant": merchant_info.best.value if merchant_info.best else None,
        "category": category_label,
    }

    candidates_payload = {
        "amount": _format_amount_candidates(amount_info.candidates),
        "category": _format_category_candidates(category_candidates),
    }

    receipt_payload = _prepare_receipt_payload(
        ocr_result=ocr_result,
        extracted=extracted,
        candidates=candidates_payload,
        model_version_id=version_id,
        image_url=image_url,
        source=source,
    )

    try:
        response = bubble_client.bubble_create("Receipt", receipt_payload, settings=settings)
    except bubble_client.BubbleAPIError as exc:
        LOGGER.error("Failed to write Receipt to Bubble: %s", exc)
        raise HTTPException(status_code=status.HTTP_424_FAILED_DEPENDENCY, detail="bubble_write_failed") from exc

    doc_id = _extract_id(response)
    if not doc_id:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="missing_doc_id")

    payload = {
        "doc_id": doc_id,
        "extracted": extracted,
        "candidates": candidates_payload,
    }
    idempotency_store.remember(idempotency_key, IdempotentResponse(doc_id=doc_id, payload=payload))
    return IngestResponse(**payload)


def _feedback_value(entry: Any) -> Tuple[str, Optional[str]]:
    if isinstance(entry, dict):
        value = entry.get("value")
        bbox = entry.get("bbox")
        bbox_json = json.dumps(bbox, ensure_ascii=False) if bbox is not None else None
        return str(value), bbox_json
    return str(entry), None


def _create_feedback_records(
    doc_id: str,
    patch: Dict[str, Any],
    *,
    settings: Settings,
    field_scope: Optional[List[str]] = None,
) -> List[str]:
    feedback_ids: List[str] = []
    targets = field_scope or list(patch.keys())
    for field_name, value in patch.items():
        if field_name not in targets:
            continue
        value_text, bbox_json = _feedback_value(value)
        payload = {
            "doc_id": doc_id,
            "field": field_name,
            "value_text": value_text,
            "bbox_json": bbox_json,
            "processed_at": None,
            "model_version_trained_on": None,
        }
        try:
            response = bubble_client.bubble_create("Feedback", payload, settings=settings)
        except bubble_client.BubbleAPIError as exc:
            LOGGER.error("Failed to create Feedback: %s", exc)
            raise HTTPException(status_code=status.HTTP_424_FAILED_DEPENDENCY, detail="bubble_feedback_failed") from exc
        created_id = _extract_id(response)
        if created_id:
            feedback_ids.append(created_id)
    return feedback_ids


@app.post("/feedback", response_model=FeedbackResponse)
async def post_feedback(payload: FeedbackRequest, settings: Settings = Depends(get_settings)) -> FeedbackResponse:
    if not payload.patch:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="empty_patch")

    receipt_update = dict(payload.patch)
    receipt_update["status"] = "corrected"

    try:
        bubble_client.bubble_update("Receipt", payload.doc_id, receipt_update, settings=settings)
    except bubble_client.BubbleAPIError as exc:
        LOGGER.error("Failed to update Receipt: %s", exc)
        raise HTTPException(status_code=status.HTTP_424_FAILED_DEPENDENCY, detail="bubble_receipt_update_failed") from exc

    feedback_ids = _create_feedback_records(
        payload.doc_id,
        payload.patch,
        settings=settings,
        field_scope=payload.field_scope,
    )

    return FeedbackResponse(feedback_ids=feedback_ids, updated_doc_id=payload.doc_id)


def _fetch_feedback(field_name: Optional[str], *, settings: Settings) -> List[Dict[str, Any]]:
    constraints: List[Dict[str, Any]] = [
        {"key": "processed_at", "constraint_type": "is_empty", "value": None}
    ]
    if field_name:
        constraints.append({"key": "field", "constraint_type": "equals", "value": field_name})

    cursor: Optional[str] = None
    results: List[Dict[str, Any]] = []
    while True:
        response = bubble_client.bubble_search(
            "Feedback",
            constraints=constraints,
            limit=100,
            cursor=cursor,
            settings=settings,
        )
        container = response.get("response") if isinstance(response.get("response"), dict) else response
        batch = container.get("results") if isinstance(container, dict) else response.get("results")
        if not isinstance(batch, list) or not batch:
            break
        results.extend([row for row in batch if isinstance(row, dict)])
        cursor = container.get("cursor") if isinstance(container, dict) else response.get("cursor")
        if not cursor:
            break
    return results


def _fetch_receipt(doc_id: str, *, settings: Settings) -> Dict[str, Any]:
    response = bubble_client.bubble_get("Receipt", doc_id, settings=settings)
    container = response.get("response")
    if isinstance(container, dict):
        return container
    if isinstance(response, dict):
        return response
    return {}


def _mark_feedback_processed(feedback_ids: List[str], *, model_version_id: Optional[str], settings: Settings) -> None:
    if not feedback_ids:
        return
    timestamp = datetime.now(timezone.utc).isoformat()
    for feedback_id in feedback_ids:
        try:
            bubble_client.bubble_update(
                "Feedback",
                feedback_id,
                {
                    "processed_at": timestamp,
                    "model_version_trained_on": model_version_id,
                },
                settings=settings,
            )
        except bubble_client.BubbleAPIError as exc:
            LOGGER.warning("Failed to mark feedback processed: id=%s error=%s", feedback_id, exc)


@app.post("/train", response_model=TrainResponse)
async def train(
    payload: TrainRequest,
    authorization: Optional[str] = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> TrainResponse:
    verify_admin_token(authorization, settings)

    tasks = [payload.task] if payload.task != "all" else ["category", "amount", "date", "merchant"]

    trained_counts: Dict[str, int] = {}
    metrics_map: Dict[str, Any] = {}
    model_ids: Dict[str, Optional[str]] = {}

    for task_name in tasks:
        field_map = {
            "category": "category",
            "amount": "amount",
            "date": "date",
            "merchant": "merchant",
        }
        field_name = field_map.get(task_name)
        feedback_rows = _fetch_feedback(field_name, settings=settings)
        trained_counts[task_name] = len(feedback_rows)
        if not feedback_rows:
            metrics_map[task_name] = {"skipped": True, "reason": "no_feedback"}
            model_ids[task_name] = None
            continue

        if task_name == "category":
            samples = []
            for row in feedback_rows:
                doc_id = row.get("doc_id")
                if not isinstance(doc_id, str):
                    continue
                receipt = _fetch_receipt(doc_id, settings=settings)
                samples.append(
                    {
                        "text": receipt.get("raw_text") or "",
                        "merchant": receipt.get("merchant"),
                        "amount": receipt.get("amount"),
                        "label": row.get("value_text"),
                    }
                )
            if len(samples) < payload.min_samples:
                metrics_map[task_name] = {
                    "skipped": True,
                    "reason": "not_enough_samples",
                    "n": len(samples),
                }
                model_ids[task_name] = None
                continue
            existing_model, _, _ = model_store.load_latest_model("category", settings=settings)
            bundle = existing_model if isinstance(existing_model, ModelBundle) else None
            model_bundle, metrics = partial_train(samples, bundle)
            if not model_bundle:
                metrics_map[task_name] = {"skipped": True, "reason": "training_failed"}
                model_ids[task_name] = None
                continue
            model_id = model_store.save_model(
                task_name,
                model_bundle,
                metrics={**metrics, "n": len(samples)},
                meta={"feedback_ids": [row.get("_id") or row.get("id") for row in feedback_rows]},
                settings=settings,
            )
            model_cache.refresh_category()
            model_ids[task_name] = model_id
            metrics_map[task_name] = metrics
            feedback_ids = [row.get("_id") or row.get("id") for row in feedback_rows if isinstance(row.get("_id") or row.get("id"), str)]
            _mark_feedback_processed(feedback_ids, model_version_id=model_id, settings=settings)
        else:
            values = [row.get("value_text") for row in feedback_rows if row.get("value_text")]
            if len(values) < payload.min_samples:
                metrics_map[task_name] = {
                    "skipped": True,
                    "reason": "not_enough_samples",
                    "n": len(values),
                }
                model_ids[task_name] = None
                continue
            model_payload = {
                "task": task_name,
                "values": values,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            model_id = model_store.save_model(
                task_name,
                model_payload,
                metrics={"n": len(values)},
                meta={"feedback_ids": [row.get("_id") or row.get("id") for row in feedback_rows]},
                settings=settings,
            )
            metrics_map[task_name] = {"n": len(values), "type": "rule"}
            model_ids[task_name] = model_id
            feedback_ids = [row.get("_id") or row.get("id") for row in feedback_rows if isinstance(row.get("_id") or row.get("id"), str)]
            _mark_feedback_processed(feedback_ids, model_version_id=model_id, settings=settings)

    return TrainResponse(trained=trained_counts, metrics=metrics_map, model_version_ids=model_ids)


@app.post("/bubble-write-test")
async def bubble_write_test(settings: Settings = Depends(get_settings)) -> Dict[str, Any]:
    payload = {"status": "predicted", "source": "bubble-write-test"}
    try:
        response = bubble_client.bubble_create("Receipt", payload, settings=settings)
    except bubble_client.BubbleAPIError as exc:
        raise HTTPException(status_code=status.HTTP_424_FAILED_DEPENDENCY, detail="bubble_write_failed") from exc
    doc_id = _extract_id(response)
    if not doc_id:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="missing_doc_id")
    return {"doc_id": doc_id}


__all__ = ["app"]
