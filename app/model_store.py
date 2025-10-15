"""Persistence helpers for machine learning models stored in Bubble."""
from __future__ import annotations

import base64
import json
import logging
import pickle
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from . import bubble_client
from .settings import Settings, get_settings

LOGGER = logging.getLogger(__name__)

MODEL_OBJECT = "ModelVersion"
DEFAULT_CHUNK_SIZE = 500_000


class ModelStoreError(RuntimeError):
    """Raised when model save/load operations fail."""


def _unwrap_response(response: Dict[str, Any]) -> Dict[str, Any]:
    container = response.get("response")
    if isinstance(container, dict):
        return container
    return response


def _extract_results(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    container = _unwrap_response(response)
    results = container.get("results") if isinstance(container, dict) else None
    if isinstance(results, list):
        return [item for item in results if isinstance(item, dict)]
    return []


def _decode_model(chunks: Iterable[str]) -> Any:
    payload = "".join(chunks)
    if not payload:
        raise ModelStoreError("empty_model_payload")
    try:
        data = base64.b64decode(payload)
        return pickle.loads(data)
    except Exception as exc:  # pragma: no cover - defensive
        raise ModelStoreError("model_decode_failed") from exc


def _collect_chunks(record: Dict[str, Any]) -> List[str]:
    chunks = record.get("chunks")
    if isinstance(chunks, list):
        return [chunk for chunk in chunks if isinstance(chunk, str)]
    return []


def load_latest_model(task: str, *, settings: Optional[Settings] = None) -> Tuple[Optional[Any], Optional[str], Optional[Dict[str, Any]]]:
    """Load the most recent model for ``task`` if available."""

    settings = settings or get_settings()
    try:
        response = bubble_client.bubble_search(
            MODEL_OBJECT,
            constraints=[
                {"key": "task", "constraint_type": "equals", "value": task},
                {"key": "is_latest", "constraint_type": "equals", "value": "yes"},
            ],
            limit=1,
            settings=settings,
        )
    except Exception as exc:  # pragma: no cover - network error logging
        LOGGER.warning("Model lookup failed: task=%s error=%s", task, exc)
        return None, None, None

    results = _extract_results(response)
    if not results:
        return None, None, None

    record = results[0]
    chunks = _collect_chunks(record)
    if not chunks:
        return None, record.get("_id") or record.get("id"), record

    model = _decode_model(chunks)
    return model, record.get("_id") or record.get("id"), record


def _chunk_data(data: bytes, chunk_size: int = DEFAULT_CHUNK_SIZE) -> List[str]:
    encoded = base64.b64encode(data).decode("ascii")
    return [encoded[i : i + chunk_size] for i in range(0, len(encoded), chunk_size)] or [""]


def _ensure_json(value: Optional[Dict[str, Any]]) -> str:
    if not value:
        return json.dumps({})
    return json.dumps(value, ensure_ascii=False)


def save_model(
    task: str,
    model: Any,
    *,
    metrics: Optional[Dict[str, Any]] = None,
    meta: Optional[Dict[str, Any]] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    settings: Optional[Settings] = None,
) -> str:
    """Serialize and store a model for ``task`` returning the created id."""

    settings = settings or get_settings()
    try:
        blob = pickle.dumps(model)
    except Exception as exc:  # pragma: no cover - pickle failure
        raise ModelStoreError("model_serialize_failed") from exc

    chunks = _chunk_data(blob, chunk_size=chunk_size)
    payload: Dict[str, Any] = {
        "task": task,
        "chunks": chunks,
        "is_latest": "yes",
        "metrics_json": json.dumps(metrics or {}, ensure_ascii=False),
        "meta_json": _ensure_json(meta),
    }

    response = bubble_client.bubble_create(MODEL_OBJECT, payload, settings=settings)
    container = _unwrap_response(response)
    created_id = container.get("id") if isinstance(container, dict) else response.get("id")
    if not isinstance(created_id, str):
        raise ModelStoreError("missing_created_id")

    _mark_previous_versions_not_latest(task, exclude_id=created_id, settings=settings)
    return created_id


def _mark_previous_versions_not_latest(task: str, *, exclude_id: str, settings: Settings) -> None:
    try:
        response = bubble_client.bubble_search(
            MODEL_OBJECT,
            constraints=[
                {"key": "task", "constraint_type": "equals", "value": task},
                {"key": "is_latest", "constraint_type": "equals", "value": "yes"},
            ],
            limit=100,
            settings=settings,
        )
    except Exception as exc:  # pragma: no cover - network error logging
        LOGGER.warning("Model search for is_latest reset failed: %s", exc)
        return

    for record in _extract_results(response):
        record_id = record.get("_id") or record.get("id")
        if not isinstance(record_id, str) or record_id == exclude_id:
            continue
        try:
            bubble_client.bubble_update(
                MODEL_OBJECT,
                record_id,
                {"is_latest": "no"},
                settings=settings,
            )
        except Exception as exc:  # pragma: no cover - network error logging
            LOGGER.warning("Failed to mark model as not latest: id=%s error=%s", record_id, exc)


def generate_version_name(task: str) -> str:
    now = datetime.now(timezone.utc).astimezone()
    return f"{task}-{now.strftime('%Y%m%d%H%M%S')}"


__all__ = [
    "ModelStoreError",
    "load_latest_model",
    "save_model",
    "generate_version_name",
]
