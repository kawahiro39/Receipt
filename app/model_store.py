"""Model persistence utilities backed exclusively by the Bubble Data API.

The service runs on Cloud Run, which offers only ephemeral local storage.  In
order to keep both training and inference stateless, trained scikit-learn
artifacts are serialized into Bubble's ``ModelVersion`` data type.  This module
encapsulates that interaction so the rest of the application can treat the
model store as a simple load/save interface.
"""

from __future__ import annotations

import base64
import json
import logging
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from . import bubble_client
from .classifier import ModelBundle

LOGGER = logging.getLogger(__name__)
MODEL_TYPE = "ModelVersion"
CHUNK_PREFIX = "model_blob_b64"
CHUNK_SIZE = 200_000


class ModelStoreError(RuntimeError):
    """Raised when model persistence fails."""


def _extract_results(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    container = response.get("response") or response
    if isinstance(container, dict):
        results = container.get("results") or []
    else:
        results = []
    if not isinstance(results, list):
        return []
    return [item for item in results if isinstance(item, dict)]


def _collect_blob(record: Dict[str, Any]) -> str:
    parts: List[str] = []
    head = record.get(CHUNK_PREFIX)
    if isinstance(head, str):
        parts.append(head)
    index = 1
    while True:
        key = f"{CHUNK_PREFIX}_part{index}"
        chunk = record.get(key)
        if not isinstance(chunk, str):
            break
        parts.append(chunk)
        index += 1
    return "".join(parts)


def load_latest_model() -> Tuple[Optional[ModelBundle], Optional[str]]:
    """Fetch and deserialize the most recent model version."""

    try:
        response = bubble_client.bubble_search(
            MODEL_TYPE,
            limit=1,
            sort_field="Created Date",
            descending=True,
        )
    except Exception as exc:  # pragma: no cover - defensive for network errors
        LOGGER.warning("Unable to query model versions: %s", exc)
        return None, None

    results = _extract_results(response)
    if not results:
        return None, None

    record = results[0]
    blob = _collect_blob(record)
    if not blob:
        return None, record.get("name")

    try:
        payload = base64.b64decode(blob)
        model = pickle.loads(payload)
    except Exception as exc:  # pragma: no cover
        LOGGER.error("Failed to deserialize model payload: %s", exc)
        raise ModelStoreError("invalid model payload") from exc

    if isinstance(model, ModelBundle):
        return model, record.get("name")

    LOGGER.error("Unexpected model type: %s", type(model))
    raise ModelStoreError("unexpected model type")


def _chunk_blob(blob: str) -> Dict[str, str]:
    if len(blob) <= CHUNK_SIZE:
        return {CHUNK_PREFIX: blob}

    payload: Dict[str, str] = {CHUNK_PREFIX: blob[:CHUNK_SIZE]}
    remainder = blob[CHUNK_SIZE:]
    index = 1
    while remainder:
        payload[f"{CHUNK_PREFIX}_part{index}"] = remainder[:CHUNK_SIZE]
        remainder = remainder[CHUNK_SIZE:]
        index += 1
    return payload


def save_model(model: ModelBundle, version_name: str, metrics: Dict[str, Any]) -> str:
    """Persist a model bundle into Bubble and return the created identifier."""

    try:
        blob = pickle.dumps(model)
    except Exception as exc:  # pragma: no cover
        raise ModelStoreError("unable to serialize model") from exc

    blob_b64 = base64.b64encode(blob).decode("ascii")
    payload: Dict[str, Any] = {
        "name": version_name,
        "metrics_json": json.dumps(metrics, ensure_ascii=False),
    }
    payload.update(_chunk_blob(blob_b64))

    response = bubble_client.bubble_create(MODEL_TYPE, payload)
    created_id = (
        response.get("response", {}).get("id")
        if isinstance(response.get("response"), dict)
        else response.get("id")
    )
    if isinstance(created_id, str):
        return created_id
    raise ModelStoreError("Bubble did not return a created id")


def generate_version_name(prefix: str = "sgd-tfidf") -> str:
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M")
    return f"{prefix}-{timestamp}"


__all__ = ["load_latest_model", "save_model", "generate_version_name", "ModelStoreError"]
