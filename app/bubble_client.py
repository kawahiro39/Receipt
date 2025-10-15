"""Bubble Data API client utilities.

This module centralizes HTTP interactions with Bubble's Data API while keeping
request defaults (base URL, credentials, headers, timeout) in one place. The
API key and base URL can be overridden via the ``BUBBLE_API_KEY`` and
``BUBBLE_API_BASE`` environment variables respectively, enabling secret
management in deployment environments such as Cloud Run. ``BUBBLE_API_BASE``
may include either ``.../api/1.1`` or ``.../api/1.1/obj``—the client
normalizes either format to avoid generating duplicate ``/obj`` segments in
request URLs.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, Mapping, Optional

import requests

DEFAULT_BASE_URL = "https://system.vanlee.co.jp/version-test/api/1.1/obj"
DEFAULT_API_KEY = "99c107353970e7d1f2f1b36709cd3e04"
TIMEOUT = 20

JsonDict = Dict[str, Any]


class BubbleClientError(RuntimeError):
    """Raised when Bubble API requests fail before reaching ``requests``."""


def _base_url() -> str:
    base = os.getenv("BUBBLE_API_BASE", DEFAULT_BASE_URL).strip()
    if not base:
        raise BubbleClientError("BUBBLE_API_BASE must not be empty")

    trimmed = base.rstrip("/")
    if trimmed.endswith("/obj"):
        trimmed = trimmed[: -len("/obj")]

    return trimmed


def _headers() -> Dict[str, str]:
    api_key = os.getenv("BUBBLE_API_KEY", DEFAULT_API_KEY).strip()
    if not api_key:
        raise BubbleClientError("BUBBLE_API_KEY must not be empty")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _request(method: str, path: str, *, params: Optional[Mapping[str, Any]] = None, json_body: Optional[Mapping[str, Any]] = None) -> JsonDict:
    url = f"{_base_url()}/{path.lstrip('/')}"
    response = requests.request(
        method,
        url,
        headers=_headers(),
        params=params,
        json=json_body,
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def bubble_create(type_name: str, payload: Mapping[str, Any]) -> JsonDict:
    """Create a new Bubble object of the given ``type_name``."""

    return _request("POST", f"obj/{type_name}", json_body=dict(payload))


def bubble_update(type_name: str, thing_id: str, payload: Mapping[str, Any]) -> JsonDict:
    """Update an existing Bubble object identified by ``thing_id``."""

    return _request("PATCH", f"obj/{type_name}/{thing_id}", json_body=dict(payload))


def bubble_get(type_name: str, thing_id: str) -> JsonDict:
    """Fetch a Bubble object by ``thing_id``."""

    return _request("GET", f"obj/{type_name}/{thing_id}")


def bubble_search(
    type_name: str,
    *,
    constraints: Optional[Iterable[Mapping[str, Any]]] = None,
    limit: Optional[int] = None,
    cursor: Optional[str] = None,
) -> JsonDict:
    """Search for Bubble objects with optional constraints, pagination limit, and cursor."""

    params: Dict[str, Any] = {}
    if constraints is not None:
        params["constraints"] = json.dumps(list(constraints))
    if limit is not None:
        params["limit"] = limit
    if cursor is not None:
        params["cursor"] = cursor

    return _request("GET", f"obj/{type_name}", params=params)


__all__ = [
    "bubble_create",
    "bubble_update",
    "bubble_get",
    "bubble_search",
    "BubbleClientError",
]
