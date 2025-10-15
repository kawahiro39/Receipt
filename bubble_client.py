"""Bubble Data API client utilities."""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional

import requests

BASE_URL = "https://system.vanlee.co.jp/version-test/api/1.1"
HEADERS = {
    "Authorization": "Bearer 99c107353970e7d1f2f1b36709cd3e04",
    "Content-Type": "application/json",
}
TIMEOUT = 20


def _request(method: str, url: str, **kwargs: Any) -> Dict[str, Any]:
    response = requests.request(method, url, headers=HEADERS, timeout=TIMEOUT, **kwargs)
    response.raise_for_status()
    return response.json()


def bubble_create(obj_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new Bubble object of the given type."""
    url = f"{BASE_URL}/obj/{obj_type}"
    return _request("POST", url, json=payload)


def bubble_update(obj_type: str, object_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Update an existing Bubble object by ID."""
    url = f"{BASE_URL}/obj/{obj_type}/{object_id}"
    return _request("PATCH", url, json=payload)


def bubble_get(obj_type: str, object_id: str) -> Dict[str, Any]:
    """Retrieve a Bubble object by ID."""
    url = f"{BASE_URL}/obj/{obj_type}/{object_id}"
    return _request("GET", url)


def bubble_search(
    obj_type: str,
    constraints: Optional[Iterable[Dict[str, Any]]] = None,
    limit: Optional[int] = None,
    cursor: Optional[str] = None,
) -> Dict[str, Any]:
    """Search for Bubble objects using optional constraints, limit, and cursor."""
    url = f"{BASE_URL}/obj/{obj_type}"

    params: Dict[str, Any] = {}
    if constraints is not None:
        params["constraints"] = json.dumps(list(constraints))
    if limit is not None:
        params["limit"] = limit
    if cursor is not None:
        params["cursor"] = cursor

    return _request("GET", url, params=params)


__all__ = [
    "bubble_create",
    "bubble_update",
    "bubble_get",
    "bubble_search",
]
