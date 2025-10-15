"""Bubble Data API client implementation with error handling."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, Mapping, Optional

import requests
from requests import Response

from .settings import Settings, get_settings

LOGGER = logging.getLogger(__name__)

JsonDict = Dict[str, Any]


class BubbleAPIError(RuntimeError):
    """Raised when Bubble API operations fail."""

    def __init__(self, message: str, *, status_code: Optional[int] = None, response_text: Optional[str] = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text


def _build_base_url(settings: Settings) -> str:
    base = settings.bubble_api_base.rstrip("/")
    return f"{base}/obj"


def _headers(settings: Settings) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {settings.bubble_api_key}",
        "Content-Type": "application/json",
    }


def _handle_response(response: Response) -> JsonDict:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:  # pragma: no cover - network errors
        LOGGER.error(
            "Bubble API error: status=%s body=%s", response.status_code, response.text.strip()
        )
        raise BubbleAPIError(
            "bubble_api_error",
            status_code=response.status_code,
            response_text=response.text,
        ) from exc
    try:
        return response.json()
    except ValueError as exc:  # pragma: no cover - defensive
        raise BubbleAPIError("invalid_json_response", response_text=response.text) from exc


def _request(
    method: str,
    path: str,
    *,
    params: Optional[Mapping[str, Any]] = None,
    json_body: Optional[Mapping[str, Any]] = None,
    settings: Optional[Settings] = None,
    timeout: int = 30,
) -> JsonDict:
    settings = settings or get_settings()
    base_url = _build_base_url(settings)
    url = f"{base_url}/{path.lstrip('/')}"
    try:
        response = requests.request(
            method,
            url,
            headers=_headers(settings),
            params=params,
            json=json_body,
            timeout=timeout,
        )
    except requests.RequestException as exc:  # pragma: no cover - defensive
        LOGGER.error("Bubble API request failure: %s", exc)
        raise BubbleAPIError("request_failed") from exc
    return _handle_response(response)


def bubble_create(type_name: str, payload: Mapping[str, Any], *, settings: Optional[Settings] = None) -> JsonDict:
    return _request("POST", f"{type_name}", json_body=payload, settings=settings)


def bubble_update(
    type_name: str,
    thing_id: str,
    payload: Mapping[str, Any],
    *,
    settings: Optional[Settings] = None,
) -> JsonDict:
    return _request("PATCH", f"{type_name}/{thing_id}", json_body=payload, settings=settings)


def bubble_get(type_name: str, thing_id: str, *, settings: Optional[Settings] = None) -> JsonDict:
    return _request("GET", f"{type_name}/{thing_id}", settings=settings)


def bubble_search(
    type_name: str,
    *,
    constraints: Optional[Iterable[Mapping[str, Any]]] = None,
    limit: Optional[int] = None,
    cursor: Optional[str] = None,
    sort_field: Optional[str] = None,
    descending: bool = True,
    settings: Optional[Settings] = None,
) -> JsonDict:
    params: Dict[str, Any] = {}
    if constraints is not None:
        params["constraints"] = json.dumps(list(constraints))
    if limit is not None:
        params["limit"] = limit
    if cursor is not None:
        params["cursor"] = cursor
    if sort_field:
        params["sort_field"] = sort_field
        params["descending"] = "yes" if descending else "no"

    return _request("GET", f"{type_name}", params=params, settings=settings)


__all__ = [
    "BubbleAPIError",
    "bubble_create",
    "bubble_get",
    "bubble_search",
    "bubble_update",
]
