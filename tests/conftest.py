from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import fastapi  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - only triggered in test envs without FastAPI
    stub = types.ModuleType("fastapi")

    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: object) -> None:
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class FastAPI:
        def __init__(self, **_: object) -> None:
            self.routes = {}

        def post(self, path: str, **__: object):
            def decorator(func):
                self.routes[("POST", path)] = func
                return func

            return decorator

    stub.FastAPI = FastAPI  # type: ignore[attr-defined]
    stub.HTTPException = HTTPException  # type: ignore[attr-defined]
    sys.modules["fastapi"] = stub
