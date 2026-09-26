"""Minimal, dependency-free client for the TypeSafe AI System One API (Jev).

Wire protocol (mirrors the official ``typesafe-sdk`` Python package):

    POST {base_url}/v1/systemone
    Authorization: Bearer <TYPESAFE_API_KEY>
    {"state": <text or JSON>, "model": "jev-latest",
     "questions": {"<name>": {"type": "choice", "instructions": ..., "criteria": {"<label>": <description>}}}}

    200 -> {"model": ..., "usage": {"input_tokens": ..., "output_tokens": ...},
            "answers": {"<name>": {"type": "choice", "choice": "<label>",
                                   "confidence": 0.9, "probabilities": {"<label>": 0.9, ...}}}}

Configuration uses the same environment variables as the official SDK:
``TYPESAFE_API_KEY`` (required), ``TYPESAFE_BASE_URL``, ``TYPESAFE_DEFAULT_MODEL``.
"""

from __future__ import annotations

import json
import os
import random
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import __version__

API_KEY_ENV = "TYPESAFE_API_KEY"
BASE_URL_ENV = "TYPESAFE_BASE_URL"
MODEL_ENV = "TYPESAFE_DEFAULT_MODEL"
TIMEOUT_ENV = "TYPESAFE_TIMEOUT"

DEFAULT_BASE_URL = "https://api.typesafe.ai"
DEFAULT_MODEL = "jev-latest"
DEFAULT_TIMEOUT = 10.0

SYSTEM_ONE_PATH = "/v1/systemone"
MODELS_PATH = "/v1/models"
RETRY_STATUSES = frozenset({408, 429, *range(500, 600)})


class TypeSafeError(RuntimeError):
    """Any failure talking to the API (configuration, network, HTTP or bad response)."""


class TypeSafeAPIError(TypeSafeError):
    def __init__(self, status: int, message: str, request_id: str | None = None):
        super().__init__(f"HTTP {status}: {message}" + (f" (request {request_id})" if request_id else ""))
        self.status = status
        self.request_id = request_id


# ---------------------------------------------------------------- configuration

def load_dotenv(*paths: str | Path) -> list[Path]:
    """Load KEY=VALUE lines from .env files into os.environ (never overriding real env vars)."""
    loaded = []
    for p in paths:
        path = Path(p)
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip().removeprefix("export ").strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            if key and key not in os.environ:
                os.environ[key] = value
        loaded.append(path)
    return loaded


@dataclass(frozen=True)
class TypeSafeConfig:
    api_key: str = field(repr=False)
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout: float = DEFAULT_TIMEOUT
    max_retries: int = 2

    @staticmethod
    def from_env(model: str | None = None, env: Mapping[str, str] | None = None) -> "TypeSafeConfig":
        env = os.environ if env is None else env
        key = env.get(API_KEY_ENV, "").strip()
        if not key or key.startswith("<"):
            raise TypeSafeError(
                f"{API_KEY_ENV} is not set. Copy .env.example to .env and put your TypeSafe API key in it, "
                f"or export {API_KEY_ENV}=...")
        if not key.isascii() or not key.isprintable() or " " in key:
            raise TypeSafeError(f"{API_KEY_ENV} must be printable ASCII without spaces")
        return TypeSafeConfig(
            api_key=key,
            base_url=(env.get(BASE_URL_ENV, "").strip() or DEFAULT_BASE_URL).rstrip("/"),
            model=model or env.get(MODEL_ENV, "").strip() or DEFAULT_MODEL,
            timeout=float(env.get(TIMEOUT_ENV, "").strip() or DEFAULT_TIMEOUT),
        )


# ---------------------------------------------------------------- client

@dataclass(frozen=True)
class SystemOneResult:
    answers: dict[str, dict[str, Any]]
    model: str
    usage: dict[str, Any]
    request_id: str | None
    latency_s: float       # wall time of the successful attempt
    total_s: float         # wall time including retries and backoff
    attempts: int


class TypeSafeClient:
    def __init__(self, config: TypeSafeConfig):
        self.config = config

    def system_one(self, state: Any, questions: Mapping[str, Any], model: str | None = None) -> SystemOneResult:
        body = {"state": state, "model": model or self.config.model, "questions": dict(questions)}
        started = time.monotonic()
        data, headers, latency, attempts = self._request("POST", SYSTEM_ONE_PATH, body)
        answers = data.get("answers") if isinstance(data, dict) else None
        if not isinstance(answers, dict):
            raise TypeSafeError(f"response has no 'answers' object: {str(data)[:200]}")
        return SystemOneResult(
            answers=answers,
            model=str(data.get("model", body["model"])),
            usage=data.get("usage") or {},
            request_id=headers.get("x-typesafe-request-id"),
            latency_s=latency,
            total_s=time.monotonic() - started,
            attempts=attempts,
        )

    def list_models(self) -> list[dict[str, Any]]:
        data, _, _, _ = self._request("GET", MODELS_PATH, None)
        return list(data.get("models", [])) if isinstance(data, dict) else []

    # ------------------------------------------------------------ transport

    def _request(self, method: str, path: str, body: Any):
        cfg = self.config
        content = None if body is None else json.dumps(body).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {cfg.api_key}",
            "Accept": "application/json",
            "User-Agent": f"blinddrive/{__version__}",
        }
        if content is not None:
            headers["Content-Type"] = "application/json"

        attempt = 0
        while True:
            req = urllib.request.Request(cfg.base_url + path, data=content, headers=dict(headers), method=method)
            if attempt:
                req.add_header("X-TypeSafe-Retry-Count", str(attempt))
            t0 = time.monotonic()
            retry_after = None
            try:
                with urllib.request.urlopen(req, timeout=cfg.timeout) as resp:
                    raw = resp.read()
                    latency = time.monotonic() - t0
                    resp_headers = {k.lower(): v for k, v in resp.headers.items()}
                try:
                    return json.loads(raw), resp_headers, latency, attempt + 1
                except ValueError:
                    raise TypeSafeError(f"response is not JSON: {raw[:200]!r}") from None
            except urllib.error.HTTPError as err:
                resp_headers = {k.lower(): v for k, v in (err.headers or {}).items()}
                error = TypeSafeAPIError(err.code, _error_message(err.read()), resp_headers.get("x-typesafe-request-id"))
                retryable = err.code in RETRY_STATUSES
                retry_after = _retry_after(resp_headers)
            except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as err:
                reason = getattr(err, "reason", err)
                error = TypeSafeError(f"cannot reach {cfg.base_url}: {reason}")
                retryable = True
            if not retryable or attempt >= cfg.max_retries:
                raise error
            attempt += 1
            delay = retry_after if retry_after is not None else min(5.0, 0.5 * 2 ** (attempt - 1))
            time.sleep(delay * (1 - 0.25 * random.random()) if retry_after is None else delay)


def _error_message(raw: bytes) -> str:
    try:
        body = json.loads(raw)
    except ValueError:
        return raw[:200].decode("utf-8", "replace") or "(no body)"
    if isinstance(body, dict):
        for key in ("message", "detail", "error"):
            value = body.get(key)
            if isinstance(value, dict):
                value = value.get("message", value)
            if value:
                return str(value)[:300]
    return str(body)[:300]


def _retry_after(headers: Mapping[str, str]) -> float | None:
    try:
        if "retry-after-ms" in headers:
            return max(0.0, float(headers["retry-after-ms"]) / 1000)
        if "retry-after" in headers:
            return max(0.0, float(headers["retry-after"]))
    except ValueError:
        pass
    return None
