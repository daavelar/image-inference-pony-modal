from __future__ import annotations

import logging
import os
import time
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

CALLBACK_META_KEYS = ("callback_url", "external_id")


def validate_callback_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("callback_url must use http or https")
    if not parsed.netloc:
        raise ValueError("callback_url must include a host")
    return url.strip()


def extract_callback_meta(body: dict) -> tuple[dict, str | None, int | str | None]:
    if not isinstance(body, dict):
        raise ValueError("JSON body must be an object")

    payload = dict(body)
    callback_url = payload.pop("callback_url", None)
    external_id = payload.pop("external_id", None)

    if callback_url is not None:
        if not isinstance(callback_url, str):
            raise ValueError("callback_url must be a string")
        callback_url = validate_callback_url(callback_url)

    if external_id is not None and not isinstance(external_id, (int, str)):
        raise ValueError("external_id must be an integer or string")

    return payload, callback_url, external_id


def build_callback_payload(
    call_id: str,
    result: dict[str, Any],
    external_id: int | str | None,
) -> dict[str, Any]:
    if result.get("error"):
        payload: dict[str, Any] = {
            "status": "failed",
            "call_id": call_id,
            "error": str(result["error"]),
        }
    else:
        payload = {
            "status": "completed",
            "call_id": call_id,
            "url": result.get("url"),
            "key": result.get("key"),
            "seed": result.get("seed"),
            "duration_seconds": result.get("duration_seconds"),
            "cost": result.get("cost"),
        }

    if external_id is not None:
        payload["external_id"] = external_id

    return payload


def post_callback(callback_url: str, payload: dict[str, Any]) -> None:
    import httpx

    webhook_key = os.environ.get("MODAL_WEBHOOK_KEY", "").strip()
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if webhook_key:
        headers["X-Modal-Webhook-Key"] = webhook_key

    backoff_seconds = (2, 4, 8)
    last_error: Exception | None = None

    for attempt, delay in enumerate(backoff_seconds, start=1):
        try:
            response = httpx.post(
                callback_url,
                json=payload,
                headers=headers,
                timeout=30.0,
            )
            if response.is_success:
                logger.info(
                    "[callback] delivered status=%s call_id=%s attempt=%s",
                    payload.get("status"),
                    payload.get("call_id"),
                    attempt,
                )
                return
            last_error = RuntimeError(
                f"callback returned HTTP {response.status_code}: {response.text[:500]}"
            )
        except Exception as exc:
            last_error = exc

        logger.warning(
            "[callback] attempt %s failed for call_id=%s: %s",
            attempt,
            payload.get("call_id"),
            last_error,
        )
        time.sleep(delay)

    logger.error(
        "[callback] all retries failed for call_id=%s url=%s error=%s",
        payload.get("call_id"),
        callback_url,
        last_error,
    )


def deliver_callback_for_call(
    call_id: str,
    callback_url: str,
    external_id: int | str | None,
) -> None:
    import modal

    fc = modal.FunctionCall.from_id(call_id)
    try:
        result = fc.get()
    except Exception as exc:
        payload = build_callback_payload(
            call_id,
            {"error": str(exc)},
            external_id,
        )
    else:
        if not isinstance(result, dict):
            result = {"error": f"unexpected job result type: {type(result).__name__}"}
        payload = build_callback_payload(call_id, result, external_id)

    post_callback(callback_url, payload)
