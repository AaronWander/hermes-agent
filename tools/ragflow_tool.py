"""RAGFlow integration tools.

This module exposes Hermes tools that call a running RAGFlow server over HTTP.

Design goals:
- Minimal coupling: treat RAGFlow as an external retrieval/QA backend.
- Hermes remains the conversation orchestrator (clarification + reporting).
- Tool returns structured JSON the model can cite and reason over.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import httpx

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)


def _env(name: str) -> str:
    return (os.getenv(name, "") or "").strip()


def check_ragflow_requirements() -> bool:
    """Toolset check: True when a base URL is configured."""
    return bool(_env("RAGFLOW_BASE_URL"))


def _build_headers(api_token: str | None) -> Dict[str, str]:
    headers = {"Accept": "text/event-stream"}
    if api_token:
        # RAGFlow accepts either "Bearer <token>" or a raw token string.
        # Prefer Bearer for clarity.
        headers["Authorization"] = f"Bearer {api_token}"
    return headers


def _iter_ragflow_sse_events(resp: httpx.Response) -> List[Dict[str, Any]]:
    """Parse RAGFlow SSE response into a list of decoded JSON payloads.

    RAGFlow streams lines like:
      data:{"code":0,"message":"","data":{...}}
      data:{"code":0,"message":"","data":true}
    """
    events: List[Dict[str, Any]] = []
    for raw_line in resp.iter_lines():
        if not raw_line:
            continue
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if not payload:
            continue
        try:
            obj = json.loads(payload)
        except Exception:
            # Keep non-JSON lines for debugging; don't fail the whole call.
            events.append({"raw": payload})
            continue
        if isinstance(obj, dict):
            events.append(obj)
    return events


def _extract_completion(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Reduce event stream to a single completion payload."""
    answer: str = ""
    reference: Any = []
    last_data: Any = None
    error_message: Optional[str] = None

    for ev in events:
        if not isinstance(ev, dict):
            continue
        code = ev.get("code")
        if code not in (None, 0):
            # RAGFlow uses {code, message, data}. Preserve message for users.
            msg = ev.get("message")
            if msg:
                error_message = str(msg)
        data = ev.get("data")
        # Terminal event is typically data=true
        if data is True:
            break
        if isinstance(data, dict):
            last_data = data
            if isinstance(data.get("answer"), str):
                answer = data.get("answer") or answer
            if "reference" in data:
                reference = data.get("reference")

    result: Dict[str, Any] = {
        "answer": answer,
        "reference": reference if reference is not None else [],
        "raw": last_data or {},
        "events": len(events),
    }
    if error_message:
        result["warning"] = error_message
    return result


RAGFLOW_COMPLETION_SCHEMA = {
    "name": "ragflow_completion",
    "description": (
        "Ask a configured RAGFlow Search App to answer a question using the knowledge base, "
        "returning the answer plus citations/references for grounding. "
        "Use this for RAG-based Q&A and to decide what clarifying questions to ask.\n\n"
        "Requires RAGFLOW_BASE_URL and usually RAGFLOW_API_TOKEN. "
        "Configure a default search app via RAGFLOW_SEARCH_ID."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "User question to ask the RAGFlow search app.",
            },
            "search_id": {
                "type": "string",
                "description": "Optional override for the RAGFlow search app ID. Defaults to env RAGFLOW_SEARCH_ID.",
            },
            "base_url": {
                "type": "string",
                "description": "Optional override for the RAGFlow base URL (e.g. http://localhost:9380). Defaults to env RAGFLOW_BASE_URL.",
            },
            "api_token": {
                "type": "string",
                "description": "Optional override auth token. Defaults to env RAGFLOW_API_TOKEN.",
            },
            "timeout_seconds": {
                "type": "number",
                "description": "HTTP timeout seconds. Defaults to env RAGFLOW_TIMEOUT_SECONDS or 60.",
            },
        },
        "required": ["question"],
    },
}


def ragflow_completion(
    *,
    question: str,
    search_id: Optional[str] = None,
    base_url: Optional[str] = None,
    api_token: Optional[str] = None,
    timeout_seconds: Optional[float] = None,
    task_id: str | None = None,
) -> str:
    try:
        q = (question or "").strip()
        if not q:
            return tool_error("ragflow_completion: `question` is required.")

        base = (base_url or _env("RAGFLOW_BASE_URL")).rstrip("/")
        if not base:
            return tool_error("ragflow_completion: RAGFLOW_BASE_URL is not configured.")

        sid = (search_id or _env("RAGFLOW_SEARCH_ID")).strip()
        if not sid:
            return tool_error("ragflow_completion: `search_id` is required (or set RAGFLOW_SEARCH_ID).")

        token = (api_token or _env("RAGFLOW_API_TOKEN")).strip() or None

        if timeout_seconds is None:
            t = _env("RAGFLOW_TIMEOUT_SECONDS")
            timeout_seconds = float(t) if t else 60.0

        url = f"{base}/api/v1/searches/{sid}/completion"
        body = {"question": q}

        timeout = httpx.Timeout(timeout_seconds)
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=body, headers=_build_headers(token))
            resp.raise_for_status()
            events = _iter_ragflow_sse_events(resp)
            completion = _extract_completion(events)

        return json.dumps(
            {
                "success": True,
                "question": q,
                "search_id": sid,
                "result": completion,
            },
            ensure_ascii=False,
        )
    except httpx.HTTPStatusError as e:
        text = ""
        try:
            text = e.response.text
        except Exception:
            pass
        logger.warning("ragflow_completion HTTP error: %s", e)
        return json.dumps(
            {"success": False, "error": f"HTTP {e.response.status_code}: {str(e)}", "details": text[:2000]},
            ensure_ascii=False,
        )
    except Exception as e:
        logger.exception("ragflow_completion failed: %s", e)
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


def _handle_ragflow_completion(args: Dict[str, Any], **kw: Any) -> str:
    return ragflow_completion(
        question=args.get("question", ""),
        search_id=args.get("search_id"),
        base_url=args.get("base_url"),
        api_token=args.get("api_token"),
        timeout_seconds=args.get("timeout_seconds"),
        task_id=kw.get("task_id"),
    )


registry.register(
    name="ragflow_completion",
    toolset="ragflow",
    schema=RAGFLOW_COMPLETION_SCHEMA,
    handler=_handle_ragflow_completion,
    check_fn=check_ragflow_requirements,
    requires_env=["RAGFLOW_BASE_URL", "RAGFLOW_SEARCH_ID"],
    emoji="📚",
)

