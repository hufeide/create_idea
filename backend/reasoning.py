"""Shared 💭 thinking capture for local OpenAI-compatible backends.

The SDK's local OpenAI harness DISCARDS the model's ``reasoning_content``, so
the thinking never reaches ``resp.chunks``. To surface it we stream a
completion straight to the server ourselves and record the deltas as 💭
thoughts. ``supported`` is auto-detected (None = unknown) so we stop wasting an
extra call once we learn the model doesn't reason.

Both the eggs and the 总指挥 (Commander) use this, so each of them can show
thinking in its 轨迹 (trace).
"""
import asyncio
import json
import os

from .config import BACKEND, OPENAI_MODEL, OPENAI_BASE_URL

_REASONING = {"enabled": BACKEND == "openai", "supported": None}

# The capture call is a THROWAWAY probe: we only want the model's thinking
# prefix, never the answer. Cap the generation so it can't run for minutes,
# and stop as soon as the model starts producing real content.
_MAX_REASON_TOKENS = int(os.environ.get("REASON_MAX_TOKENS", "1024"))
_MAX_REASON_CHARS = int(os.environ.get("REASON_MAX_CHARS", "8000"))


def set_reasoning_capture(on):
    _REASONING["enabled"] = bool(on)
    _REASONING["supported"] = None  # re-probe when re-enabled


def reasoning_capture_state():
    return {"enabled": _REASONING["enabled"], "supported": _REASONING["supported"]}


def reasoning_enabled():
    """True when a reasoning-capture call should be attempted."""
    return bool(_REASONING["enabled"]) and _REASONING["supported"] is not False


async def capture_reasoning(system_instructions, text):
    """Stream the model's ``reasoning_content`` deltas (best-effort).

    Yields the raw thinking text. Never raises: a failure here must not break
    the real model call that follows.
    """
    if not (OPENAI_MODEL and OPENAI_BASE_URL):
        return
    import httpx  # available with the stack; imported lazily to stay safe
    payload = {
        "model": OPENAI_MODEL,
        "stream": True,
        # Cap the throwaway probe — we only need the thinking prefix.
        "max_tokens": _MAX_REASON_TOKENS,
        "messages": [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": text},
        ],
    }
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    url = OPENAI_BASE_URL.rstrip("/") + "/chat/completions"
    got = False
    total = 0
    finished = False  # probe ran to completion (not cancelled / not a net error)
    try:
        async with httpx.AsyncClient(timeout=None) as cli:
            async with cli.stream("POST", url, json=payload, headers=headers) as r:
                async for line in r.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except Exception:
                        continue
                    stop = False
                    for ch in obj.get("choices", []):
                        d = ch.get("delta") or {}
                        rc = d.get("reasoning_content") or d.get("reasoning")
                        if rc:
                            got = True
                            total += len(rc)
                            yield rc
                        elif d.get("content"):
                            # The model moved on to the ANSWER: no more
                            # thinking is coming, so stop the probe here instead
                            # of streaming the whole answer a second time.
                            stop = True
                    if stop or total >= _MAX_REASON_CHARS:
                        finished = True
                        return
                # The stream ended on its own: the probe completed.
                finished = True
    except asyncio.CancelledError:
        raise
    except Exception:
        return  # best-effort: never break the actual run
    finally:
        # Only a COMPLETED probe says anything about support: a stream that ran
        # to its end with no reasoning means the model doesn't reason. A probe
        # that was cancelled (e.g. _REASON_TIMEOUT) or died on a transient
        # network error must NOT set supported=False — that would permanently
        # disable capture although the model may reason just fine.
        if finished:
            _REASONING["supported"] = got
