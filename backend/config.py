"""Backend configuration for IdeaCanvas.

One place to pick where the LLM runs (mirrors antigravity-sdk examples/me/
local_models.py: the LiteRT model path is only needed if you explicitly choose
`litert`; the default is `openai`, a local OpenAI-compatible server, so no
`.litertlm` file is required):

  IDEA_BACKEND=openai  -> local OpenAI-compatible server (Ollama / LM Studio /
                         vLLM). DEFAULT. Needs a server at OPENAI_BASE_URL.
  IDEA_BACKEND=litert  -> on-device LiteRT-LM (.litertlm, run `litert-lm import`
                         first; only then is GEMMA4_MODEL_PATH used).
  IDEA_BACKEND=cloud   -> Google Antigravity cloud (GEMINI_API_KEY or Vertex env).

All modules call :func:`build_agent_config` instead of constructing a config
class directly, so switching backends is a single env var.
"""
import contextlib
import logging
import os
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent

# Each egg gets its own worktree directory under here.
WORKTREE_ROOT = WORKSPACE / "worktrees"
WORKTREE_ROOT.mkdir(parents=True, exist_ok=True)

# Each Canvas "session" (one idea project) is a self-contained dated directory
# holding canvas.json, agent_data/ (SDK conversation persistence) and the eggs'
# worktrees/. Listing this dir == the user's saved idea sessions.
SESSIONS_DIR = WORKSPACE / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

# ---- Backend selection ----
# Default to the local OpenAI-compatible server (no .litertlm file required).
BACKEND = os.environ.get("IDEA_BACKEND", "openai").lower()


@contextlib.contextmanager
def quiet_sdk_start():
    """Suppress the SDK's own ERROR traceback when an agent fails to start.

    The SDK logs a verbose stack trace (even during exception unwinding) when a
    connection/model is unavailable; we only want our one-line status instead,
    so disable all logging briefly. ``logging.disable`` blocks every logger
    regardless of the SDK's own logger/handler setup.
    """
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(logging.NOTSET)

# cloud (Google Antigravity)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
USE_VERTEX = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true")
PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
CLOUD_MODEL = os.environ.get("ANTIGRAVITY_MODEL", "")

# litert (on-device)
GEMMA4_MODEL_PATH = os.environ.get(
    "GEMMA4_MODEL_PATH",
    os.path.expanduser("~/.litert-lm/models/gemma4-26b/model.litertlm"),
)

# local OpenAI-compatible
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:9000/v1")


def _detect_openai_model(base_url, timeout=2.0):
    """Best-effort: use the first model a local OpenAI server actually serves.

    The default model name varies per deployment, so when OPENAI_MODEL is unset
    we ask the server's /v1/models endpoint instead of guessing.
    """
    import json
    import urllib.request
    try:
        url = base_url.rstrip("/") + "/models"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        for m in data.get("data", []):
            name = m.get("id") or m.get("model") or m.get("name")
            if name:
                return name
    except Exception:
        return None
    return None


OPENAI_MODEL = (
    os.environ.get("OPENAI_MODEL")
    or _detect_openai_model(OPENAI_BASE_URL)
    or "gemma-4-26B-A4B-it"
)


def build_agent_config(
    system_instructions,
    *,
    tools=None,
    policies=None,
    capabilities=None,
    subagents=None,
    response_schema=None,
    conversation_id=None,
    workspaces=None,
    app_data_dir=None,
    save_dir=None,
    hook_runner=None,
):
    """Return an AgentConfig for the selected backend.

    Local backends get a CompactionConfig so small context windows stay healthy.
    The matching config class is imported on demand (so this module still imports
    without the SDK). Optional keys a given class does not accept are dropped
    automatically so the same call works across all backends.
    """
    base = {"system_instructions": system_instructions}
    if tools:
        base["tools"] = tools
    if policies:
        base["policies"] = policies
    if capabilities is not None:
        base["capabilities"] = capabilities
    if subagents:
        base["subagents"] = subagents
    if response_schema is not None:
        base["response_schema"] = response_schema
    if conversation_id:
        base["conversation_id"] = conversation_id
    if workspaces:
        base["workspaces"] = workspaces
    if app_data_dir:
        base["app_data_dir"] = app_data_dir
    if save_dir:
        base["save_dir"] = save_dir
    if hook_runner is not None:
        base["hook_runner"] = hook_runner

    if BACKEND == "litert":
        from google.antigravity import LiteRTAgentConfig
        from google.antigravity import types
        base["model_path"] = os.path.abspath(GEMMA4_MODEL_PATH)
        base["compaction_config"] = types.CompactionConfig(
            token_threshold=15360,
            checkpoint_interval_tokens=10240,
            max_context_tokens=40960,
        )
        cls = LiteRTAgentConfig
    elif BACKEND == "openai":
        from google.antigravity import LocalOpenAIAgentConfig
        from google.antigravity import types
        base["model"] = OPENAI_MODEL
        base["base_url"] = OPENAI_BASE_URL
        base["compaction_config"] = types.CompactionConfig(
            token_threshold=15360,
            checkpoint_interval_tokens=10240,
            max_context_tokens=40960,
        )
        cls = LocalOpenAIAgentConfig
    else:  # cloud
        from google.antigravity import LocalAgentConfig
        if GEMINI_API_KEY:
            base["api_key"] = GEMINI_API_KEY
        if USE_VERTEX:
            base["vertex"] = True
            base["project"] = PROJECT
            base["location"] = LOCATION
        if CLOUD_MODEL:
            base["model"] = CLOUD_MODEL
        cls = LocalAgentConfig

    try:
        return cls(**base)
    except TypeError:
        # Drop optional keys the concrete class may not accept, then retry.
        for k in ("workspaces", "app_data_dir", "save_dir", "subagents",
                  "response_schema", "conversation_id", "compaction_config",
                  "hook_runner"):
            base.pop(k, None)
        return cls(**base)
