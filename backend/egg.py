"""One idea egg = one Antigravity SDK Agent scoped to its own worktree.

The egg holds a long-lived Agent session (history accumulates across turns)
and writes code into ``worktree`` via the SDK's built-in file/command tools.
When a model is connected (cloud key or local OpenAI/LiteRT backend) the egg
streams the model's real responses; if no model is reachable it reports that
explicitly instead of faking a completion.
"""
import asyncio
import datetime
import json
import os
from pathlib import Path

from .config import (WORKTREE_ROOT, build_agent_config, BACKEND, quiet_sdk_start,
                     OPENAI_MODEL, OPENAI_BASE_URL)

try:  # pragma: no cover - depends on installed SDK
    from google.antigravity import Agent, CapabilitiesConfig, types
    from google.antigravity.types import RunCommandConfig, HookResult
    from google.antigravity.hooks import policy, hooks as sdk_hooks
    from google.antigravity.hooks import hook_runner as _hr_mod
    SDK_OK = True
except Exception:  # pragma: no cover
    SDK_OK = False


class _LLMBox:
    """Holds retry policy for model chat() calls.

    Concurrency is governed PER-EGG by ``EggAgent._lock`` (so a single egg never
    runs two model turns at once), which lets DIFFERENT eggs truly run in
    parallel. We deliberately do NOT serialize chat() across all eggs — modern
    OpenAI-compatible backends handle concurrent requests. Failed attempts are
    retried every ``delay`` seconds (see the retry loop in ``EggAgent.run``).
    """

    def __init__(self, max_attempts=6, delay=1.0):
        self.max_attempts = max_attempts
        self.delay = delay


_LLM_BOX = _LLMBox()


# Per-turn safety timeout. If a model turn (e.g. a hung tool call / a stream that
# never ends) doesn't finish, abandon it so the box lock is released and the egg
# returns to 空闲 instead of being stuck 运行中 forever. Generous by default;
# override with EGG_TURN_TIMEOUT (seconds).
_TURN_TIMEOUT = float(os.environ.get("EGG_TURN_TIMEOUT", "900"))


# The SDK's local OpenAI harness DISCARDS the model's `reasoning_content`, so
# for local backends we capture the thinking directly via a streaming call and
# record it as 💭 thoughts. `supported` is auto-detected (None=unknown) so we
# stop wasting an extra call once we learn the model doesn't reason.
_REASONING = {"enabled": BACKEND == "openai", "supported": None}


def set_reasoning_capture(on):
    _REASONING["enabled"] = bool(on)
    _REASONING["supported"] = None  # re-probe when re-enabled


def reasoning_capture_state():
    return {"enabled": _REASONING["enabled"], "supported": _REASONING["supported"]}


def _chunk_to_text(chunk):
    """Convert an SDK response chunk into display text."""
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, types.Text):
        return chunk.text
    if isinstance(chunk, types.Thought):
        return f"💭 {chunk.text}\n"
    if isinstance(chunk, types.ToolCall):
        return f"\n🔧 调用工具：{chunk.name}\n"
    if isinstance(chunk, types.ToolResult):
        return "   ↳ 工具结果已处理\n"
    return ""


class EggAgent:
    def __init__(self, id, title, prompt, type_, position, subagents=None, parent=None,
                 session_dir=None, conversation_id=None):
        self.id = id
        self.title = title
        self.prompt = prompt
        self.type = type_
        self.x, self.y = position
        self.subagents = subagents or []
        self.parent = parent
        # Every egg belongs to a Canvas session; its code worktree and the SDK's
        # conversation persistence (save_dir) live under that session directory so
        # a whole idea project is self-contained under sessions/<timestamp>/.
        self.session_dir = Path(session_dir) if session_dir else WORKTREE_ROOT
        # Per-egg conversation id. None means "not yet established": on first
        # connect we DON'T pass it so the SDK creates a fresh conversation and
        # assigns an id (read back via self._agent.conversation_id). After that we
        # persist + pass this id so the SDK RESUMES the same session across turns
        # and restarts. Passing an id absent from save_dir makes the SDK raise
        # "conversation ... not found ... cannot resume".
        self.conversation_id = conversation_id
        # Top-level eggs (idea eggs) live directly under the session worktrees
        # dir; sub-eggs nest inside their parent's directory as sub-directories.
        self.worktree: Path = (self.session_dir / "worktrees" / parent / id
                               if parent else self.session_dir / "worktrees" / id)
        self.worktree.mkdir(parents=True, exist_ok=True)
        # touch a placeholder file so "review code" has something to show
        scope = f" (sub-egg of `{parent}`)" if parent else ""
        (self.worktree / "README.md").write_text(
            f"# {title}\n\n> worktree for egg `{id}`{scope}\n\n## Prompt\n{prompt}\n",
            encoding="utf-8",
        )
        self._lock = asyncio.Lock()
        self._cm = None
        self._agent = None
        self._start_attempted = False
        self._start_error = None
        self._recorded_init = False
        self._si = ""
        # Run state for the UI + interrupt: busy while a model turn streams, and
        # _task points at the task iterating run() so it can be cancelled.
        self.busy = False
        self._task = None
        # The active SDK ChatResponse while a model turn streams, so interrupt()
        # can send halt_request to the backend (not merely close the local reader).
        self._active_resp = None
        # Chronological conversation trace (user commands, model text/thoughts,
        # system prompt, tool calls/results) — viewable via /api/trace/{egg_id}.
        self.transcript = []

    # ---- serialization for the frontend ----
    def spec(self):
        return {
            "id": self.id, "title": self.title, "prompt": self.prompt,
            "type": self.type, "x": self.x, "y": self.y, "parent": self.parent,
            "conversation_id": self.conversation_id,
            "running": self.busy,
        }

    # ---- SDK lifecycle ----
    def _make_hook_runner(self):
        """Build a HookRunner whose tool hooks record into THIS egg's trace.

        Used by the cloud backend, which surfaces tool calls/results through
        these hooks. The local openai backend executes tools internally and does
        not dispatch them (verified), so per-call tool entries only appear there
        when such a backend is used.
        """
        if not SDK_OK:
            return None

        @sdk_hooks.pre_tool_call_decide
        def _tc(ctx: sdk_hooks.HookContext, data: types.ToolCall):
            self._record("assistant", "tool_call", name=getattr(data, "name", None),
                         args=getattr(data, "args", None))
            return HookResult(allow=True)

        @sdk_hooks.post_tool_call
        def _tr(ctx: sdk_hooks.HookContext, data: types.ToolResult):
            self._record("tool", "tool_result", name=getattr(data, "name", None),
                         result=getattr(data, "result", None),
                         error=getattr(data, "error", None))

        return _hr_mod.HookRunner(
            pre_tool_call_decide_hooks=[_tc],
            post_tool_call_hooks=[_tr],
        )

    def _config(self, resume=True):
        si = (
            f"You are an autonomous LLM module living inside an 'idea egg' named "
            f"'{self.title}'.\nYour module responsibility (prompt): {self.prompt}\n"
            f"Your code workspace is the directory: {self.worktree}\n"
            f"Implement your module by WRITING AND RUNNING code inside that "
            f"workspace using the available file/command tools. Create real source "
            f"files there. Coordinates with connected eggs only when instructed. "
            f"Be concise and show what you built."
        )
        # Only keep the minimal SWE toolset (write/run code) — drop the default
        # web/image/subagent tools that bloat the context and are unused here.
        # `finish` is always added by the harness.
        caps = types.CapabilitiesConfig(
            enabled_tools=types.BuiltinTools.minimal(),
            enable_subagents=False,
            agent_behavior=types.AgentBehavior.MINIMAL,
        )
        if BACKEND == "cloud":
            caps.run_command_config = RunCommandConfig(enable_sandbox=True)
        self._si = si
        return build_agent_config(
            si,
            workspaces=[str(self.worktree)],
            app_data_dir=str(self.worktree / "agent_data"),
            save_dir=str(self.session_dir / "agent_data"),
            # When resuming we pass the stable per-egg id so the SDK restores the
            # full LLM session; when it's gone we omit it (fresh conversation).
            conversation_id=(self.conversation_id if resume else None),
            capabilities=caps,
            policies=[policy.allow_all()],
            subagents=None,
            hook_runner=self._make_hook_runner(),
        )

    def _sync_conv_id(self):
        """Adopt the runtime-assigned conversation id (available only after the
        first message) so a recovered/fresh conversation resumes correctly on the
        next app start instead of retrying a stale id forever."""
        try:
            cid = self._agent.conversation_id if self._agent else None
        except Exception:
            cid = None
        if cid and cid != self.conversation_id:
            self.conversation_id = cid

    async def _open_agent(self, resume=True):
        """Build + enter an Agent, cleaning up if entering fails."""
        with quiet_sdk_start():
            cm = Agent(self._config(resume=resume))
            try:
                agent = await cm.__aenter__()
                return cm, agent
            except Exception:
                try:
                    await cm.__aexit__(None, None, None)
                except Exception:
                    pass
                raise

    async def start(self):
        if not SDK_OK:
            return
        try:
            self._cm, self._agent = await self._open_agent(resume=True)
            return
        except Exception as ex:
            # A stale conversation id — the session's agent_data was cleared, or
            # the egg is resumed under a fresh session dir — makes the SDK refuse
            # to resume ("... not found ... (cannot resume)"). Recover by starting
            # a FRESH conversation instead of leaving the egg permanently offline.
            if "cannot resume" in str(ex) or "not found" in str(ex).lower():
                try:
                    self._cm, self._agent = await self._open_agent(resume=False)
                    try:
                        self.conversation_id = (self._agent.conversation_id
                                                or self.conversation_id)
                    except Exception:
                        self.conversation_id = uuid.uuid4().hex
                    print(f"[egg {self.id}] stale conversation -> started fresh "
                          f"(id={self.conversation_id})")
                    return
                except Exception as ex2:
                    ex = ex2
            print(f"[egg {self.id}] agent start failed: {ex}")
            self._cm = None
            self._agent = None
            self._start_error = ex

    async def stop(self):
        if self._cm:
            try:
                await self._cm.__aexit__(None, None, None)
            except Exception:
                pass
        self._cm = None
        self._agent = None

    # ---- interactions ----
    async def _ensure_started(self):
        """Lazily connect to the model on first use; cache failure per process."""
        if self._agent is not None:
            return True
        if self._start_attempted:
            return False
        self._start_attempted = True
        await self.start()
        # If the SDK created a brand-new conversation (we did NOT pass a
        # conversation_id to resume), capture the id it assigned so later turns /
        # process restarts resume THIS session instead of failing with
        # "conversation ... not found ... cannot resume".
        if self.conversation_id is None and self._agent is not None:
            cid = getattr(self._agent, "conversation_id", None)
            if cid:
                self.conversation_id = cid
        # Recover from a missing persisted conversation: if resume failed because
        # the conversation_id isn't in save_dir, drop it and reconnect so the SDK
        # starts a FRESH conversation instead of hard-failing every turn.
        if self._agent is None and self._start_error is not None and self.conversation_id is not None:
            err = str(self._start_error)
            if "conversation" in err and ("not found" in err or "cannot resume" in err
                                          or "resume" in err):
                self.conversation_id = None
                self._start_attempted = False
                self._start_error = None
                await self.start()
                if self.conversation_id is None and self._agent is not None:
                    cid = getattr(self._agent, "conversation_id", None)
                    if cid:
                        self.conversation_id = cid
        if self._agent is not None and not self._recorded_init:
            self._recorded_init = True
            # Capture the FULL system prompt and the tool set into the trace so
            # the trace shows every prompt + tool, not just the conversation.
            self._record("system", "system_prompt", text=self._si)
            tools = []
            for t in types.BuiltinTools.minimal():
                tools.append(getattr(t, "name", str(t)))
            tool_line = "enabled: " + (", ".join(tools) or "（无）") + " (+ finish)"
            if BACKEND == "cloud":
                tool_line += " [cloud: run_command sandbox]"
            self._record("system", "tools", text=tool_line)
        return self._agent is not None

    def _offline_msg(self, action):
        msg = f"⚠ 该蛋的 LLM 未连接，无法{action}。"
        if self._start_error:
            err = str(self._start_error)
            msg += f"\n原因：{err}"
            # A recoverable session-state problem (persisted conversation_id missing
            # in save_dir) should NOT be misreported as a backend/model outage.
            if "conversation" in err and ("not found" in err or "cannot resume" in err
                                          or "resume" in err):
                msg += "\n（会话状态文件丢失；下次对话将自动以新会话开始。）"
                return msg
        msg += "\n请确认 IDEA_BACKEND 与本地/云端模型可用（openai 后端需 :9000 服务在跑）。"
        return msg

    # ---- conversation trace ----
    def _record(self, role, kind, text=None, name=None, args=None,
                result=None, error=None):
        self.transcript.append({
            "ts": datetime.datetime.now().isoformat(timespec="milliseconds"),
            "role": role,
            "kind": kind,
            "text": text,
            "name": name,
            "args": args,
            "result": result,
            "error": error,
        })

    def _record_text(self, text):
        """Append to the last assistant text entry to avoid per-token spam."""
        if (self.transcript and self.transcript[-1]["role"] == "assistant"
                and self.transcript[-1]["kind"] == "text"):
            self.transcript[-1]["text"] += text
        else:
            self._record("assistant", "text", text=text)

    def _record_thought(self, text):
        """Append to the last thought entry so the model's thinking STREAMS as
        one growing block instead of one entry per chunk."""
        if (self.transcript and self.transcript[-1]["role"] == "assistant"
                and self.transcript[-1]["kind"] == "thought"):
            self.transcript[-1]["text"] += text
        else:
            self._record("assistant", "thought", text=text)

    def _record_chunk(self, chunk):
        if isinstance(chunk, str):
            self._record_text(chunk)
        elif isinstance(chunk, types.Text):
            self._record_text(chunk.text)
        elif isinstance(chunk, types.Thought):
            self._record_thought(chunk.text)
        elif isinstance(chunk, types.ToolCall):
            self._record("assistant", "tool_call", name=chunk.name,
                         args=chunk.args)
        elif isinstance(chunk, types.ToolResult):
            self._record("tool", "tool_result", name=chunk.name,
                         result=chunk.result, error=chunk.error)

    def interrupt(self):
        """Cancel this egg's in-flight model turn (used by the stop button).

        Cancels the local streaming task AND asks the backend to halt the active
        turn through the SDK's ``ChatResponse.cancel()`` (which sends a
        ``halt_request`` to the localharness). Closing only the local reader via
        aclose() leaves the server-side model generating, so the stop button
        would not actually stop the remote turn.
        """
        resp = self._active_resp
        if resp is not None:
            try:
                # Fire-and-forget on the running loop: the halt_request is what
                # really stops the backend turn; we don't need to await it.
                asyncio.ensure_future(resp.cancel()).add_done_callback(
                    lambda fut: fut.exception()  # swallow cancellation-side errors
                )
            except RuntimeError:
                pass  # no running loop (e.g. during shutdown) — local cancel stands
        t = self._task
        if t is not None and not t.done():
            t.cancel()

    async def _chat_once(self, text, context=None):
        """One chat attempt: stream the FULL semantic chunk stream.

        Iterate ``resp.chunks`` (NOT ``async for chunk in resp``) — the latter's
        ``__aiter__`` filters to ``Text`` only and would DROP ``Thought``,
        ``ToolCall`` and ``ToolResult``. Using the unfiltered stream lets the
        trace capture 💭 thoughts and 🔧 tool calls/results.
        """
        if context:
            # Prepend the connected eggs' conversation history so this egg can
            # reference what its neighbours have actually done/decided. The SDK's
            # compaction keeps the transcript from ballooning across turns.
            text = (
                "【参考上下文：以下是在本次任务中相关的其他蛋的会话历史/观点，"
                "仅供了解背景】\n"
                f"{context}\n"
                "【以上为参考上下文。以下是用户给你的指令】\n"
                f"{text}"
            )
        resp = await self._agent.chat(text)
        self._active_resp = resp
        try:
            async for chunk in resp.chunks:
                self._record_chunk(chunk)
                t = _chunk_to_text(chunk)
                if t:
                    yield t
        finally:
            self._active_resp = None

    async def _capture_reasoning(self, text):
        """Stream the model's ``reasoning_content`` deltas (best-effort).

        The SDK's local OpenAI harness drops reasoning, so for local backends we
        stream a completion straight to the server to surface the 💭 thinking.
        Sets ``_REASONING['supported']`` so we stop re-calling if the model
        never reasons.
        """
        if not (OPENAI_MODEL and OPENAI_BASE_URL):
            return
        import httpx  # available with the stack; imported lazily to stay safe
        payload = {
            "model": OPENAI_MODEL,
            "stream": True,
            "messages": [
                {"role": "system", "content": self._si},
                {"role": "user", "content": text},
            ],
        }
        headers = {"Content-Type": "application/json"}
        key = os.environ.get("OPENAI_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        url = OPENAI_BASE_URL.rstrip("/") + "/chat/completions"
        got = False
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
                        for ch in obj.get("choices", []):
                            d = ch.get("delta") or {}
                            rc = d.get("reasoning_content") or d.get("reasoning")
                            if rc:
                                got = True
                                yield rc
        except asyncio.CancelledError:
            raise
        except Exception:
            return  # best-effort: never break the actual run
        finally:
            _REASONING["supported"] = got

    async def _heal_conversation(self, err):
        """Reset a corrupt/invalid SDK conversation so the next chat() starts fresh.

        The SDK persists the LLM session under ``save_dir`` keyed by
        ``conversation_id``. A turn that produced neither ``content`` nor
        ``tool_calls`` (e.g. a reasoning-only turn with empty content) gets
        recorded as an empty assistant message; on the NEXT ``chat()`` the API
        rejects the whole history with HTTP 400
        ("Assistant message must contain either 'content' or 'tool_calls'").
        Dropping ``conversation_id`` forces a brand-new conversation that does not
        replay the broken history, so the egg recovers instead of erroring forever
        (which otherwise leaves it stuck "运行中" with an empty box).
        """
        e = str(err)
        if not ("must contain either" in e or "invalid_request" in e
                or "cannot resume" in e or "not found" in e.lower()):
            return False
        try:
            await self.stop()
        except Exception:
            pass
        self.conversation_id = None
        self._start_attempted = False
        self._agent = None
        self._cm = None
        try:
            self._cm, self._agent = await self._open_agent(resume=False)
            if self._agent is not None:
                cid = getattr(self._agent, "conversation_id", None)
                self.conversation_id = cid or uuid.uuid4().hex
            else:
                self.conversation_id = uuid.uuid4().hex
        except Exception:
            self._cm = None
            self._agent = None
        return True

    async def run(self, text, context=None):
        """Stream the model's real response for a command inside this egg.

        The chat runs under the shared box lock (one model request at a time)
        and is retried every ``_LLM_BOX.delay`` seconds on failure.
        """
        self._record("user", "text", text=text)
        if not SDK_OK:
            self._record("system", "status",
                         text="⚠ 该蛋的 LLM SDK 不可用（未安装 google.antigravity）。")
            yield "⚠ 该蛋的 LLM SDK 不可用（未安装 google.antigravity）。"
            return
        # Mark busy + capture the task BEFORE connecting/starting so the stop
        # button can interrupt even during agent startup (first run) or while
        # waiting for the box lock — not only during the active model stream.
        self.busy = True
        self._task = asyncio.current_task()
        try:
            if not await self._ensure_started():
                msg = self._offline_msg("执行命令")
                self._record("system", "status", text=msg)
                yield msg
                return
            async with self._lock:
                # Surface the model's 💭 thinking first (local backends drop it in
                # the SDK harness, so we capture reasoning_content directly).
                if (BACKEND == "openai" and _REASONING["enabled"]
                        and _REASONING["supported"] is not False):
                    try:
                        async with asyncio.timeout(_TURN_TIMEOUT):
                            async for rc in self._capture_reasoning(text):
                                self._record_thought(rc)
                                yield f"💭 {rc}"
                    except asyncio.TimeoutError:
                        yield f"\n⚠ 思考超时（{_TURN_TIMEOUT:g}s），已停止。\n"
                        return
                for attempt in range(1, _LLM_BOX.max_attempts + 1):
                    try:
                        chat = self._chat_once(text, context=context)
                        try:
                            async with asyncio.timeout(_TURN_TIMEOUT):
                                async for t in chat:
                                    yield t
                        finally:
                            # stop the model call + free locks on interrupt
                            try:
                                await chat.aclose()
                            except Exception:  # pragma: no cover
                                pass
                        self._sync_conv_id()   # persist the real conversation id
                        return
                    except asyncio.CancelledError:
                        raise
                    except asyncio.TimeoutError:
                        yield f"\n⚠ 本次运行超时（{_TURN_TIMEOUT:g}s），已停止。\n"
                        return
                    except Exception as ex:
                        # A corrupt/invalid conversation (e.g. an empty assistant
                        # message -> HTTP 400) cannot be retried in place; heal it
                        # (fresh conversation) so the egg recovers instead of
                        # erroring forever and staying stuck "运行中".
                        healed = await self._heal_conversation(ex)
                        if healed and self._agent is None:
                            yield f"\n⚠ 无法重建会话，已停止：{ex}\n"
                            return
                        if attempt >= _LLM_BOX.max_attempts:
                            yield f"\n⚠ 请求多次失败，已停止：{ex}\n"
                            return
                        yield (f"\n⚠ 请求失败（{ex}），{_LLM_BOX.delay:g}s 后重试 "
                               f"（{attempt}/{_LLM_BOX.max_attempts}）…\n")
                        await asyncio.sleep(_LLM_BOX.delay)
        finally:
            self.busy = False
            self._task = None

    async def compress(self):
        """Return a concise summary of the egg's conversation (real model)."""
        self._record("user", "text",
                     text="（压缩会话）请将到目前为止的对话压缩为规格说明。")
        if not SDK_OK:
            self._record("system", "status", text="⚠ 该蛋的 LLM SDK 不可用。")
            return "⚠ 该蛋的 LLM SDK 不可用。"
        if not await self._ensure_started():
            msg = self._offline_msg("压缩会话")
            self._record("system", "status", text=msg)
            return msg
        async with self._lock:
            self.busy = True
            self._task = asyncio.current_task()
            try:
                summary = None
                for attempt in range(1, _LLM_BOX.max_attempts + 1):
                    try:
                        resp = await self._agent.chat(
                            "将我们到目前为止的对话压缩为一段简洁的规格说明"
                            "（markdown），只保留关键决策与产出。"
                        )
                        self._active_resp = resp
                        try:
                            summary = await resp.text()
                        finally:
                            self._active_resp = None
                        break
                    except asyncio.CancelledError:
                        raise
                    except Exception as ex:
                        if await self._heal_conversation(ex) and self._agent is None:
                            summary = "⚠ 压缩失败：无法重建会话。"
                            break
                        if attempt >= _LLM_BOX.max_attempts:
                            summary = "⚠ 压缩失败：模型多次无响应。"
                            break
                        await asyncio.sleep(_LLM_BOX.delay)
                summary = summary or ""
                self._record("assistant", "text", text=summary)
                return summary
            finally:
                self.busy = False
                self._task = None

    # ---- conversation history (used when combining eggs) ----
    def history_text(self, mode="brief", limit=None):
        """Render this egg's chronological transcript into readable text.

        ``mode``:
          - "brief" (default): only 用户+助手 text turns (commands & replies),
            skipping thoughts / tool calls / tool results.
          - "full": everything (thoughts 💭, tool calls, tool results, plans…).
        ``limit`` caps the output to its last N characters (keeps recent turns).
        """
        lines = []
        for r in self.transcript:
            kind = r.get("kind")
            role = r.get("role")
            # skip pure infrastructure entries
            if kind in ("system_prompt", "tools", "status"):
                continue
            # brief mode keeps only user + assistant text turns
            if mode != "full" and kind != "text":
                continue
            text = r.get("text")
            name = r.get("name")
            if kind == "prompt":
                lines.append(f"[上下文] {text}")
            elif kind == "plan":
                lines.append(f"[计划] {text}")
            elif kind == "thought":
                lines.append(f"[思考] {text}")
            elif kind == "tool_call":
                try:
                    args = json.dumps(r.get("args"), ensure_ascii=False)
                except Exception:
                    args = str(r.get("args"))
                lines.append(f"[调用工具 {name}] {args}")
            elif kind == "tool_result":
                if r.get("error") is not None:
                    lines.append(f"[工具结果 {name}] 错误: {r.get('error')}")
                else:
                    res = r.get("result")
                    res = res if isinstance(res, str) else str(res)
                    lines.append(f"[工具结果 {name}] {res}")
            elif role == "user":
                lines.append(f"用户: {text}")
            elif role == "assistant":
                lines.append(f"助手: {text}")
        out = "\n".join(lines)
        if limit and len(out) > limit:
            out = "…（更早的历史略去）\n" + out[-limit:]
        return out

    # ---- worktree inspection (used by the "review code" panel) ----
    def list_files(self):
        out = []
        root = self.worktree.resolve()
        for p in self.worktree.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(self.worktree)
            if rel.parts and rel.parts[0] in ("agent_data",):
                continue
            out.append(str(rel))
        return sorted(out)

    def read_file(self, path):
        fp = (self.worktree / path).resolve()
        root = self.worktree.resolve()
        if fp != root and root not in fp.parents:
            raise ValueError("path escapes worktree")
        if not fp.is_file():
            return None
        return fp.read_text(errors="replace")
