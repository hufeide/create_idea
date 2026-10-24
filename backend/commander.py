"""The 总指挥 (Commander): a coordinator Agent.

It does not implement modules itself; it receives the current canvas graph as
context and proposes coordination / interfacing changes between eggs. It streams
the model's real response; if no model is reachable it reports that explicitly.
"""
import asyncio
import datetime
import os
from pathlib import Path

from .config import (build_agent_config, quiet_sdk_start, BACKEND,
                     is_corrupt_conversation_error)
from .reasoning import reasoning_enabled, capture_reasoning

# Per-turn timeout for the extra reasoning-capture call (best-effort).
_REASON_TIMEOUT = float(os.environ.get("COMMANDER_REASON_TIMEOUT", "90"))

# A turn that goes SILENT for this long is stalled — abort it so the commander
# doesn't sit in "运行中" with a frozen panel.
_STALL_TIMEOUT = float(os.environ.get("COMMANDER_STALL_TIMEOUT", "120"))


async def _aclose_quiet(cm):
    """Close an SDK agent context manager in the background, swallowing errors."""
    try:
        await cm.__aexit__(None, None, None)
    except Exception:
        pass

try:  # pragma: no cover
    from google.antigravity import Agent, types
    SDK_OK = True
except Exception:  # pragma: no cover
    SDK_OK = False


def _chunk_to_text(chunk):
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


class Commander:
    def __init__(self, session_dir=None, conversation_id=None):
        self._cm = None
        self._agent = None
        self._lock = asyncio.Lock()
        self._start_attempted = False
        self._start_error = None
        self._recorded_init = False
        self._si = ""
        # Session binding: when set, the commander's chat is persisted under the
        # session's save_dir and resumes via conversation_id across restarts.
        self.session_dir = session_dir
        self._commander_dir = None
        # None => first connect lets the SDK create a fresh conversation and assign
        # an id (read back via self._agent.conversation_id); we persist that id so
        # later turns / restarts resume the SAME session. Passing an id absent from
        # save_dir makes the SDK raise "cannot resume".
        self.conversation_id = conversation_id
        # Chronological coordinator trace — viewable via GET /api/trace/commander.
        self.transcript = []

    def bind_session(self, session_dir, conversation_id=None):
        """Attach this commander to a canvas session so its chat persists/resumes."""
        self.session_dir = session_dir
        # Always adopt the given id: an explicit id resumes a saved session, while
        # None (e.g. a brand-new canvas) starts a FRESH commander conversation
        # instead of leaking the previous canvas's id into the new one.
        self.conversation_id = conversation_id
        # Commander keeps its OWN sub-directory inside the session (separate from
        # the eggs' shared agent_data), just like each egg has its own worktree.
        self._commander_dir = (Path(session_dir) / "commander_data"
                               if session_dir else None)
        # Force a fresh connection on next use with the new session params.
        # Reset ALL start-related state so a brand-new conversation truly starts
        # from scratch (no stale error / "already recorded" flags leaking in).
        self._start_attempted = False
        self._start_error = None
        self._recorded_init = False
        self._agent = None
        self._cm = None

    async def start(self):
        if not SDK_OK:
            return
        si = (
            "你是总指挥（Commander），负责协调画布上的各个 idea 蛋（LLM 模块）。"
            "你通过接口 LLM 与各蛋沟通，提出修改与对齐建议。"
            "只有用户建立的连线才允许蛋之间互通消息。"
            "你可以用只读工具（view_file / list_directory / search_directory / "
            "find_file）查看各蛋工作区里真实产出的文件，以便基于事实协调。"
            "请基于给定的画布状态给出协调方案，简明扼要。"
        )
        self._si = si

        async def _connect():
            # The commander coordinates eggs; it may INSPECT what they produced
            # (read-only tools) so its 轨迹 shows the real 🔧 tool calls. We keep
            # the set small (read-only) and use MINIMAL behavior to shrink ctx.
            caps = types.CapabilitiesConfig(
                enabled_tools=[
                    types.BuiltinTools.VIEW_FILE,
                    types.BuiltinTools.LIST_DIR,
                    types.BuiltinTools.SEARCH_DIR,
                    types.BuiltinTools.FIND_FILE,
                ],
                enable_subagents=False,
                agent_behavior=types.AgentBehavior.MINIMAL,
            )
            self._cm = Agent(build_agent_config(
                system_instructions=self._si, capabilities=caps,
                save_dir=(str(self._commander_dir) if self._commander_dir else None),
                conversation_id=(self.conversation_id
                                 if self._commander_dir and self.conversation_id
                                 else None),
            ))
            self._agent = await self._cm.__aenter__()
            # Capture the id the SDK assigned on a fresh connection so the session
            # can be resumed after restarts.
            if self.conversation_id is None and self._agent is not None:
                cid = getattr(self._agent, "conversation_id", None)
                if cid:
                    self.conversation_id = cid

        try:
            await _connect()
        except Exception as ex:
            err = str(ex)
            # Recoverable: a persisted conversation_id is missing in save_dir.
            # Retry once with a fresh conversation so the user isn't stuck.
            if (self.conversation_id is not None and "conversation" in err
                    and ("not found" in err or "cannot resume" in err
                         or "resume" in err)):
                self.conversation_id = None
                self._cm = None
                try:
                    await _connect()
                except Exception as ex2:
                    print(f"[commander] start failed: {ex2}")
                    self._agent = None
                    self._start_error = ex2
            else:
                print(f"[commander] start failed: {ex}")
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

    async def _ensure_started(self):
        if self._agent is not None:
            return True
        if self._start_attempted:
            return False
        self._start_attempted = True
        await self.start()
        if self._agent is not None and not self._recorded_init:
            self._recorded_init = True
            self._record("system", "system_prompt", text=self._si)
            self._record("system", "tools",
                         text="enabled: view_file / list_directory / "
                              "search_directory / find_file（只读）")
        return self._agent is not None

    async def _heal_conversation(self, err):
        """Reset a corrupt/invalid conversation so the next chat() starts fresh.

        The commander persists its session under ``commander_data`` keyed by
        ``conversation_id``; a turn that produced neither ``content`` nor
        ``tool_calls`` is stored as an empty assistant message, and the model
        server then rejects EVERY later request in that conversation with
        HTTP 400 "Assistant message must contain either 'content' or
        'tool_calls'!". Dropping ``conversation_id`` forces a brand-new
        conversation that does not replay the broken history.
        """
        if not is_corrupt_conversation_error(err):
            return False
        # Close the OLD session in the BACKGROUND: `Agent.__aexit__` can block
        # while it tears down the harness conversation, and doing it inline
        # would stall the whole command round (the UI would look frozen).
        old_cm = self._cm
        self._cm = None
        self._agent = None
        self.conversation_id = None
        self._start_attempted = False
        self._start_error = None
        if old_cm is not None:
            asyncio.ensure_future(_aclose_quiet(old_cm))
        try:
            await self.start()
            if self._agent is not None:
                cid = getattr(self._agent, "conversation_id", None)
                self.conversation_id = cid or self.conversation_id
        except Exception as ex:  # pragma: no cover
            print(f"[commander] heal failed: {ex}")
        return True

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
        """Merge consecutive thoughts so the thinking streams as one block."""
        if (self.transcript and self.transcript[-1]["role"] == "assistant"
                and self.transcript[-1]["kind"] == "thought"):
            self.transcript[-1]["text"] += text
        else:
            self._record("assistant", "thought", text=text)

    def history_text(self, limit=None):
        """Chronological user+assistant dialogue ONLY.

        Tool calls, tool results, thinking and status entries are deliberately
        excluded — the commander/egg channels carry plain conversation context.
        """
        lines = []
        for r in self.transcript:
            if r.get("role") not in ("user", "assistant"):
                continue
            if r.get("kind") != "text":
                continue
            text = (r.get("text") or "").strip()
            if not text:
                continue
            who = "用户" if r.get("role") == "user" else "总指挥"
            lines.append(f"{who}：{text}")
        out = "\n".join(lines)
        if limit and len(out) > limit:
            out = "…（更早的历史略去）\n" + out[-limit:]
        return out

    async def run(self, text, graph_summary):
        prompt = f"{graph_summary}\n\n用户协调指令：{text}\n请给出协调/修改建议。"
        self._record("user", "text", text=text)
        self._record("system", "prompt", text=prompt)
        if not SDK_OK:
            self._record("system", "status",
                         text="⚠ 总指挥的 LLM SDK 不可用（未安装 google.antigravity）。")
            yield "⚠ 总指挥的 LLM SDK 不可用（未安装 google.antigravity）。"
            return
        if not await self._ensure_started():
            msg = "⚠ 总指挥的 LLM 未连接，无法协调。"
            if self._start_error:
                msg += f"\n原因：{self._start_error}"
            msg += "\n请确认 IDEA_BACKEND 与本地/云端模型可用（openai 后端需 :9000 服务在跑）。"
            self._record("system", "status", text=msg)
            yield msg
            return
        async with self._lock:
            # A corrupt conversation (e.g. an empty assistant message -> HTTP 400
            # "Assistant message must contain either 'content' or 'tool_calls'")
            # can never be fixed in place: heal it (fresh conversation) and retry
            # once, so the commander recovers instead of erroring out forever.
            for attempt in (1, 2):
                try:
                    # Surface the 💭 thinking first (the local OpenAI harness
                    # drops reasoning, so we capture it directly) — same as the
                    # eggs do, so the 总指挥's 轨迹 shows its reasoning too.
                    if BACKEND == "openai" and reasoning_enabled():
                        try:
                            async with asyncio.timeout(_REASON_TIMEOUT):
                                async for rc in capture_reasoning(self._si, text):
                                    self._record_thought(rc)
                                    yield f"💭 {rc}"
                        except asyncio.TimeoutError:
                            yield f"\n⚠ 思考超时（{_REASON_TIMEOUT:g}s），已停止。\n"
                            return
                    resp = await self._agent.chat(prompt)
                    payload = False
                    # Iterate resp.chunks (NOT `async for chunk in resp`): the
                    # latter filters to Text only and would DROP Thought,
                    # ToolCall and ToolResult — so the trace would never show
                    # the commander's thinking or its tool calls.
                    # A STALL watchdog aborts a silent turn so the commander
                    # never sits in "运行中" with a frozen panel.
                    achunks = resp.chunks.__aiter__()
                    while True:
                        try:
                            chunk = await asyncio.wait_for(
                                achunks.__anext__(), timeout=_STALL_TIMEOUT)
                        except StopAsyncIteration:
                            break
                        except asyncio.TimeoutError:
                            yield (f"\n⚠ 模型流静默超过 {_STALL_TIMEOUT:g}s，"
                                   f"已中断本轮。\n")
                            try:
                                await resp.cancel()
                            except Exception:
                                pass
                            break
                        if isinstance(chunk, types.Thought):
                            self._record_thought(chunk.text)
                        elif isinstance(chunk, types.Text):
                            payload = True
                            self._record_text(chunk.text)
                        elif isinstance(chunk, types.ToolCall):
                            payload = True
                            self._record("assistant", "tool_call",
                                         name=getattr(chunk, "name", None),
                                         args=getattr(chunk, "args", None))
                        elif isinstance(chunk, types.ToolResult):
                            self._record("tool", "tool_result",
                                         name=getattr(chunk, "name", None),
                                         result=getattr(chunk, "result", None),
                                         error=getattr(chunk, "error", None))
                        elif isinstance(chunk, str):
                            payload = True
                            self._record_text(chunk)
                        t = _chunk_to_text(chunk)
                        if t:
                            yield t
                    if not payload:
                        # Empty turn -> it will be persisted as an empty assistant
                        # message and poison the next request. Drop the
                        # conversation now.
                        print("[commander] empty model turn -> resetting conversation")
                        await self._heal_conversation(
                            "Assistant message must contain either "
                            "'content' or 'tool_calls'!")
                    return
                except asyncio.CancelledError:
                    raise
                except Exception as ex:
                    if attempt == 1 and await self._heal_conversation(ex):
                        continue
                    self._record("system", "status",
                                 text=f"⚠ 总指挥请求失败：{ex}")
                    yield f"⚠ 总指挥请求失败：{ex}"
                    return
