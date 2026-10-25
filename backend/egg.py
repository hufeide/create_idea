"""One idea egg = one Antigravity SDK Agent scoped to its own worktree.

The egg holds a long-lived Agent session (history accumulates across turns)
and writes code into ``worktree`` via the SDK's built-in file/command tools.
When a model is connected (cloud key or local OpenAI/LiteRT backend) the egg
streams the model's real responses; if no model is reachable it reports that
explicitly instead of faking a completion.
"""
import asyncio
import datetime
import os
import time
import uuid
from pathlib import Path

from .config import (WORKTREE_ROOT, build_agent_config, BACKEND, quiet_sdk_start,
                     OPENAI_MODEL, OPENAI_BASE_URL, is_corrupt_conversation_error)
from .reasoning import (set_reasoning_capture, reasoning_capture_state,
                        reasoning_enabled, capture_reasoning)

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


async def _aclose_quiet(cm):
    """Close an SDK agent context manager in the background, swallowing errors."""
    try:
        await cm.__aexit__(None, None, None)
    except Exception:
        pass


# Per-turn safety timeout. If a model turn (e.g. a hung tool call / a stream that
# never ends) doesn't finish, abandon it so the box lock is released and the egg
# returns to 空闲 instead of being stuck 运行中 forever. Generous by default;
# override with EGG_TURN_TIMEOUT (seconds).
_TURN_TIMEOUT = float(os.environ.get("EGG_TURN_TIMEOUT", "300"))

# A turn that goes SILENT (no chunk at all for this long) is stalled — the
# model/server stopped streaming. Without this watchdog the egg would keep the
# "运行中" badge for the whole _TURN_TIMEOUT and the panel/trace would look
# frozen. Abort the turn instead so the egg returns to 空闲 promptly.
_STALL_TIMEOUT = float(os.environ.get("EGG_STALL_TIMEOUT", "120"))

# The reasoning probe is a throwaway extra call; it must never hold the run
# hostage for long.
_REASON_TIMEOUT = float(os.environ.get("EGG_REASON_TIMEOUT", "90"))


# The SDK's local OpenAI harness DISCARDS the model's `reasoning_content`, so
# for local backends we capture the thinking directly via a streaming call and
# record it as 💭 thoughts. The shared helpers live in .reasoning so the
# 总指挥 (Commander) can surface its thinking the same way.
__all__ = ["set_reasoning_capture", "reasoning_capture_state"]


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
        # Per-turn tracking so the trace can show REAL tool results when the
        # backend provides them, and add a single honest note only when it cannot
        # (the local openai backend executes tools server-side and returns no
        # ToolResult chunk / does not fire the post_tool_call hook).
        self._turn_tool_calls = 0
        self._got_tool_result = False
        # Run state for the UI + interrupt: busy while a model turn streams, and
        # _task points at the task iterating run() so it can be cancelled.
        self.busy = False
        self._task = None
        # The active SDK ChatResponse while a model turn streams, so interrupt()
        # can send halt_request to the backend (not merely close the local reader).
        self._active_resp = None
        # Whether the CURRENT turn produced a usable assistant payload (text or
        # a structured tool call). A turn with NEITHER is persisted by the SDK as
        # an empty assistant message, which makes the model server reject every
        # later request in this conversation with HTTP 400
        # ("Assistant message must contain either 'content' or 'tool_calls'!").
        # We detect that and drop the conversation before it can poison a turn.
        self._turn_payload = False
        # Per-run flag: the fully-assembled model input (with injected neighbour
        # context) is recorded into the trace only once per run, not per retry.
        self._prompt_recorded = False
        # Chronological conversation trace (user commands, model text/thoughts,
        # system prompt, tool calls/results) — viewable via /api/trace/{egg_id}.
        self.transcript = []
        # ---- run bookkeeping (read by the commander's monitor loop) ----
        # When the CURRENT run started (monotonic) / None when idle.
        self.run_started_at = None
        # When the transcript last grew — used to detect a SILENT (stuck) run.
        self.last_activity = time.monotonic()
        # What the last finished run produced (reply length, duration, …).
        self.last_completion = None
        # How many runs this egg has completed (monotonic counter).
        self.run_seq = 0
        self._run_mark = 0      # transcript index where the current run started
        self._run_started = None

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
            self._got_tool_result = True
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
            f"You have a workspace directory: {self.worktree} "
            f"(file/command tools are available if you need them).\n"
            f"BEFORE you start, decide what kind of task this is:\n"
            f"(A) The task genuinely REQUIRES CODE (implementing a program, "
            f"algorithm, script, service or tool). Then implement it for real: "
            f"write actual source files into the workspace and RUN them with the "
            f"command tools to verify they work. Never fake it — only report what "
            f"you actually ran and observed.\n"
            f"(B) The task is a GENERAL (non-programming) task — e.g. writing "
            f"copy/text, analysis, research summary, planning, brainstorming, "
            f"data organization, answering a question. Then do NOT treat it as a "
            f"programming task: complete it directly in your reply, and do not "
            f"invent a program just to look productive. Saving genuinely useful "
            f"artifacts (notes, documents, data) into the workspace is allowed "
            f"but optional.\n"
            f"Either way, the deliverable is your final reply: state clearly what "
            f"you produced. If you wrote code, list the files and how to run them; "
            f"if you did not, present the content/result directly. "
            f"Whenever you must hand over a written artifact, write it as a "
            f"Markdown file in the ROOT of your workspace directory — that root "
            f"is exactly what the user inspects with 「审阅代码」. Never put a "
            f"deliverable inside agent_data/ (an internal folder the user "
            f"cannot see). "
            f"Coordinates with connected eggs only when instructed. Be concise."
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
            # Re-enable the "composed egg inherits its sources as cloud
            # sub-agents" feature: combine() builds SubagentConfig objects and
            # hands them to _spawn -> self.subagents. build_agent_config drops
            # this key for backends that don't accept it, so passing it here is
            # safe everywhere.
            subagents=self.subagents,
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
    @property
    def transcript_len(self):
        """Total character length of the conversation trace.

        The commander's monitor compares this across ticks to answer
        "会话有没有变长" (is the egg still making progress?).
        """
        return sum(len(r.get("text") or "") for r in self.transcript)

    def _record(self, role, kind, text=None, name=None, args=None,
                result=None, error=None):
        self.last_activity = time.monotonic()
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
            self._turn_payload = True
            self._record_text(chunk)
        elif isinstance(chunk, types.Text):
            self._turn_payload = True
            self._record_text(chunk.text)
        elif isinstance(chunk, types.Thought):
            self._record_thought(chunk.text)
        elif isinstance(chunk, types.ToolCall):
            # A structured tool call IS a valid assistant payload (the API
            # accepts an assistant message that carries tool_calls).
            self._turn_payload = True
            self._turn_tool_calls += 1
            self._record("assistant", "tool_call", name=chunk.name,
                         args=chunk.args)
        elif isinstance(chunk, types.ToolResult):
            self._got_tool_result = True
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
                    lambda fut: None if fut.cancelled() else fut.exception()
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
            if not self._prompt_recorded:
                self._prompt_recorded = True
                # 记录实际发送给模型的完整输入（含注入的邻居上下文）——轨迹必须
                # 反映模型真实看到的交互内容，而不仅是用户敲下的那句指令。
                self._record("system", "prompt", text=text)
        self._turn_tool_calls = 0
        self._got_tool_result = False
        resp = await self._agent.chat(text)
        self._active_resp = resp
        try:
            async for chunk in resp.chunks:
                self._record_chunk(chunk)
                t = _chunk_to_text(chunk)
                if t:
                    yield t
            # The local openai backend executes tools server-side and returns no
            # ToolResult chunk (nor fires post_tool_call), so the client never
            # sees the result. Say so ONCE per turn — but only when there really
            # was a tool call AND no real result was recorded (cloud backends DO
            # surface results via the hook, so they are never penalised here).
            if self._turn_tool_calls > 0 and not self._got_tool_result:
                self._record("tool", "tool_result",
                             result="（本后端不回传工具结果：工具在服务端执行，"
                                    "结果直接进入模型上下文；产物见本蛋工作目录）")
        finally:
            self._active_resp = None

    async def _capture_reasoning(self, text):
        """Stream the model's ``reasoning_content`` deltas (best-effort).

        The SDK's local OpenAI harness drops reasoning, so for local backends we
        stream a completion straight to the server to surface the 💭 thinking.
        """
        async for rc in capture_reasoning(self._si, text):
            yield rc

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
        if not is_corrupt_conversation_error(e):
            return False
        # Close the OLD session in the BACKGROUND: `Agent.__aexit__` tears down
        # the harness conversation and can BLOCK for a while. Doing it inline
        # would hold the egg's lock (and the WS event stream) hostage — the egg
        # would sit in "运行中" with no output. Detach it and let it finish on
        # its own task.
        old_cm = self._cm
        self._cm = None
        self._agent = None
        self.conversation_id = None
        self._start_attempted = False
        if old_cm is not None:
            asyncio.ensure_future(_aclose_quiet(old_cm))
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
        self._prompt_recorded = False
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
        # Remember where this run's transcript starts so the completion facts
        # (reply length) can be computed when it finishes.
        self._run_mark = len(self.transcript)
        self._run_started = time.monotonic()
        self.run_started_at = self._run_started
        try:
            if not await self._ensure_started():
                msg = self._offline_msg("执行命令")
                self._record("system", "status", text=msg)
                yield msg
                return
            async with self._lock:
                # Surface the model's 💭 thinking first (local backends drop it in
                # the SDK harness, so we capture reasoning_content directly).
                if (BACKEND == "openai" and reasoning_enabled()):
                    try:
                        async with asyncio.timeout(_REASON_TIMEOUT):
                            async for rc in self._capture_reasoning(text):
                                self._record_thought(rc)
                                yield f"💭 {rc}"
                    except asyncio.TimeoutError:
                        yield f"\n⚠ 思考超时（{_REASON_TIMEOUT:g}s），已跳过。\n"
                for attempt in range(1, _LLM_BOX.max_attempts + 1):
                    try:
                        self._turn_payload = False
                        chat = self._chat_once(text, context=context)
                        got_text = False
                        # Whole-turn safety: the stall watchdog below only catches a
                        # SILENT stream (no chunk for _STALL_TIMEOUT). A turn that keeps
                        # trickling chunks past _TURN_TIMEOUT must still be abandoned so
                        # the egg never sits 运行中 forever — this arms the advertised
                        # _TURN_TIMEOUT (which was previously dead code).
                        async with asyncio.timeout(_TURN_TIMEOUT):
                            try:
                                # Stream with a STALL watchdog: if no chunk arrives
                                # for _STALL_TIMEOUT the model/server has gone quiet
                                # (often a truncated/aborted upstream turn). Abort so
                                # the egg leaves "运行中" and the panel/trace stop
                                # looking frozen.
                                agen = chat.__aiter__()
                                while True:
                                    try:
                                        t = await asyncio.wait_for(
                                            agen.__anext__(),
                                            timeout=_STALL_TIMEOUT)
                                    except StopAsyncIteration:
                                        break
                                    except asyncio.TimeoutError:
                                        yield (f"\n⚠ 模型流静默超过 {_STALL_TIMEOUT:g}s，"
                                               f"已中断本轮。\n")
                                        resp = self._active_resp
                                        if resp is not None:
                                            try:
                                                await resp.cancel()
                                            except Exception:
                                                pass
                                        break
                                    stripped = t.lstrip()
                                    if not (t.startswith("💭")
                                            or stripped.startswith("🔧")
                                            or stripped.startswith("↳")):
                                        got_text = True
                                    yield t
                            finally:
                                # stop the model call + free locks on interrupt
                                try:
                                    await chat.aclose()
                                except Exception:  # pragma: no cover
                                    pass
                        if not got_text and not self._turn_payload:
                            # The turn produced NOTHING (no text, no tool call) —
                            # the response was truncated or rejected upstream.
                            # Say so instead of silently "succeeding" with an
                            # empty answer.
                            yield ("\n⚠ 本轮没有收到任何模型输出（可能被截断或"
                                   "上游中断），可重试。\n")
                        # An EMPTY turn (no text, no structured tool call) is
                        # persisted by the SDK as an assistant message with
                        # neither content nor tool_calls — which makes the model
                        # server reject every later request in this conversation
                        # (HTTP 400 "Assistant message must contain either
                        # 'content' or 'tool_calls'!"). Drop the conversation NOW
                        # so the NEXT command starts from a clean one instead of
                        # failing forever.
                        if not self._turn_payload:
                            print(f"[egg {self.id}] empty model turn -> "
                                  f"resetting conversation to avoid HTTP 400")
                            await self._heal_conversation(
                                "Assistant message must contain either "
                                "'content' or 'tool_calls'!")
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
            self._finish_run()

    def _finish_run(self):
        """Close out a run and publish the facts the commander's monitor reads.

        ``last_completion`` lets the commander tell "this egg finished" from
        "this egg is still working" without waiting on the run's own task.
        """
        self.busy = False
        self._task = None
        now = time.monotonic()
        started = self._run_started or now
        self.run_started_at = None
        reply_len = 0
        for r in self.transcript[self._run_mark:]:
            if r.get("kind") == "text" and r.get("role") == "assistant":
                reply_len += len(r.get("text") or "")
        self.last_completion = {
            "ts": time.time(),
            "seq": self.run_seq,
            "reply_len": reply_len,
            "transcript_len": self.transcript_len,
            "duration": round(now - started, 1),
        }
        self.run_seq += 1

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
        # Mark busy + capture the task BEFORE waiting for the lock — same as
        # run(): a compress queued behind an in-flight run must show 运行中 and
        # stay cancellable via the stop button, not look 空闲.
        self.busy = True
        self._task = asyncio.current_task()
        try:
            async with self._lock:
                summary = None
                prompt = ("将我们到目前为止的对话压缩为一段简洁的规格说明"
                          "（markdown），只保留关键决策与产出。")
                # 压缩实际发送的提示词与 user 记录措辞不同，如实记入轨迹。
                self._record("system", "prompt", text=prompt)
                for attempt in range(1, _LLM_BOX.max_attempts + 1):
                    try:
                        resp = await self._agent.chat(prompt)
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
        """Render this egg's conversation for PASSING ON to another agent.

        ONLY the assistant's own answers are transmitted. The 💭 thinking, the
        🔧 tool calls / ↳ results and the status entries stay in this egg's
        local 轨迹 — they are process noise that must never enter another
        egg's or the commander's context (it confuses the receiver and burns
        its context window).

        ``mode`` is kept for call-site compatibility; it no longer widens the
        payload (there is no "thinking + tools" transfer any more).
        ``limit`` caps the output to its last N characters (keeps recent turns).
        """
        lines = []
        for r in self.transcript:
            if r.get("kind") != "text":
                continue
            if r.get("role") != "assistant":
                continue
            text = (r.get("text") or "").strip()
            if not text:
                continue
            lines.append(f"助手: {text}")
        out = "\n".join(lines)
        if limit and len(out) > limit:
            out = "…（更早的历史略去）\n" + out[-limit:]
        return out

    # ---- worktree inspection (used by the "review code" panel) ----
    def list_files(self):
        """Files the user can inspect in the 「审阅代码」 panel.

        The worktree root is the review panel's root, so a deliverable written
        here is directly visible/openable by the user. ``agent_data/`` holds the
        SDK's internal app data (conversation db, logs) and is hidden — except
        for markdown files, which may be deliverables and must never stay
        invisible to the user.
        """
        out = []
        for p in self.worktree.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(self.worktree)
            if rel.parts and rel.parts[0] in ("agent_data",):
                if p.suffix.lower() != ".md":
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
