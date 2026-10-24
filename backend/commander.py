"""The 总指挥 (Commander): a coordinator Agent.

It does not implement modules itself; it receives the current canvas graph as
context and proposes coordination / interfacing changes between eggs. It streams
the model's real response; if no model is reachable it reports that explicitly.
"""
import asyncio
import datetime

from .config import build_agent_config, quiet_sdk_start

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
    def __init__(self):
        self._cm = None
        self._agent = None
        self._lock = asyncio.Lock()
        self._start_attempted = False
        self._start_error = None
        self._recorded_init = False
        self._si = ""
        # Chronological coordinator trace — viewable via GET /api/trace/commander.
        self.transcript = []

    async def start(self):
        if not SDK_OK:
            return
        si = (
            "你是总指挥（Commander），负责协调画布上的各个 idea 蛋（LLM 模块）。"
            "你通过接口 LLM 与各蛋沟通，提出修改与对齐建议。"
            "只有用户建立的连线才允许蛋之间互通消息。"
            "请基于给定的画布状态给出协调方案，简明扼要。"
        )
        self._si = si
        try:
            with quiet_sdk_start():
                # Commander is a pure text coordinator: no tools needed, so strip
                # the default tool schemas and use MINIMAL behavior to shrink ctx.
                caps = types.CapabilitiesConfig(
                    enabled_tools=[],
                    enable_subagents=False,
                    agent_behavior=types.AgentBehavior.MINIMAL,
                )
                self._cm = Agent(build_agent_config(
                    system_instructions=si, capabilities=caps))
                self._agent = await self._cm.__aenter__()
        except Exception as ex:  # pragma: no cover
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
            self._record("system", "tools", text="enabled: （无，纯文本协调）")
        return self._agent is not None

    def _record(self, role, kind, text=None):
        self.transcript.append({
            "ts": datetime.datetime.now().isoformat(timespec="milliseconds"),
            "role": role,
            "kind": kind,
            "text": text,
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
            resp = await self._agent.chat(prompt)
            async for chunk in resp:
                if isinstance(chunk, types.Thought):
                    self._record_thought(chunk.text)
                elif isinstance(chunk, types.Text):
                    self._record_text(chunk.text)
                elif isinstance(chunk, str):
                    self._record_text(chunk)
                t = _chunk_to_text(chunk)
                if t:
                    yield t
