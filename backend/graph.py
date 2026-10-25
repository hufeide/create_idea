"""Canvas state: eggs, edges, and the operations that mutate them.

This is the only module that knows about the Antigravity SDK's higher-level
features:
  * ``generate`` uses structured output (response_schema) to plan eggs.
  * ``combine`` wires the source eggs in as SubagentConfigs of the new egg,
    so the combined egg can delegate to its parts.
"""
import asyncio
import datetime
import json
import os
import re
import time
from pathlib import Path
from typing import Literal

import pydantic

from .egg import EggAgent, reasoning_capture_state
from .commander import Commander
from .config import (build_agent_config, BACKEND, SESSIONS_DIR,
                     is_corrupt_conversation_error)

try:  # pragma: no cover
    from google.antigravity import Agent, types, CapabilitiesConfig
    SDK_OK = True
except Exception:  # pragma: no cover
    SDK_OK = False


_PUMP_DONE = object()   # sentinel: an egg's run generator finished / was cancelled

# ---- commander monitoring / loop tuning -------------------------------------
# How often the commander polls the eggs while they work (seconds).
_MONITOR_INTERVAL = float(os.environ.get("COMMANDER_MONITOR_INTERVAL", "3"))
# A running egg that produced NOTHING for this long is considered stuck; the
# commander is told so it can re-dispatch / switch eggs / wrap up.
_EGG_SILENT = float(os.environ.get("COMMANDER_EGG_SILENT", "180"))
# A running egg that has been going for longer than this is considered overdue.
_EGG_OVERDUE = float(os.environ.get("COMMANDER_EGG_OVERDUE", "420"))
# Safety cap on the commander's loop (it normally ends with action="finish").
_MAX_ROUNDS = int(os.environ.get("COMMANDER_MAX_ROUNDS", "24"))
# How many consecutive rounds may end without a runnable task before the
# commander is forced to wrap up (guards against an endless no-op loop).
_MAX_IDLE = int(os.environ.get("COMMANDER_MAX_IDLE", "3"))

# Task order is controlled ONLY by explicit depends_on fields.
# Do not infer dependencies from verbs, titles, role names, or visual canvas edges.

# ---- what counts as a REAL conclusion ----------------------------------------
# A run's streamed text mixes three very different things: the model's 💭
# thinking, its 🔧 tool calls / ↳ results, and its actual answer. Only the last
# one is a conclusion — the others are process noise that must never be reported
# as "this egg finished its work".
_ACTIVITY_PREFIXES = ("💭", "🔧", "↳", "⚠", "（执行出错", "（任务流被取消")
# Markers that prove the turn was CUT OFF (stall watchdog, upstream error,
# timeout). A cut-off turn is not a conclusion, even if some text came through.
_ABORT_MARKERS = (
    "已中断本轮",
    "本轮没有收到任何模型输出",
    "请求失败",
    "请求多次失败",
    "已停止",
    "思考超时",
    "静默超过",
)


async def _pump(agen, q):
    """Drain an async generator into a queue on its OWN task.

    Running egg.run() in a dedicated task means egg.interrupt() cancels only
    that task (egg._task), leaving the caller's task — and thus the whole
    plan/connection — alive, so the plan can continue to the next egg. The
    generator is aclose()d in finally so the egg's locks are released at once.
    """
    try:
        async for item in agen:
            await q.put(item)
    except asyncio.CancelledError:
        pass
    except Exception as ex:  # pragma: no cover - surface model/tool errors
        await q.put(("__err__", str(ex)))
    finally:
        try:
            await agen.aclose()
        except Exception:  # pragma: no cover
            pass
        await q.put(_PUMP_DONE)


class IdeaSpec(pydantic.BaseModel):
    title: str
    prompt: str
    type: str = "module"


class IdeaPlan(pydantic.BaseModel):
    eggs: list[IdeaSpec]


class CommanderTask(pydantic.BaseModel):
    """One dispatch decision for the NEXT round only (reactive adjudication)."""
    egg_id: str
    message: str
    # Box ids (the id of the egg that produced the box) that the commander
    # AUTHORIZES this task to read. Only the commander may route a box from one
    # egg to another — an undelivered box stays with the commander, who
    # arbitrates it.
    deliver: list[str] = pydantic.Field(default_factory=list)
    # Who this egg must NOTIFY once it finishes. The commander is ALWAYS
    # notified (that is the default channel); ids listed here are ADDITIONAL
    # eggs the commander authorizes to receive this egg's conclusion.
    notify: list[str] = pydantic.Field(default_factory=list)
    # Eggs whose CONCLUSION this task depends on. A task that depends on
    # another task OF THE SAME ROUND cannot start yet — the commander defers
    # it to the next round (e.g. a judge that must weigh both debaters).
    depends_on: list[str] = pydantic.Field(default_factory=list)


class CommanderDecision(pydantic.BaseModel):
    """The commander's decision for ONE round.

    ``action``:
      - "dispatch" -> run ``tasks`` now (1 task = serial, N tasks = parallel)
      - "finish"   -> the instruction is complete; ``summary`` is the recap
    """
    action: Literal["dispatch", "finish"] = "dispatch"
    tasks: list[CommanderTask] = pydantic.Field(default_factory=list)
    summary: str = ""
    note: str = ""


class Canvas:
    def __init__(self):
        self.eggs = {}
        self.edges = []
        self._id = 0
        self.commander = Commander()
        self.history_mode = "brief"
        self.session_id = None
        self.session_dir = None
        self.created_at = None
        self.label = None
        self._last_save = None
        # Notifications and task-scoped results survive commander_run calls and restarts.
        self._pending_notifications = {}
        # task_id -> {egg_id, phase, message, status, result, updated_at}
        self._task_results = {}
        # egg_id -> the last delivery check of that egg (delivered / not, path).
        # The commander reads it so "已交付" is never a guess.
        self._last_delivery = {}
        # run_id -> persisted phase-machine state, so interrupted workflows are diagnosable.
        self._workflow_runs = {}

    # ---- id helpers ----
    def _new_id(self, p="e"):
        """Return a unique id even when loading legacy sessions with a stale counter."""
        used = set(self.eggs)
        used.update(ed.get("id") for ed in self.edges if ed.get("id"))
        while True:
            self._id += 1
            candidate = f"{p}{self._id}"
            if candidate not in used:
                return candidate

    # ---- lifecycle ----
    async def start(self, session_id=None):
        # Boot to an EMPTY canvas (no auto-seed per product decision). If a session
        # id is supplied the saved canvas is resumed; otherwise the client creates
        # or loads a session explicitly.
        if session_id:
            try:
                await self.load(session_id)
            except Exception as ex:  # pragma: no cover
                print(f"[canvas] load {session_id} failed: {ex}")

    async def shutdown(self):
        """Best-effort save and stop every agent even if one stop operation fails."""
        try:
            self.save()
        except Exception:
            pass
        for egg in list(self.eggs.values()):
            try:
                await egg.stop()
            except Exception as ex:  # pragma: no cover - defensive cleanup
                print(f"[canvas] failed to stop {getattr(egg, 'id', '?')}: {ex}")
        try:
            await self.commander.stop()
        except Exception as ex:  # pragma: no cover
            print(f"[canvas] failed to stop commander: {ex}")

    # ---- session (canvas) persistence ----
    def _new_session(self, label=None, clear_notifications=True):
        """Create a fresh dated session directory and switch this canvas to it."""
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sid = ts
        base = SESSIONS_DIR / sid
        n = 1
        while base.exists():
            n += 1
            sid = f"{ts}_{n}"
            base = SESSIONS_DIR / sid
        base.mkdir(parents=True, exist_ok=True)
        (base / "agent_data").mkdir(parents=True, exist_ok=True)
        self.session_id = sid
        self.session_dir = base
        self.created_at = ts
        self.label = label
        self._id = 0
        self._last_save = None
        if clear_notifications:
            self._pending_notifications = {}
            self._task_results = {}
            self._workflow_runs = {}
        # A fresh canvas gets a FRESH commander conversation (not the previous
        # canvas's id/transcript); the commander keeps its own sub-directory
        # (session_dir/commander_data) just like each egg has its worktree.
        self.commander.bind_session(base, None)
        self.commander.transcript = []
        return sid

    def ensure_session(self):
        """Create a session if this canvas is still unbound (empty / fresh)."""
        if self.session_id is None:
            # Binding an existing in-memory canvas must not discard pending inbox items.
            self._new_session(clear_notifications=False)

    def _to_dict(self):
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "label": self.label,
            "history_mode": self.history_mode,
            "reasoning": reasoning_capture_state(),
            "eggs": [
                {**e.spec(), "transcript": e.transcript}
                for e in self.eggs.values()
            ],
            "edges": [dict(ed) for ed in self.edges],
            "commander": self.commander.transcript,
            "commander_conversation_id": self.commander.conversation_id,
            "pending_notifications": self._pending_notifications,
            "task_results": list(self._task_results.values()),
            "workflow_runs": list(self._workflow_runs.values()),
        }

    def autosave(self, min_interval=1.0, force=False):
        """Best-effort persistence after a mutation. Throttled (min_interval) so
        drags/rapid edits don't storm the disk; force=True bypasses the throttle
        for meaningful checkpoints (a finished model turn, a new session). A
        pristine empty canvas is skipped so it doesn't create a stray session dir.
        """
        if self.session_id is None and not self.eggs:
            return
        now = time.monotonic()
        if not force and self._last_save is not None and (now - self._last_save) < min_interval:
            return
        self._last_save = now
        try:
            self.save()
        except Exception:
            pass

    def save(self):
        """Persist layout + per-egg transcripts + conversation ids so the session can
        be resumed. No-op for a pristine empty canvas (no session, no eggs) so a
        blank load doesn't create a stray session dir. Worktree code is already on
        disk; the SDK's conversation state lives under session_dir/agent_data
        (save_dir) keyed by conversation_id."""
        if self.session_id is None and not self.eggs:
            return None
        if self.session_id is None:
            self.ensure_session()
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        path = self.session_dir / "canvas.json"
        path.write_text(
            json.dumps(self._to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    async def load(self, session_id):
        """Resume a saved session and restore ids, histories and pending notifications."""
        sessions_root = SESSIONS_DIR.resolve()
        base = (sessions_root / str(session_id)).resolve()
        # Session ids are directory names returned by list_sessions; reject traversal.
        if base.parent != sessions_root or base == sessions_root:
            raise ValueError(f"invalid session id: {session_id!r}")
        path = base / "canvas.json"
        if not path.exists():
            raise FileNotFoundError(f"session not found: {session_id}")
        data = json.loads(path.read_text(encoding="utf-8"))

        try:
            await self.commander.stop()
        except Exception:
            pass
        for egg in list(self.eggs.values()):
            try:
                await egg.stop()
            except Exception:
                pass

        self.eggs.clear()
        self.edges.clear()
        self._id = 0
        self.session_id = data.get("session_id", str(session_id))
        self.session_dir = base
        self.created_at = data.get("created_at")
        self.label = data.get("label")
        self.history_mode = data.get("history_mode", "brief")

        for ed in data.get("eggs", []):
            try:
                egg = EggAgent(
                    ed["id"], ed["title"], ed["prompt"], ed.get("type", "module"),
                    (ed.get("x", 0), ed.get("y", 0)),
                    subagents=ed.get("subagents"),
                    parent=ed.get("parent"),
                    session_dir=str(base),
                    conversation_id=ed.get("conversation_id"),
                )
                egg.transcript = ed.get("transcript", []) or []
                self.eggs[ed["id"]] = egg
            except (KeyError, TypeError) as ex:
                print(f"[canvas] skipping malformed egg entry: {ex}")

        used_edge_ids = set()
        for edge in data.get("edges", []):
            if (edge.get("from") in self.eggs and edge.get("to") in self.eggs
                    and edge.get("from") != edge.get("to")):
                edge_id = edge.get("id") or self._new_id("edge")
                # Repair duplicate legacy edge ids without changing endpoints.
                if edge_id in used_edge_ids:
                    edge_id = self._new_id("edge")
                used_edge_ids.add(edge_id)
                self.edges.append({"id": edge_id, "from": edge["from"], "to": edge["to"]})

        self.commander.transcript = data.get("commander", []) or []
        self.commander.bind_session(base, data.get("commander_conversation_id"))

        # Restore the id counter from both egg and edge ids (the counter is shared).
        suffixes = []
        for value in list(self.eggs) + [e.get("id", "") for e in self.edges]:
            match = re.search(r"(\d+)$", str(value))
            if match:
                suffixes.append(int(match.group(1)))
        self._id = max(suffixes, default=0)

        raw_inbox = data.get("pending_notifications", {}) or {}
        self._pending_notifications = {
            eid: list(messages) if isinstance(messages, list) else [str(messages)]
            for eid, messages in raw_inbox.items()
            if eid in self.eggs and messages
        }
        raw_task_results = data.get("task_results", []) or []
        self._task_results = {}
        if isinstance(raw_task_results, dict):
            raw_task_results = list(raw_task_results.values())
        for item in raw_task_results:
            if isinstance(item, dict) and item.get("task_id"):
                self._task_results[str(item["task_id"])] = item
        raw_workflow_runs = data.get("workflow_runs", []) or []
        self._workflow_runs = {}
        if isinstance(raw_workflow_runs, dict):
            raw_workflow_runs = list(raw_workflow_runs.values())
        for item in raw_workflow_runs:
            if isinstance(item, dict) and item.get("run_id"):
                self._workflow_runs[str(item["run_id"])] = item
        self._last_save = None
        return self.view()

    @staticmethod
    def list_sessions():
        """Metadata for every saved session, newest first."""
        if not SESSIONS_DIR.exists():
            return []
        out = []
        for d in sorted(SESSIONS_DIR.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            info = {"id": d.name, "created_at": d.name, "label": None, "eggs": 0}
            cj = d / "canvas.json"
            if cj.exists():
                try:
                    data = json.loads(cj.read_text(encoding="utf-8"))
                    info["label"] = data.get("label")
                    info["created_at"] = data.get("created_at", d.name)
                    info["eggs"] = len(data.get("eggs", []))
                except Exception:
                    pass
            out.append(info)
        return out

    def view(self):
        return {
            "eggs": [e.spec() for e in self.eggs.values()],
            "edges": [dict(e) for e in self.edges],
            "history_mode": self.history_mode,
            "reasoning": reasoning_capture_state(),
        }

    def neighbor_context(self, eid):
        """Text snapshot of the conversation histories of eggs directly connected
        to ``eid`` (by an edge, either direction), so a normal chat with ``eid``
        can reference what its neighbours have actually done/decided. Returns ""
        when there are no connected eggs or none have history yet.
        """
        e = self.eggs.get(eid)
        if not e:
            return ""
        nbr_ids = set()
        for ed in self.edges:
            if ed["from"] == eid and ed["to"] in self.eggs:
                nbr_ids.add(ed["to"])
            elif ed["to"] == eid and ed["from"] in self.eggs:
                nbr_ids.add(ed["from"])
        if not nbr_ids:
            return ""
        blocks = []
        for nid in nbr_ids:
            n = self.eggs[nid]
            hist = n.history_text(mode=self.history_mode, limit=3000).strip()
            blocks.append(
                f"### 相连的蛋「{n.title}」（{n.type}）职责：{n.prompt}\n"
                f"会话历史：\n{hist if hist else '（暂无会话记录）'}"
            )
        return ("以下是与你直接相连的蛋的会话历史（仅供参考，用于了解上下文）：\n\n"
                + "\n\n".join(blocks))

    # ---- spawning ----
    async def _spawn(self, title, prompt, type_, x, y, subagents=None, parent=None):
        eid = self._new_id()
        egg = EggAgent(
            eid, title, prompt, type_, (x, y), subagents=subagents, parent=parent,
            session_dir=str(self.session_dir) if self.session_dir is not None else None, conversation_id=None,
        )
        # Agents connect lazily on first use, so boot never depends on a model.
        self.eggs[eid] = egg
        return egg

    async def _seed(self):
        """Initial example so the canvas isn't empty on first load.

        The goal is a top-level idea egg; the two modules are its sub-eggs, each
        living in its own sub-directory under the goal's worktree.
        """
        goal = await self._spawn(
            "目标", "天气查询机器人：用户输入城市，返回未来3天天气。", "idea", 80, 80
        )
        data = await self._spawn(
            "数据获取蛋", "调用天气 API，按城市返回结构化天气数据。", "module",
            480, 360, parent=goal.id,
        )
        view = await self._spawn(
            "展示蛋", "将天气数据渲染为友好的卡片 UI。", "module", 880, 360,
            parent=goal.id,
        )
        self.edges.append({"id": self._new_id("edge"), "from": goal.id, "to": data.id})
        self.edges.append({"id": self._new_id("edge"), "from": goal.id, "to": view.id})

    # ---- goal -> eggs (LLM planner) ----
    async def generate(self, goal):
        # Each "generate idea" opens a NEW dated session so every idea project is
        # self-contained under sessions/<timestamp>/. The previous session (if any)
        # is persisted first so no work is lost.
        if self.eggs:
            try:
                self.save()
            except Exception:  # pragma: no cover
                pass
        for e in list(self.eggs.values()):
            await e.stop()
        # Close the previous commander's LLM connection so the fresh session does
        # not leak the old conversation handle; bind_session then starts clean.
        await self.commander.stop()
        self.eggs.clear()
        self.edges.clear()
        self._new_session(label=(goal[:40] if goal else None))
        # The goal becomes a parent "idea egg" with its own directory; the
        # planner's modules become sub-eggs nested under it (each its own
        # independent sub-directory), so the project has a clear folder tree.
        parent = await self._spawn(
            f"项目：{goal[:24]}", goal, "idea", 80, 80
        )
        specs = await self._plan(goal)
        created = [parent]
        for i, s in enumerate(specs):
            egg = await self._spawn(
                s["title"], s["prompt"], s.get("type", "module"),
                80 + i * 360, 380, parent=parent.id,
            )
            created.append(egg)
            self.edges.append(
                {"id": self._new_id("edge"), "from": parent.id, "to": egg.id}
            )
        self.autosave(force=True)
        return created

    async def _plan(self, goal):
        if not SDK_OK:
            return self._fake_plan(goal)
        try:
            # The planner only returns structured JSON and needs NO tools, so we
            # use CustomSystemInstructions to REPLACE the SDK's default agent
            # prompt (which would otherwise be appended and bloat the context with
            # tool/environment instructions it doesn't need) and pass an empty
            # CapabilitiesConfig to drop the default tool schemas.
            sys_instr = types.CustomSystemInstructions(
                text=(
                    "Decompose the user's goal into 2-4 independent module eggs for "
                    "an idea canvas. Each egg is an autonomous LLM that will implement "
                    "one part. Return concise titles and clear prompts. "
                    "You MUST return AT LEAST 2 eggs and AT MOST 4 — never return a "
                    "single egg, and never return zero."
                )
            )
            # The planner needs NO tools and only returns structured JSON, so we
            # strip all 14 default builtin tool schemas (enabled_tools=[]) and use
            # MINIMAL behavior to prune prompt overhead. Without this the harness
            # still injects the full tool set, bloating the context.
            caps = types.CapabilitiesConfig(
                enabled_tools=[],
                enable_subagents=False,
                agent_behavior=types.AgentBehavior.MINIMAL,
            )
            async with Agent(build_agent_config(
                system_instructions=sys_instr,
                capabilities=caps,
                response_schema=IdeaPlan,
            )) as planner:
                resp = await planner.chat(goal)
                data = await resp.structured_output()
            eggs = data.get("eggs") if isinstance(data, dict) else getattr(data, "eggs", [])
            if not eggs:
                # structured_output() returns None (not an exception) when the
                # local model emits prose / malformed JSON that fails schema
                # validation. Without this guard generate() would spawn only the
                # parent "项目" box and zero modules — "只能生成一个框".
                return self._fake_plan(goal)
            specs = []
            for i, e in enumerate(eggs[:4]):
                title = (e.get("title") if isinstance(e, dict) else getattr(e, "title", "")) or f"模块{i + 1}"
                prompt = (e.get("prompt") if isinstance(e, dict) else getattr(e, "prompt", "")) or ""
                if not prompt:
                    continue
                specs.append({
                    "title": title,
                    "prompt": prompt,
                    "type": (e.get("type") if isinstance(e, dict) else getattr(e, "type", "module")) or "module",
                })
            # Guard against a model that returns an empty/garbage list despite the
            # schema — never hand generate() an empty plan (which would drop all
            # module boxes).
            return specs if specs else self._fake_plan(goal)
        except Exception as ex:  # pragma: no cover
            print(f"[planner] failed, using fallback: {ex}")
            return self._fake_plan(goal)

    def _fake_plan(self, goal):
        return [
            {"title": "数据获取模块",
             "prompt": f"从目标「{goal}」拆解：负责采集/接入外部数据，输出结构化结果。",
             "type": "module"},
            {"title": "处理逻辑模块",
             "prompt": "对数据进行清洗、转换与核心计算，承上启下。", "type": "module"},
            {"title": "输出展示模块",
             "prompt": "将处理结果格式化并呈现给用户（UI/API）。", "type": "module"},
        ]

    # ---- manual add ----
    async def add_egg(self, title="新蛋", prompt="", x=None, y=None):
        self.ensure_session()
        if x is None:
            x = 120 + (len(self.eggs) % 4) * 60
        if y is None:
            y = 120 + (len(self.eggs) % 4) * 60
        egg = await self._spawn(title, prompt, "module", x, y)
        self.autosave()
        return egg

    # ---- combine into a new egg ----
    def _combine_block(self, egg):
        """One source egg's contribution: its role + its historical conversation."""
        header = f"### 蛋「{egg.title}」（{egg.type}）\n职责：{egg.prompt}"
        hist = egg.history_text(mode=self.history_mode).strip()
        return f"{header}\n历史会话：\n{hist if hist else '（暂无会话记录）'}"

    async def combine(self, ids):
        ids = list(dict.fromkeys(i for i in ids if i in self.eggs))
        if not ids:
            return None
        src = [self.eggs[i] for i in ids]
        # Combine each egg's HISTORICAL CONVERSATION (not just its prompt), so the
        # new egg inherits what each part actually did/decided.
        blocks = "\n\n".join(self._combine_block(e) for e in src)
        combined_prompt = (
            f"你是将以下 {len(src)} 个蛋组合而成的整合蛋。请基于它们各自的职责与"
            f"历史会话，协调这些模块并给出一个整合后的实现。\n\n{blocks}"
        )
        subagents = []
        if SDK_OK and BACKEND == "cloud":
            for e in src:
                subagents.append(
                    types.SubagentConfig(
                        name=e.title[:20],
                        description=f"子模块：{e.title}",
                        system_instructions=self._combine_block(e),
                    )
                )
        cx = max(e.x for e in src) + 400
        cy = min(e.y for e in src)
        egg = await self._spawn(
            f"组合蛋#{len(self.eggs) + 1}", combined_prompt, "composed", cx, cy,
            subagents=subagents,
        )
        for i in ids:
            self.edges.append({"id": self._new_id("edge"), "from": i, "to": egg.id})
        self.autosave(force=True)
        return egg

    # ---- mutations ----
    async def delete_egg(self, eid):
        e = self.eggs.pop(eid, None)
        if e:
            await e.stop()
        # Cascade-delete any sub-eggs that belonged to this parent.
        for cid, child in list(self.eggs.items()):
            if child.parent == eid:
                await self.delete_egg(cid)
        self.edges = [ed for ed in self.edges if ed["from"] != eid and ed["to"] != eid]
        self._pending_notifications.pop(eid, None)
        self.autosave(force=True)

    async def delete_edge(self, edge_id):
        self.edges = [ed for ed in self.edges if ed["id"] != edge_id]
        self.autosave()

    def _has_path(self, start, target):
        """Return True if a directed path exists from start to target."""
        stack = [start]
        seen = set()
        adjacency = {}
        for edge in self.edges:
            adjacency.setdefault(edge.get("from"), []).append(edge.get("to"))
        while stack:
            node = stack.pop()
            if node == target:
                return True
            if node in seen:
                continue
            seen.add(node)
            stack.extend(n for n in adjacency.get(node, []) if n not in seen)
        return False

    async def add_edge(self, from_, to):
        if from_ == to or from_ not in self.eggs or to not in self.eggs:
            return False
        if any(ed["from"] == from_ and ed["to"] == to for ed in self.edges):
            return False
        # Adding from_ -> to forms a cycle exactly when to already reaches from_.
        if self._has_path(to, from_):
            return False
        self.ensure_session()
        self.edges.append({"id": self._new_id("edge"), "from": from_, "to": to})
        self.autosave(force=True)
        return True

    async def move(self, eid, x, y):
        e = self.eggs.get(eid)
        if e:
            e.x, e.y = x, y
        self.autosave()

    # ---- deterministic multi-round debate workflow ----
    def _debate_role_for_egg(self, eid):
        """Classify a debate participant from its title, never from its prompt body."""
        egg = self.eggs.get(eid)
        if not egg:
            return None
        title = str(getattr(egg, "title", "") or "").strip().lower()
        if re.search(r"(总结陈词|总结总结|总结汇报|收尾总结|最终总结|finalizer|closing statement)", title):
            return "finalizer"
        if re.search(r"(裁判|评委|评判|裁决|仲裁|judge|referee|adjudicat)", title):
            return "judge"
        if re.search(r"(自由辩论主持|辩论主持|主持人|主持|moderator|host)", title):
            return "host"
        if re.search(r"(正方|支持方|正方辩手|\bpro\b|affirmative)", title):
            return "pro"
        if re.search(r"(反方|反对方|反方辩手|\bcon\b|negative)", title):
            return "con"
        return None

    def _requested_debate_rounds(self, text):
        """Return an explicitly requested debate-round count, or None.

        A round count may be stated literally ("运行一轮辩论", "打3轮"), or implied
        by an explicit start/run verb aimed at a debate ("开始一场辩论", "来辩论",
        "debate 开始") — that is treated as a 1-round execution request so the
        workflow actually runs instead of the commander only giving advice.
        Design/discussion verbs (设计/讨论/规划/准备…) deliberately do NOT count.
        """
        instruction = str(text or "")
        if not re.search(r"(辩论|debate)", instruction, re.I):
            return None
        match = re.search(r"(\d+|一|二|两|三|四)\s*轮", instruction)
        if match:
            raw = match.group(1)
            mapping = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4}
            try:
                rounds = int(raw) if raw.isdigit() else mapping.get(raw, 0)
            except (TypeError, ValueError):
                return None
            return rounds if 1 <= rounds <= 4 else None
        # No literal "X轮" — accept a clear start/run verb as an implicit 1 round.
        if re.search(
            r"(开始|启动|开启|进行|开展|运行|执行|来|跑|搞|弄|组织|举办|发起|直接|马上|开)\s*"
            r"(一\s*)?(场|次|个|轮|局|回合|下)?\s*(辩论|debate)"
            r"|(辩论|debate)\s*(开始|启动|开启|进行|开展|运行|执行|来|搞|发起|开)",
            instruction, re.I):
            return 1
        return None

    def _is_explicit_debate_execution_request(self, text):
        """Distinguish an execution command from a request to discuss/design a debate."""
        instruction = str(text or "").strip()
        if self._requested_debate_rounds(instruction) is None:
            return False
        # Some frontends display/send the mode label as a prefix, e.g. "全局协调建议：".
        # Remove only that known UI label so it cannot accidentally suppress the command.
        command_text = re.sub(
            r"^\s*(?:\*\*)?[（(【]?(?:全局协调建议|协调建议|全局协调)[）)】]?(?:\*\*)?\s*[:：\\\n]*",
            "", instruction, count=1)
        # Leading advice/planning formulations should not start agents. Do not reject a
        # genuine command merely because the UI mode label contains the word "建议".
        if re.search(r"^\s*(?:我?建议|如何|怎么|设计|规划|分析|解释|为什么|不要|别\s*开始|先别|不执行)", command_text):
            return False
        return bool(re.search(
            r"(开始|启动|开展|组织|进行|主持|来一场|打一场|start|run|conduct|organize).{0,24}(辩论|debate)"
            r"|(辩论|debate).{0,16}(开始|启动|开展|组织|进行|start|run|conduct)",
            command_text, re.I))

    def _edge_connected_components(self):
        """Return weakly connected components based only on actual canvas edges."""
        adjacency = {eid: set() for eid in self.eggs}
        for edge in self.edges:
            src, dst = edge.get("from"), edge.get("to")
            if src in adjacency and dst in adjacency and src != dst:
                adjacency[src].add(dst)
                adjacency[dst].add(src)
        components = []
        unseen = set(adjacency)
        # Preserve canvas insertion order within components for stable UI ordering.
        order = {eid: i for i, eid in enumerate(self.eggs)}
        while unseen:
            first = min(unseen, key=lambda eid: order[eid])
            stack, found = [first], set()
            while stack:
                current = stack.pop()
                if current in found:
                    continue
                found.add(current)
                stack.extend(adjacency[current] - found)
            unseen.difference_update(found)
            components.append(sorted(found, key=lambda eid: order[eid]))
        return components

    def _resolve_connected_debate_targets(self, scope_ids=None):
        """Resolve one unambiguous debate group from connected canvas nodes.

        If a project/root or only some participants were passed as targets, restrict the
        search to connected components intersecting those ids. Structural/project eggs
        may connect participant eggs, but only role-labelled participants are dispatched.
        Disconnected role groups are never silently combined.
        """
        scope_ids = set(scope_ids or [])
        qualified = []
        for component in self._edge_connected_components():
            if scope_ids and not (scope_ids & set(component)):
                continue
            roles = {key: [] for key in ("pro", "con", "host", "judge", "finalizer")}
            for eid in component:
                role = self._debate_role_for_egg(eid)
                if role:
                    roles[role].append(eid)
            if len(roles["pro"]) == 1 and len(roles["con"]) == 1 and (
                    roles["host"] or roles["judge"] or roles["finalizer"]):
                participants = [eid for eid in component if self._debate_role_for_egg(eid)]
                qualified.append((participants, roles))

        if len(qualified) == 1:
            return qualified[0][0], ""
        if not qualified:
            return [], (
                "没有在同一个连线连通组中找到唯一的一名正方、一名反方，"
                "以及至少一名主持人、裁判或总结节点。请确认这些蛋已在画布中连线，"
                "并且标题能明确识别角色。"
            )
        return [], (
            f"当前画布有 {len(qualified)} 组符合条件的辩论参与者，无法判断应运行哪一组。"
            "请选中目标辩论组后再下达指令。"
        )

    def _detect_debate_workflow(self, text, targets):
        """Recognize explicit round-count debate requests with role-labelled targets."""
        rounds = self._requested_debate_rounds(text)
        if rounds is None:
            return None
        roles = {key: [] for key in ("pro", "con", "host", "judge", "finalizer")}
        for eid in targets:
            role = self._debate_role_for_egg(eid)
            if not role:
                return None
            roles[role].append(eid)
        # Exactly one pro and one con are required. Do not silently choose among
        # multiple debaters; that would make stage completion ambiguous.
        if len(roles["pro"]) != 1 or len(roles["con"]) != 1:
            return None
        if any(len(roles[k]) > 1 for k in ("host", "judge", "finalizer")):
            return None
        if not (roles["host"] or roles["judge"] or roles["finalizer"]):
            return None
        return {"rounds": rounds, **roles}

    async def _run_debate_workflow(self, text, targets, workflow):
        """Run: opening statements -> N rebuttal rounds -> adjudication/closing.

        The controller, not the LLM planner, owns phase transitions. Results are stored
        by task_id and phase and are not overwritten when the same egg speaks again.
        Notifications are injected into a real task's context, but never generate tasks
        by themselves.
        """
        summary = self._commander_context(targets, mode="brief")
        boxes, channels, cmd_hist = {}, {}, [{"role": "user", "text": text}]
        inbox = self._pending_notifications
        notified, all_results = [], []
        running, bg = {}, []
        events = asyncio.Queue()
        stage_results = []
        phase_outputs = {}
        run_key = f"debate-{time.time_ns()}"
        workflow_record = {
            "run_id": run_key,
            "workflow": "debate",
            "instruction": str(text),
            "targets": list(targets),
            "roles": {key: list(workflow.get(key) or []) for key in
                      ("pro", "con", "host", "judge", "finalizer")},
            "rounds": int(workflow["rounds"]),
            "status": "running",
            "current_phase": "starting",
            "completed_phases": [],
            "failure": "",
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        self._workflow_runs[run_key] = workflow_record
        self.autosave(force=True)
        normal_exit = False
        final_summary = ""
        phase_index = 0
        failed_phase = None
        # Eggs that were still talking to the LLM when the workflow ended.
        stopped_eggs = []
        stop_note = ""

        pro_id = workflow["pro"][0]
        con_id = workflow["con"][0]
        host_id = workflow["host"][0] if workflow["host"] else None
        judge_id = workflow["judge"][0] if workflow["judge"] else None
        finalizer_id = workflow["finalizer"][0] if workflow["finalizer"] else None
        total_rounds = workflow["rounds"]

        def context_text():
            parts = [f"【用户原始指令】\n{text}",
                     "【当前工作流约束】\n本工作流由调度器按阶段控制；不得宣布其他阶段已完成，不得跳过尚未执行的阶段。"]
            for item in stage_results:
                parts.append(
                    f"【已完成阶段：{item['phase']}｜{item['title']}｜task_id={item['task_id']}】\n"
                    f"{item['result'][:5000]}"
                )
            joined = "\n\n".join(parts)
            # Keep a bounded context while retaining the most recent exchange.
            # The bound must stay well below the model's context window: the
            # egg's own instructions + tool schemas already cost ~5.5k tokens,
            # and an oversized prompt made the run fail with
            # "the most recent step could not fit in the available context limit".
            if len(joined) > 7000:
                joined = joined[:2500] + "\n\n【较早阶段内容省略】\n\n" + joined[-4500:]
            return joined

        async def execute_phase(phase_key, phase_label, task_specs):
            """Yield UI events while running one barrier-synchronized stage."""
            nonlocal phase_index, failed_phase
            phase_index += 1
            # Every phase's eggs should be able to read the OTHER participants'
            # deliverable directories (the worktree roots the user sees in
            # 「审阅代码」). Use the full coordinated set (`targets`), not just
            # this batch, so a single-egg phase (e.g. the judge) still receives
            # all peer directories.
            all_peers = list(targets)
            workflow_record["current_phase"] = phase_key
            workflow_record["status"] = "running"
            workflow_record["updated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
            self.autosave(force=True)
            self.commander._record("system", "phase", text=f"【阶段 {phase_index}：{phase_label}】")
            yield {"k": "say_break"}
            yield {"k": "say", "text": f"【辩论阶段 {phase_index}：{phase_label}】正在执行；本阶段完成前不会进入下一阶段。", "role": "system"}

            # Freeze context at the stage boundary. A retry in this same stage must not
            # accidentally read another participant's newly generated same-stage reply.
            phase_context = context_text()
            pending = [(dict(spec), 0) for spec in task_specs]
            phase_successes = {}
            phase_failures = []
            while pending:
                batch = []
                attempts_by_id = {}
                for original, attempt in pending:
                    eid = original["egg_id"]
                    task_id = f"{run_key}:{phase_key}:{eid}:attempt{attempt + 1}"
                    step = {
                        "egg_id": eid,
                        "title": self.eggs[eid].title,
                        "message": original["message"] + (
                            "\n\n【受控重试】上一次输出未成功保存。请重新完成本阶段任务，只输出本阶段交付物。"
                            if attempt else ""),
                        "deliver": [], "notify": [], "depends_on": [],
                        "phase": phase_key, "task_id": task_id,
                        "goal_context": original.get("goal_context") or phase_context,
                        "_debate_attempt": attempt,
                    }
                    batch.append(step)
                    attempts_by_id[eid] = (original, attempt)

                plan_steps = [{
                    "egg_id": st["egg_id"], "title": st["title"],
                    "message": st["message"], "deliver": [], "notify": [],
                    "depends_on": [], "mode": "runnable", "phase": phase_key,
                    "task_id": st["task_id"],
                } for st in batch]
                yield {"k": "plan", "round": phase_index, "steps": plan_steps}
                for st in plan_steps:
                    yield {"k": "status", "egg": st["egg_id"], "status": "planned"}

                started_ids = []
                # Preflight the entire barrier before starting anyone. This avoids
                # a partial phase where e2 begins but e4 is already busy, leaving
                # an externally visible half-round before the phase is rejected.
                unavailable = []
                for st in batch:
                    eid = st["egg_id"]
                    egg = self.eggs.get(eid)
                    if (egg is None or eid in running or
                            bool(getattr(egg, "busy", False))):
                        unavailable.append(eid)
                if unavailable:
                    started = 0
                else:
                    started = self._start_runs(
                        batch, boxes, channels, inbox, events, running, bg,
                        started_ids=started_ids, goal_context=None,
                        extra_peers=all_peers,
                    )
                if started != len(batch):
                    # A phase is a barrier: partial dispatch must fail the whole
                    # phase instead of retrying just the tasks that never started.
                    missing_steps = [st for st in batch if st["egg_id"] not in started_ids]
                    reasons = [
                        f"{st['title']} 无法启动（节点不存在、已在其他运行中，或启动失败）"
                        for st in missing_steps
                    ]
                    failure_note = (
                        f"阶段「{phase_label}」未能完整启动；已启动的同阶段任务会被取消，"
                        "未启动任务不会单独重试，以免破坏阶段一致性。"
                        + ("原因：" + "；".join(reasons) if reasons else "")
                    )
                    cancel_tasks = []
                    started_records = []
                    for eid in started_ids:
                        rec = running.pop(eid, None)
                        if rec is None:
                            continue
                        started_records.append(rec)
                        for key in ("pump", "consumer"):
                            task = rec.get(key)
                            if task is not None and not task.done():
                                task.cancel()
                                cancel_tasks.append(task)
                    if cancel_tasks:
                        await asyncio.gather(*cancel_tasks, return_exceptions=True)
                    now = datetime.datetime.now().isoformat(timespec="seconds")
                    # Persist a failed task record for every requested member of
                    # the stage, including members that never started.
                    for st in batch:
                        task_id = str(st.get("task_id") or f"{run_key}:{phase_key}:{st['egg_id']}")
                        self._task_results[task_id] = {
                            "task_id": task_id,
                            "egg_id": st["egg_id"],
                            "title": st["title"],
                            "phase": phase_key,
                            "message": str(st.get("message") or ""),
                            "status": "failed",
                            "result": failure_note,
                            "updated_at": now,
                        }
                    for rec in started_records:
                        rec["failed"] = True
                        rec["cancelled"] = True
                        rec["done"] = True
                        rec["error"] = failure_note
                    failed_phase = phase_label
                    workflow_record["status"] = "failed"
                    workflow_record["failure"] = failure_note
                    workflow_record["updated_at"] = now
                    self.autosave(force=True)
                    for st in plan_steps:
                        yield {"k": "status", "egg": st["egg_id"], "status": "idle"}
                        yield {"k": "end_egg", "egg": st["egg_id"]}
                    yield {"k": "say", "text": failure_note, "role": "system"}
                    return
                active = set(started_ids)
                retry_pending = []
                while active:
                    while not events.empty():
                        yield events.get_nowait()
                    finished_ids = [eid for eid in active
                                    if eid in running and running[eid].get("done")]
                    if not finished_ids:
                        event = await self._next_event(events, _MONITOR_INTERVAL)
                        if event is not None:
                            yield event
                        continue
                    for eid in finished_ids:
                        rec = running.pop(eid)
                        active.discard(eid)
                        # A stage counts as finished ONLY when the egg really
                        # concluded something. A turn that produced nothing but
                        # 💭 thinking / 🔧 tool chatter, or that was cut off
                        # (stall watchdog / upstream error), is a FAILURE — it
                        # must not be reported as "阶段完成" while the egg is
                        # still talking to the LLM.
                        has_output = self._run_has_output(rec)
                        if (not (rec.get("reply") or "").strip()
                                or not has_output
                                or self._run_was_cut_off(rec)):
                            rec["failed"] = True
                            rec["error"] = rec.get("error") or (
                                "本轮没有产出结论（只有思考/工具过程）"
                                if not has_output else
                                "本轮被中断，未产出完整结论")
                        self._file_result(rec, boxes, channels, cmd_hist, inbox,
                                          notified, all_results)
                        yield {"k": "status", "egg": eid, "status": "idle"}
                        yield {"k": "end_egg", "egg": eid}
                        original, attempt = attempts_by_id[eid]
                        reply = rec.get("reply") or ""
                        if rec.get("failed"):
                            if attempt < 1:
                                retry_pending.append((original, attempt + 1))
                            else:
                                reason = rec.get("error") or "空输出"
                                phase_failures.append(
                                    f"{self.eggs[eid].title} 在重试后仍未成功"
                                    f"（{reason}）：{reply[:300]}")
                        else:
                            # Only the assistant's own answer may travel to the
                            # next stage / be embedded into another egg's prompt
                            # (e.g. the host's "本轮问题结果" handed to pro/con).
                            # The raw `reply` still carries 💭 thinking and 🔧 tool
                            # chatter, which would pollute the receiver's context.
                            phase_successes[eid] = self._assistant_only(self.eggs[eid], rec)
                            stage_results.append({
                                "task_id": rec["step"].get("task_id"),
                                "egg_id": eid,
                                "title": self.eggs[eid].title,
                                "phase": phase_key,
                                # Only the assistant's own text travels to the
                                # next stage — never the 💭 thinking / 🔧 tools.
                                "result": self._assistant_only(self.eggs[eid], rec),
                            })
                            self.autosave(force=True)
                pending = retry_pending
                if phase_failures:
                    failed_phase = phase_label
                    workflow_record["status"] = "failed"
                    workflow_record["failure"] = "; ".join(phase_failures)
                    workflow_record["updated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
                    self.autosave(force=True)
                    return
            # Do not move to a next stage until every member of this stage succeeded.
            phase_outputs[phase_key] = dict(phase_successes)
            workflow_record["completed_phases"].append(phase_key)
            workflow_record["updated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
            self.autosave(force=True)

        try:
            role_intro = (
                f"正方：{self.eggs[pro_id].title}（{pro_id}）；反方：{self.eggs[con_id].title}（{con_id}）。"
                f"本次要求 {total_rounds} 轮交锋，另含开篇立论和最终裁判/总结。"
            )
            self.commander._record("assistant", "text", text=role_intro)
            yield {"k": "say_break"}
            yield {"k": "say", "text": "已识别为分阶段辩论任务。" + role_intro, "role": "system"}

            # Stage 1: opening statements in parallel.
            opening = []
            for eid, side in ((pro_id, "正方"), (con_id, "反方")):
                opening.append({
                    "egg_id": eid,
                    "message": (
                        f"【阶段：开篇立论】你是本场{side}。围绕辩题「{text}」明确立场，\n"
                        "给出核心论点、至少三个分论点及其推理链，并预判对方最有力的反驳。\n"
                        "只完成立论，不代替对方回答，不做最终裁决。对于统计数据或研究结论，"
                        "没有可靠来源时不要编造具体数字。"
                    ),
                })
            async for event in execute_phase("opening", "双方开篇立论", opening):
                yield event
            if failed_phase:
                final_summary = f"辩论未完成：阶段「{failed_phase}」失败，后续阶段没有启动。请检查对应子蛋输出后重试。"

            # Rounds: optional host poses focused prompts, then both sides answer in parallel.
            if not failed_phase:
                for round_no in range(1, total_rounds + 1):
                    if host_id:
                        host_task = [{
                            "egg_id": host_id,
                            "message": (
                                f"【主持阶段：第 {round_no} 轮交锋问题】基于此前双方实际输出，设计本轮 1-2 个尖锐、"
                                "对双方公平且能引出实质论证的问题。明确标出问题分别要求正方和反方回应的内容。"
                                "只提出本轮问题和规则，不替双方作答、不裁决。"
                            ),
                        }]
                        async for event in execute_phase(f"round{round_no}_host", f"主持人设计第 {round_no} 轮问题", host_task):
                            yield event
                        if failed_phase:
                            break
                    exchanges = []
                    for eid, side, opponent in ((pro_id, "正方", "反方"), (con_id, "反方", "正方")):
                        moderator_suffix = (
                            f"必须直接回答主持人本轮问题。主持人本轮问题结果：{phase_outputs.get(f'round{round_no}_host', {}).get(host_id, '')[:2000]}"
                            if host_id else "本轮没有单独主持人，请直接针对对方上一阶段论点提出回应和追问。"
                        )
                        exchanges.append({
                            "egg_id": eid,
                            "message": (
                                f"【第 {round_no} 轮交锋】你是{side}。请基于已提供的前序阶段记录，逐点回应{opponent}最近的论证，"
                                "先准确概括对方观点，再给出反驳、依据和反问；承认对方确有说服力的部分并说明边界。"
                                "不要仅重复初始立论，也不要声称未在记录中出现的内容。\n" + moderator_suffix
                            ),
                        })
                    async for event in execute_phase(f"round{round_no}_exchange", f"第 {round_no} 轮正反交锋", exchanges):
                        yield event
                    if failed_phase:
                        break

            # Final assessment: judge first, finalizer after it if separate.
            if not failed_phase and judge_id:
                async for event in execute_phase("adjudication", "裁判综合评判", [{
                    "egg_id": judge_id,
                    "message": (
                        f"【最终裁判】评判辩题「{text}」。必须综合开篇立论和全部 {total_rounds} 轮真实交锋记录，"
                        "比较双方论点的一致性、回应有效性、证据质量和逻辑漏洞；区分事实与价值判断，"
                        "指出双方各自最强论证与仍未解决的问题，并给出有理由的裁决或说明为何难分胜负。"
                        "不得把尚未发生的交锋当作事实，也不要编造数据或引用。"
                    ),
                }]):
                    yield event
                if not failed_phase:
                    finalizer_id = finalizer_id if finalizer_id and finalizer_id != judge_id else None

            if not failed_phase and finalizer_id:
                async for event in execute_phase("closing", "最终总结陈词", [{
                    "egg_id": finalizer_id,
                    "message": (
                        f"【最终总结】根据本场辩题「{text}」的开篇立论、全部 {total_rounds} 轮交锋，"
                        "以及已有裁判意见（如有），分别总结正方、反方最有力的论证，归纳分歧核心与共同点，"
                        "给出中立观众点评。请明确区分已发生的交锋与仍可继续讨论的问题。"
                    ),
                }]):
                    yield event

            if not failed_phase:
                # If a host is the only available controller, ask it to close neutrally.
                if not judge_id and not finalizer_id and host_id:
                    async for event in execute_phase("host_closing", "主持人中立收束", [{
                        "egg_id": host_id,
                        "message": (
                            f"【最终收束】根据开篇立论和全部 {total_rounds} 轮交锋，给出中立总结："
                            "双方最强论点、关键分歧和未决问题。不要偏袒任何一方。"
                        ),
                    }]):
                        yield event
                final_summary = (
                    f"{total_rounds} 轮辩论工作流已完成：双方开篇立论、{total_rounds} 轮交锋"
                    + ("、裁判评判" if judge_id else "")
                    + ("、最终总结" if finalizer_id else "")
                    + "均已按阶段执行并保存任务级结果。"
                )
                # "已完成" may only be claimed when every participant really
                # produced a conclusion. A node whose turn was cut off (or that
                # only produced thinking) is NOT done — say so instead.
                missing = []
                for eid in (pro_id, con_id, host_id, judge_id, finalizer_id):
                    if not eid or eid not in self.eggs:
                        continue
                    if not any(item.get("egg_id") == eid for item in stage_results):
                        missing.append(self.eggs[eid].title)
                if missing:
                    failed_phase = "结论缺失"
                    final_summary = (
                        "辩论未完整完成："
                        + "、".join(missing)
                        + " 没有产出可用结论（运行被中断或只产生了思考过程），"
                        "相关阶段不会被报告为成功。请查看这些节点的轨迹后重试。"
                    )
            workflow_record["status"] = "failed" if failed_phase else "completed"
            if failed_phase:
                workflow_record["failure"] = workflow_record.get("failure") or final_summary
            else:
                workflow_record["failure"] = ""
            workflow_record["updated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
            self.autosave(force=True)
            normal_exit = True
            self.commander._record("assistant", "text", text=final_summary)
            yield {"k": "say_break"}
            yield {"k": "say", "text": final_summary}
        finally:
            # ALWAYS stop whatever is still running. A workflow that "ends" while
            # an egg is still talking to the LLM leaves the model generating on the
            # server (tokens keep burning) and the egg stuck in 运行中. The eggs
            # are interrupted with the SDK's halt_request, so the SERVER-side turn
            # really stops — not just the local reader.
            # NOTE: no `yield` in here: this block also runs while the generator
            # is being closed (GeneratorExit), where yielding is illegal.
            for eid, rec in list(running.items()):
                for key in ("pump", "consumer"):
                    task = rec.get(key)
                    if task is not None and not task.done():
                        task.cancel()
                rec["failed"] = True
                rec["cancelled"] = True
                rec["done"] = True
                if eid not in stopped_eggs:
                    stopped_eggs.append(eid)
            for task in list(bg):
                if not task.done():
                    task.cancel()
            for eid in stopped_eggs:
                egg = self.eggs.get(eid)
                if egg is None:
                    continue
                try:
                    egg.interrupt()
                except Exception:  # pragma: no cover - defensive
                    pass
            if bg:
                await asyncio.gather(*bg, return_exceptions=True)
            if stopped_eggs:
                names = "、".join(self.eggs[e].title for e in stopped_eggs
                                  if e in self.eggs)
                stop_note = (f"⚠ 工作流结束时仍有节点在运行，已中止：{names}。"
                             "这些节点没有产出可用结论，请查看其轨迹后重试。")
                self.commander._record("system", "status", text=stop_note)
                workflow_record["stopped"] = stopped_eggs
                if workflow_record.get("status") == "completed":
                    workflow_record["status"] = "failed"
                    workflow_record["failure"] = stop_note
            if not normal_exit:
                workflow_record["status"] = "interrupted"
                workflow_record["updated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
            self.autosave(force=True)
        if stopped_eggs:
            for eid in stopped_eggs:
                yield {"k": "status", "egg": eid, "status": "idle"}
                yield {"k": "end_egg", "egg": eid}
            yield {"k": "say", "text": stop_note, "role": "system"}
        yield {"k": "end"}

    # ---- commander ----
    def interrupt_commander(self):
        """Stop the in-flight 总指挥 run (the stop button).

        Halts the commander's model turn and cancels the task that streams it.
        When the commander was driving a one-click run ('run'), the eggs it
        started are interrupted as well — pressing their own stop button has
        exactly the same effect.
        """
        kind = getattr(self.commander, "run_kind", None)
        self.commander.interrupt()
        if kind == "run":
            for egg in self.eggs.values():
                if getattr(egg, "busy", False):
                    egg.interrupt()

    async def commander_run(self, text, targets=None):
        """Run explicit debate commands from selected targets or one connected debate group."""
        targets = list(dict.fromkeys(t for t in (targets or []) if t in self.eggs))
        explicit_debate = self._is_explicit_debate_execution_request(text)

        # 轨迹完整性：用户的指令与总指挥的运行环境（system prompt + 工具）必须
        # 出现在总指挥的轨迹里，即使这一轮并不走 commander.run()（例如分阶段
        # 辩论由调度器直接驱动子蛋）。
        self.commander._record_env()
        self.commander._record("user", "text", text=text)

        # Important: the UI's "全局协调建议" entry often supplies no selected IDs.
        # Before falling back to plain Commander chat, resolve an explicit command such
        # as "开始一轮辩论" against one unambiguous, edge-connected debate group.
        if explicit_debate:
            original_targets = list(targets)
            debate = self._detect_debate_workflow(text, targets) if targets else None
            if debate is None:
                # This also handles a frontend that passes the project root, or only
                # some selected participants, instead of an empty target list.
                resolved, reason = self._resolve_connected_debate_targets(scope_ids=targets)
                if not resolved:
                    message = "识别到明确的辩论执行请求，但没有启动任何子蛋。" + reason
                    self.commander._record("assistant", "text", text=message)
                    yield {"k": "say", "text": message, "role": "system"}
                    yield {"k": "end"}
                    return
                targets = resolved
                debate = self._detect_debate_workflow(text, targets)
                if debate is None:
                    message = (
                        "找到了连线范围内的辩论节点，但角色配置仍不唯一。"
                        "请检查正方、反方、主持人/裁判/总结的节点标题，且每种角色只能有一个候选。"
                    )
                    yield {"k": "say", "text": message, "role": "system"}
                    yield {"k": "end"}
                    return
                titles = "、".join(f"{self.eggs[eid].title}（{eid}）" for eid in targets)
                scope_note = "全局画布" if not original_targets else "所选节点所在的连线组"
                yield {"k": "say", "text": f"已从{scope_note}自动识别辩论参与者：{titles}。正在启动分阶段辩论。", "role": "system"}

        debate = self._detect_debate_workflow(text, targets) if explicit_debate else None
        if debate is not None:
            try:
                async for event in self._run_debate_workflow(text, targets, debate):
                    yield event
            except asyncio.CancelledError:
                # Client disconnect / caller cancellation must propagate so the
                # workflow's finally block can mark the run interrupted and clean up.
                raise
            except Exception as ex:
                message = f"辩论工作流异常终止：{type(ex).__name__}: {ex}。后续阶段已停止，请检查日志后重试。"
                # _run_debate_workflow normally marks interruption in its finally;
                # this wrapper upgrades the latest unfinished run to a visible failure.
                for record in reversed(list(self._workflow_runs.values())):
                    if record.get("workflow") == "debate" and record.get("status") in {"running", "interrupted"}:
                        record["status"] = "failed"
                        record["failure"] = message
                        record["updated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
                        break
                self.autosave(force=True)
                self.commander._record("assistant", "text", text=message)
                yield {"k": "say_break"}
                yield {"k": "say", "text": message, "role": "system"}
                # The workflow generator's trailing end event is skipped when it
                # raises; emit the terminal UI event here for the error path.
                yield {"k": "end"}
            return
        if explicit_debate:
            message = (
                "识别到辩论执行请求，但无法建立唯一的阶段工作流。请确保目标中恰有一名正方、一名反方，"
                "并且至少有一名主持人、裁判或总结节点；同一角色不能有多个候选。"
                "如果使用全局协调，请将参与者放在同一个连线组中。"
            )
            self.commander._record("assistant", "text", text=message)
            yield {"k": "say", "text": message, "role": "system"}
            yield {"k": "end"}
            return

        summary = self._commander_context(targets)
        if not targets:
            async for token in self.commander.run(text, summary):
                yield {"k": "say", "text": token}
            yield {"k": "end"}
            return

        summary = self._commander_context(targets, mode="brief")
        boxes = {}
        channels = {}
        cmd_hist = [{"role": "user", "text": text}]
        # Instance/session-owned inbox: notifications to currently unselected eggs
        # remain available when those eggs are selected in a later commander_run.
        inbox = self._pending_notifications
        notified = []
        running = {}
        events = asyncio.Queue()
        watch_base = {}
        postponed = []
        reported = set()
        all_results = []
        final_summary = ""
        round_no = 0
        idle_rounds = 0
        bg = []
        normal_exit = False
        target_set = set(targets)

        def task_copy(task):
            eid = task.get("egg_id")
            egg = self.eggs.get(eid)
            return {
                "egg_id": eid,
                "title": task.get("title") or (egg.title if egg else eid),
                "message": task.get("message") or "",
                "deliver": list(task.get("deliver") or []),
                "notify": list(task.get("notify") or []),
                "depends_on": list(task.get("depends_on") or []),
                "_pending_reason": task.get("_pending_reason", ""),
                "_dependency_errors": list(task.get("_dependency_errors") or []),
            }

        def dependencies_ready(task):
            eid = task.get("egg_id")
            deps = list(task.get("depends_on") or [])
            if task.get("_dependency_errors") or eid in deps:
                return False
            return all(dep in boxes and dep not in running for dep in deps)

        def upsert_pending(mapping, task, reason=None):
            if task.get("egg_id") not in target_set:
                return
            item = task_copy(task)
            if reason:
                item["_pending_reason"] = reason
            mapping[item["egg_id"]] = item

        try:
            while True:
                # Forward events produced by background pumps/consumers.
                while not events.empty():
                    yield events.get_nowait()

                # File all completed runs before making the next decision.
                finished = [eid for eid, rec in list(running.items()) if rec.get("done")]
                for eid in finished:
                    rec = running.pop(eid)
                    # Only a run that really concluded counts as a result: a turn
                    # that only produced 💭 thinking / 🔧 tool chatter, or that was
                    # cut off (stall watchdog / upstream error), is a FAILURE —
                    # it must not be filed as a delivered conclusion.
                    has_output = self._run_has_output(rec)
                    if (not (rec.get("reply") or "").strip()
                            or not has_output
                            or self._run_was_cut_off(rec)):
                        rec["failed"] = True
                        rec["error"] = rec.get("error") or (
                            "本轮没有产出结论（只有思考/工具过程）"
                            if not has_output else
                            "本轮被中断，未产出完整结论")
                    self._file_result(rec, boxes, channels, cmd_hist, inbox,
                                      notified, all_results)
                    yield {"k": "status", "egg": eid, "status": "idle"}
                    yield {"k": "end_egg", "egg": eid}

                watch = self._watch_eggs(targets, watch_base, inbox)
                for eid, state in watch.items():
                    if state["delta"] > 0:
                        reported.discard((eid, "stuck"))
                        reported.discard((eid, "overdue"))
                alerts = self._watch_alerts(watch, reported, cmd_hist)

                # Ask for another decision after completions/alerts or when idle.
                if finished or alerts or not running:
                    round_no += 1
                    if round_no > _MAX_ROUNDS:
                        pending_names = ", ".join(
                            f"{p.get('title', p.get('egg_id'))}"
                            for p in postponed
                        )
                        final_summary = (
                            f"已达到指挥轮数上限 {_MAX_ROUNDS}，流程被安全停止。"
                            + (f" 尚有待办任务：{pending_names}。" if pending_names else "")
                            + "请检查未完成的子蛋及其依赖。"
                        )
                        break

                    yield {"k": "say_break"}
                    dec_task = asyncio.create_task(self._commander_decide(
                        text, summary, targets, boxes, channels, cmd_hist,
                        round_no, watch=watch, notified=list(notified),
                        deferred=list(postponed),
                        sink=lambda value: events.put_nowait({"k": "say", "text": value}),
                    ))
                    bg.append(dec_task)
                    while not dec_task.done():
                        event = await self._next_event(events, 0.5)
                        if event is not None:
                            yield event
                    decision = await dec_task
                    if dec_task in bg:
                        bg.remove(dec_task)

                    tasks = []
                    for raw_task in (decision.get("tasks") or []):
                        if not isinstance(raw_task, dict):
                            continue
                        eid = self._resolve_egg_reference(raw_task.get("egg_id"), target_set)
                        message = raw_task.get("message")
                        if eid not in target_set or eid not in self.eggs or not message:
                            continue
                        dependency_holder = {
                            "egg_id": eid,
                            "depends_on": raw_task.get("depends_on"),
                            "_dependency_errors": list(raw_task.get("_dependency_errors") or []),
                        }
                        normalized_dependencies = self._normalize_task_refs(
                            dependency_holder, "depends_on", set(self.eggs))
                        tasks.append({
                            "egg_id": eid,
                            "title": self.eggs[eid].title,
                            "message": str(message),
                            # Titles are accepted only when they exactly match a unique egg.
                            "deliver": [d for d in self._normalize_egg_refs(
                                raw_task.get("deliver"), set(self.eggs), "deliver") if d != eid],
                            "notify": [n for n in self._normalize_egg_refs(
                                raw_task.get("notify"), set(self.eggs), "notify") if n != eid],
                            "depends_on": normalized_dependencies,
                            "_dependency_errors": list(dependency_holder.get("_dependency_errors") or []),
                        })
                    # Models sometimes emit duplicate variants for one egg. Keep the most
                    # detailed instruction and merge authorization/dependency metadata.
                    tasks = self._dedupe_commander_tasks(tasks)

                    action = decision.get("action")
                    if action == "error":
                        final_summary = decision.get("summary") or (
                            "总指挥本轮决策无法解析，已停止自动派发，未将未完成任务报告为成功。"
                        )
                        self.commander._record("assistant", "text", text=final_summary)
                        # Any in-flight tasks are cancelled by the cleanup path. The caller
                        # receives a clear failure instead of a guessed fallback dispatch.
                        break
                    if action == "finish":
                        ready_pending = [p for p in postponed
                                         if p.get("egg_id") not in running
                                         and dependencies_ready(p)]
                        if ready_pending:
                            tasks = [task_copy(p) for p in ready_pending]
                            action = "dispatch"
                            decision["note"] = "继续执行依赖已满足的待办任务"
                            cmd_hist.append({"role": "user", "text":
                                             "【调度保护】仍有依赖已满足的待办任务，不能结束。"})
                        else:
                            # A finish decision cannot abandon in-flight work.
                            if running:
                                action = "dispatch"
                                tasks = []
                                cmd_hist.append({"role": "user", "text":
                                                 "【调度保护】仍有子蛋运行中，暂不接受 finish；"
                                                 "请等待结果返回后再判断是否完成。"})
                            else:
                                unsatisfied = [p for p in postponed
                                               if not dependencies_ready(p)]
                                # Never break a declared dependency cycle by manufacturing
                                # a producer task with depends_on=[] during finish recovery.
                                cycle_nodes = self._find_dependency_cycle(postponed)
                                if cycle_nodes:
                                    final_summary = (
                                        "任务未能完整完成：待办任务存在显式依赖循环："
                                        + " -> ".join(cycle_nodes)
                                        + "。请修正 depends_on 后重新运行；未完成任务不会被报告为成功。"
                                    )
                                    normal_exit = True
                                    break
                                missing_deps = []
                                for pending in unsatisfied:
                                    for dep in (pending.get("depends_on") or []):
                                        if dep in target_set and dep not in boxes and dep not in running:
                                            missing_deps.append((dep, pending))
                                # Force missing producers to run rather than finishing while
                                # a deferred reviewer waits forever for their conclusions.
                                forced = {}
                                for dep, pending in missing_deps:
                                    if dep in self.eggs and dep not in forced:
                                        forced[dep] = {
                                            "egg_id": dep,
                                            "title": self.eggs[dep].title,
                                            "message": (
                                                f"请完成你在本项目中的职责「{self.eggs[dep].title}」，"
                                                "并给出明确、可核验的结论。只处理本节点负责的部分。"
                                            ),
                                            "deliver": [], "notify": [], "depends_on": [],
                                        }
                                # Never accept finish while any selected egg lacks a successful
                                # conclusion box. If needed, dispatch the missing targets too.
                                missing_targets = [target for target in targets
                                                   if target not in boxes and target not in running]
                                # Notifications are context for a future real task, not tasks by themselves.
                                must_run_targets = list(dict.fromkeys(missing_targets))
                                # Dispatch missing explicit producers first. If there are no
                                # such producers, recover one target at a time in selection
                                # order; never guess that a host/finalizer is ready in parallel.
                                if not forced and must_run_targets:
                                    missing_target = must_run_targets[0]
                                    egg = self.eggs[missing_target]
                                    notification_instruction = (
                                        "请先阅读并处理你收到的完成通知，再结合相关结论继续完成职责，"
                                        "明确说明如何使用通知中的信息。"
                                    ) if inbox.get(missing_target) else ""
                                    forced[missing_target] = {
                                        "egg_id": missing_target,
                                        "title": egg.title,
                                        "message": (notification_instruction or
                                            f"请完成你在本项目中的职责「{egg.title}」，"
                                            "并给出明确、可核验的结论。只处理本节点负责的部分。"),
                                        "deliver": [], "notify": [], "depends_on": [],
                                    }
                                if forced:
                                    tasks = list(forced.values())
                                    action = "dispatch"
                                    decision["note"] = "先补齐未产出结论的目标或待办任务"
                                    cmd_hist.append({"role": "user", "text":
                                                     "【调度保护】仍有未产出成功结论的目标或缺失上游，不能结束；已自动派发。"})
                                elif unsatisfied:
                                    # Distinguish an unselected dependency from an actual cycle.
                                    blockers = []
                                    for pending in unsatisfied:
                                        for dep in (pending.get("depends_on") or []):
                                            if dep not in boxes and dep not in running:
                                                if dep not in target_set:
                                                    blockers.append(f"{pending.get('title', pending.get('egg_id'))} 依赖未选中节点 {dep}")
                                                else:
                                                    blockers.append(f"{pending.get('title', pending.get('egg_id'))} 等待 {dep}")
                                    final_summary = (
                                        "任务未能完整完成：" + ("；".join(dict.fromkeys(blockers)) or
                                        "仍有待办任务的显式依赖未满足") +
                                        "。已停止自动调度，避免错误地把未完成流程报告为成功。"
                                    )
                                    normal_exit = True
                                    break
                                else:
                                    final_summary = decision.get("summary") or "指令处理结束。"
                                    normal_exit = True
                                    break

                    # If the model omitted an already-ready postponed task, force it
                    # into this decision so a pending task is never silently forgotten.
                    if action != "finish":
                        existing_ids = {t["egg_id"] for t in tasks}
                        for pending in postponed:
                            eid = pending.get("egg_id")
                            if (eid in target_set and eid not in existing_ids
                                    and eid not in running and dependencies_ready(pending)):
                                tasks.append(task_copy(pending))
                                existing_ids.add(eid)
                    # If the model returns an empty dispatch while postponed work still
                    # waits for selected producers, schedule those producers instead of
                    # spinning until the idle limit.
                    existing_task_ids = {t.get("egg_id") for t in tasks}
                    pending_producers = {}
                    for pending in postponed:
                        for dep in (pending.get("depends_on") or []):
                            if (dep in target_set and dep not in boxes and dep not in running
                                    and dep not in existing_task_ids and dep in self.eggs):
                                pending_producers.setdefault(dep, pending)
                    for dep, consumer in pending_producers.items():
                        tasks.append({
                            "egg_id": dep,
                            "title": self.eggs[dep].title,
                            "message": (
                                f"请完成你在本项目中的职责「{self.eggs[dep].title}」，"
                                "并给出明确、可核验的结论。只处理本节点负责的部分。"
                            ),
                            "deliver": [], "notify": [], "depends_on": [],
                        })
                        existing_task_ids.add(dep)

                    # Validate explicitly declared dependency edges, then schedule missing
                    # producers named in depends_on when they are selected and idle.
                    if tasks:
                        self._resolve_dependencies(tasks, boxes, eligible_ids=targets)
                        task_ids = {t["egg_id"] for t in tasks}
                        producers = {}
                        for task in list(tasks):
                            for dep in task.get("depends_on") or []:
                                if (dep in target_set and dep not in boxes and dep not in running
                                        and dep not in task_ids and dep in self.eggs):
                                    producers.setdefault(dep, task)
                        for dep, consumer in producers.items():
                            tasks.append({
                                "egg_id": dep,
                                "title": self.eggs[dep].title,
                                "message": (
                                    f"请完成你在本项目中的职责「{self.eggs[dep].title}」，"
                                    "并给出明确、可核验的结论。只处理本节点负责的部分。"
                                ),
                                "deliver": [], "notify": [], "depends_on": [],
                            })
                        # Dedupe again after synthesizing producer tasks.
                        tasks = self._dedupe_commander_tasks(tasks)
                        self._resolve_dependencies(tasks, boxes, eligible_ids=targets)

                    # A new instruction for a currently-running egg must be queued,
                    # never silently dropped by _start_runs.
                    busy_tasks = [t for t in tasks if t["egg_id"] in running]
                    tasks = [t for t in tasks if t["egg_id"] not in running]
                    runnable, deferred = self._split_by_dependency(
                        tasks, boxes, eligible_ids=targets, running_ids=set(running)
                    )

                    pending_map = {p.get("egg_id"): task_copy(p) for p in postponed
                                   if p.get("egg_id") in target_set}
                    for task in busy_tasks:
                        upsert_pending(pending_map, task,
                                       "当前蛋仍在运行；新指令排队等待本次运行结束")
                    for task in deferred:
                        upsert_pending(pending_map, task,
                                       task.get("_pending_reason") or "等待依赖结果")
                    for task in runnable:
                        pending_map.pop(task["egg_id"], None)
                    postponed = list(pending_map.values())

                    # If every task is deferred and no producer is running, the remaining
                    # in-round dependency graph has no runnable root (normally a real cycle).
                    # Never execute an arbitrary task to conceal a cycle.
                    if (tasks or postponed) and not runnable and not running:
                        pending_for_check = list({
                            p.get("egg_id"): p for p in (postponed + tasks)
                            if p.get("egg_id")
                        }.values())
                        cycle_nodes = self._find_dependency_cycle(pending_for_check)
                        invalid_refs = [
                            (p.get("egg_id"), err)
                            for p in pending_for_check
                            for err in (p.get("_dependency_errors") or [])
                        ]
                        unselected_deps = [
                            (p.get("egg_id"), dep)
                            for p in pending_for_check
                            for dep in (p.get("depends_on") or [])
                            if dep in self.eggs and dep not in target_set
                            and dep not in boxes and dep not in running
                        ]
                        waiting = "、".join(
                            f"{p.get('title', p.get('egg_id'))}({p.get('_pending_reason', '依赖未满足')})"
                            for p in pending_for_check
                        )
                        if cycle_nodes:
                            reason = "检测到显式 depends_on 循环：" + " -> ".join(cycle_nodes)
                        elif invalid_refs:
                            reason = f"存在无法解析的依赖引用：{invalid_refs}。请修正依赖中的节点 ID/唯一标题。"
                        elif unselected_deps:
                            reason = f"依赖指向本次未选中的节点：{unselected_deps}。请一并选择上游节点后重试。"
                        else:
                            reason = "依赖尚未满足，且当前没有可运行的上游任务。请检查显式 depends_on。"
                        final_summary = (
                            "无法继续调度：" + reason
                            + (f" 待办：{waiting}。" if waiting else "")
                            + "系统已停止自动调度，未将未完成任务报告为成功。"
                        )
                        normal_exit = True
                        break

                    plan_steps = []
                    mode = "并行" if len(runnable) > 1 else ("串行" if runnable else "等待")
                    for task in runnable:
                        plan_steps.append({
                            "egg_id": task["egg_id"],
                            "title": self.eggs[task["egg_id"]].title,
                            "message": task["message"],
                            "deliver": [d for d in (task.get("deliver") or [])
                                        if d in boxes and d not in running],
                            "notify": list(task.get("notify") or []),
                            "depends_on": list(task.get("depends_on") or []),
                            "mode": "runnable",
                        })
                    for task in deferred:
                        if task.get("egg_id") not in self.eggs:
                            continue
                        plan_steps.append({
                            "egg_id": task["egg_id"],
                            "title": self.eggs[task["egg_id"]].title,
                            "message": task.get("message", ""),
                            "deliver": task.get("deliver") or [],
                            "notify": task.get("notify") or [],
                            "depends_on": task.get("depends_on") or [],
                            "mode": "deferred",
                        })
                    for task in busy_tasks:
                        if task.get("egg_id") not in self.eggs:
                            continue
                        plan_steps.append({
                            "egg_id": task["egg_id"],
                            "title": self.eggs[task["egg_id"]].title,
                            "message": task.get("message", ""),
                            "deliver": task.get("deliver") or [],
                            "notify": task.get("notify") or [],
                            "depends_on": task.get("depends_on") or [],
                            "mode": "queued",
                        })

                    note = decision.get("note") or ""
                    narration = (f"【第 {round_no} 轮 · {mode}】{note}\n" + "\n".join(
                        f"→ [{step['title']}] {step['message']}"
                        + (f"（授权转交：{', '.join(step['deliver'])}）" if step["deliver"] else "")
                        + (f"（完成后通知：{', '.join(step['notify'])}）" if step["notify"] else "")
                        + (f"（等待依赖：{', '.join(step['depends_on'])}）"
                           if step["mode"] == "deferred" else "")
                        + ("（排队等待当前运行结束）" if step["mode"] == "queued" else "")
                        for step in plan_steps
                    ))
                    monitor_line = self._fmt_watch_line(watch)
                    if monitor_line:
                        narration += "\n📡 监控：" + monitor_line
                    if alerts:
                        narration += "\n⚠ " + "\n⚠ ".join(alerts)
                    self.commander._record("system", "plan", text=narration)
                    self.commander._record("assistant", "text", text=narration)
                    cmd_hist.append({"role": "assistant", "text": narration})
                    yield {"k": "say_break"}
                    yield {"k": "say", "text": narration, "role": "system"}
                    yield {"k": "plan", "round": round_no, "steps": plan_steps}
                    for step in plan_steps:
                        if step["mode"] == "runnable":
                            yield {"k": "status", "egg": step["egg_id"], "status": "planned"}

                    started_ids = []
                    started = self._start_runs(
                        [s for s in plan_steps if s["mode"] == "runnable"],
                        boxes, channels, inbox, events, running, bg,
                        started_ids=started_ids, goal_context=text,
                        # List EVERY egg this coordination round touches (not just
                        # this batch) so even a later deferred batch still receives
                        # all peers' deliverable directories to reference.
                        extra_peers=[s["egg_id"] for s in plan_steps],
                    )
                    # Only remove a pending task once it has actually started.
                    if started_ids:
                        postponed = [p for p in postponed
                                     if p.get("egg_id") not in set(started_ids)]
                    if started == 0:
                        if running:
                            idle_rounds = 0
                        else:
                            idle_rounds += 1
                            cmd_hist.append({"role": "user", "text":
                                             f"（第 {round_no} 轮没有可执行的子任务。请继续补齐依赖；"
                                             "若确实无法完成，请明确报告阻塞原因，不要伪称成功。）"})
                            if idle_rounds >= _MAX_IDLE:
                                final_summary = (
                                    decision.get("summary") or
                                    "自动调度因连续多轮没有可执行任务而停止；请检查待办依赖。"
                                )
                                break
                            await asyncio.sleep(_MONITOR_INTERVAL)
                    else:
                        idle_rounds = 0

                    notified.clear()
                    watch_base = {key: value["len"] for key, value in watch.items()}
                    continue

                event = await self._next_event(events, _MONITOR_INTERVAL)
                if event is not None:
                    yield event
        finally:
            # Whatever ends this loop, nothing may keep talking to the LLM:
            # cancel the remaining jobs AND interrupt the eggs that are still
            # busy (the SDK sends a halt_request, so the SERVER-side turn really
            # stops instead of burning tokens in the background).
            for eid, record in list(running.items()):
                for key in ("pump", "consumer"):
                    task = record.get(key)
                    if task is not None and not task.done():
                        task.cancel()
                egg = self.eggs.get(eid)
                if egg is not None and getattr(egg, "busy", False):
                    try:
                        egg.interrupt()
                    except Exception:  # pragma: no cover - defensive
                        pass
            for task in list(bg):
                if not task.done():
                    task.cancel()
            if bg:
                await asyncio.gather(*bg, return_exceptions=True)
            self.autosave(force=True)

        try:
            if final_summary:
                self.commander._record("assistant", "text", text=final_summary)
                yield {"k": "say_break"}
                yield {"k": "say", "text": final_summary}
            else:
                results_text = "\n\n".join(
                    f"[{title}] 指令：{message}\n结果：{reply[:600]}"
                    for title, message, reply in all_results
                )
                recap_prompt = (
                    f"用户指令：{text}\n\n各蛋执行情况：\n{results_text}\n\n"
                    "请用 2-4 句中文总结实际完成情况、未完成项及后续建议；不要把失败任务描述为成功。"
                )
                async for token in self.commander.run(recap_prompt, summary, role="system"):
                    yield {"k": "say", "text": token}
        except Exception as ex:  # pragma: no cover
            yield {"k": "say", "text": f"（总结失败：{ex}）", "role": "system"}
        yield {"k": "end"}

    # ---- commander monitoring helpers ----
    async def _next_event(self, events, timeout):
        """Wait for the next UI event; None on timeout (a monitor tick)."""
        try:
            return await asyncio.wait_for(events.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def _watch_eggs(self, targets, base=None, inbox=None):
        """Snapshot every monitored egg for the commander's monitor panel.

        Per egg:
          running  -> is a model turn in flight right now?
          len      -> current conversation length (characters)
          delta    -> how much the conversation GREW since the last decision
                      (会话有没有变长)
          silent   -> seconds since the conversation last grew
          elapsed  -> seconds since the current run started
          stuck    -> running but silent for too long
          overdue  -> running for too long
          pending  -> completion notifications waiting for this egg
        """
        now = time.monotonic()
        base = base or {}
        inbox = inbox or {}
        watch = {}
        # Watch the selected eggs AND any egg that has a pending completion
        # notification (so a notification never goes unnoticed).
        watched = list(targets) + [e for e in inbox if e not in targets]
        for t in watched:
            e = self.eggs.get(t)
            if not e:
                continue
            cur = e.transcript_len
            prev = base.get(t, cur)
            silent = max(0.0, now - (e.last_activity or now))
            running = bool(e.busy)
            elapsed = 0.0
            if running and e.run_started_at:
                elapsed = max(0.0, now - e.run_started_at)
            watch[t] = {
                "id": t,
                "title": e.title,
                "running": running,
                "len": cur,
                "delta": cur - prev,
                "silent": round(silent, 1),
                "elapsed": round(elapsed, 1),
                "stuck": bool(running and silent > _EGG_SILENT),
                "overdue": bool(running and elapsed > _EGG_OVERDUE),
                "pending": len(inbox.get(t) or []),
                "completed": bool(e.last_completion),
                "delivered": ((self._last_delivery.get(t) or {}).get("delivered")),
                "delivery": ((self._last_delivery.get(t) or {}).get("detail")),
            }
        return watch

    def _watch_alerts(self, watch, reported, cmd_hist):
        """Turn stuck / overdue eggs into alerts for the commander.

        Each alert is recorded once (``reported``) so the commander is not
        spammed with the same warning every monitor tick.
        """
        alerts = []
        for eid, st in watch.items():
            if not st["running"]:
                continue
            kind = "overdue" if st["overdue"] else ("stuck" if st["stuck"] else None)
            if not kind:
                continue
            key = (eid, kind)
            if key in reported:
                continue
            reported.add(key)
            what = "运行超时" if kind == "overdue" else "长时间没有新输出"
            msg = (f"【监控告警】蛋「{st['title']}」{what}"
                   f"（已运行 {st['elapsed']}s，静默 {st['silent']}s，"
                   f"会话长度 {st['len']} 字）。")
            cmd_hist.append({"role": "user", "text": msg})
            alerts.append(msg)
        return alerts

    def _fmt_watch_line(self, watch):
        """One-line monitor summary (shown in the commander's narration)."""
        if not watch:
            return ""
        parts = []
        for eid, st in watch.items():
            if st["running"]:
                growth = f"+{st['delta']}字" if st["delta"] > 0 else "无新增"
                parts.append(f"{eid} 运行中(会话{st['len']}字，"
                             f"本轮{growth}，静默{st['silent']}s)")
            else:
                pend = f"，待处理通知{st['pending']}条" if st["pending"] else ""
                mark = ""
                if st.get("delivered") is True:
                    mark = "，已交付"
                elif st.get("delivered") is False:
                    mark = "，【未交付】"
                parts.append(f"{eid} 空闲(会话{st['len']}字{pend}{mark})")
        return " / ".join(parts)

    def _fmt_watch(self, watch, notified=None):
        """Render the monitor panel for the commander's decision prompt."""
        if not watch:
            return "（没有可监控的子蛋）"
        lines = []
        for eid, st in watch.items():
            if st["delta"] > 0:
                growth = f"正在变长（+{st['delta']} 字）"
            elif st["delta"] == 0:
                growth = "没有变长"
            else:
                growth = f"变短（{st['delta']} 字）"
            state = "运行中" if st["running"] else "空闲"
            extra = ""
            if st["elapsed"]:
                extra += f"，已运行 {st['elapsed']}s"
            if st["silent"] > 5:
                extra += f"，静默 {st['silent']}s"
            if st["pending"]:
                extra += f"，有 {st['pending']} 条待你处理的完成通知"
            if st["overdue"]:
                extra += "，【运行超时】"
            elif st["stuck"]:
                extra += "，【长时间无输出】"
            if st.get("delivered") is True:
                extra += f"，【已交付】{st.get('delivery') or ''}"
            elif st.get("delivered") is False:
                extra += f"，【未交付】{st.get('delivery') or ''}"
            lines.append(f"- id={eid} 「{st['title']}」{state} | 会话长度 "
                         f"{st['len']} 字，{growth}{extra}")
        if notified:
            lines.append("【待处理的完成通知（子蛋已完成并通知了你）】")
            for n in notified[-3:]:
                dv = n.get("delivery")
                lines.append(f"- 蛋「{n.get('title')}」的结论："
                             f"{(n.get('reply') or '')[:300]}"
                             + (f"｜{dv}" if dv else ""))
        return "\n".join(lines)

    def _resolve_egg_reference(self, value, allowed_ids=None):
        """Resolve a model-provided egg reference as either a real id or a unique title.

        The model schema asks for ids, but local models sometimes emit titles in
        deliver/notify/depends_on.  Resolve exact, unambiguous titles instead of
        silently dropping them.  Partial/fuzzy title matching is deliberately not
        used because two eggs may have similar names.
        """
        if value is None:
            return None
        raw = str(value).strip().strip("`\"'“”‘’")
        candidates = [raw]
        # Common harmless wrappers seen in model output.
        for prefix, suffix in (("蛋「", "」"), ("蛋『", "』"), ("「", "」"), ("『", "』")):
            if raw.startswith(prefix) and raw.endswith(suffix):
                candidates.append(raw[len(prefix):-len(suffix)].strip())
        if raw.lower().startswith("id="):
            candidates.append(raw.split("=", 1)[1].strip())

        allowed = set(allowed_ids) if allowed_ids is not None else set(self.eggs)
        for candidate in candidates:
            if candidate in self.eggs and candidate in allowed:
                return candidate

        title_matches = []
        for candidate in candidates:
            matches = [eid for eid, egg in self.eggs.items()
                       if eid in allowed and str(getattr(egg, "title", "")).strip() == candidate]
            title_matches.extend(matches)
        title_matches = list(dict.fromkeys(title_matches))
        if len(title_matches) == 1:
            return title_matches[0]
        if len(title_matches) > 1:
            print(f"[commander] ambiguous egg title reference ignored: {value!r}")
        return None

    def _normalize_egg_refs(self, values, allowed_ids=None, field="reference"):
        """Normalize a single reference or list of references into unique egg ids."""
        if values is None:
            return []
        if isinstance(values, (str, int)):
            values = [values]
        if not isinstance(values, (list, tuple, set)):
            print(f"[commander] invalid {field} value ignored: {values!r}")
            return []
        out = []
        for value in values:
            eid = self._resolve_egg_reference(value, allowed_ids)
            if eid is None:
                print(f"[commander] unknown/ambiguous {field} reference ignored: {value!r}")
                continue
            if eid not in out:
                out.append(eid)
        return out

    def _dedupe_commander_tasks(self, tasks):
        """Keep one task per egg while retaining the most specific instruction and metadata."""
        ordered = []
        by_id = {}
        for incoming in tasks:
            task = dict(incoming)
            eid = task.get("egg_id")
            if eid not in by_id:
                by_id[eid] = task
                ordered.append(task)
                continue
            current = by_id[eid]
            if current.get("message") != task.get("message"):
                print(f"[commander] duplicate tasks for {eid}; keeping the more detailed instruction")
                if len(str(task.get("message") or "")) > len(str(current.get("message") or "")):
                    current["message"] = task.get("message")
            for field in ("deliver", "notify", "depends_on", "_dependency_errors"):
                merged = list(current.get(field) or [])
                for value in (task.get(field) or []):
                    if value not in merged:
                        merged.append(value)
                current[field] = merged
        return ordered

    def _normalize_task_refs(self, task, field, allowed_ids=None):
        """Normalize ids or exact, unique titles for task reference fields.

        Invalid explicit dependencies are preserved as diagnostics so they cannot be
        silently dropped and accidentally make a task runnable.
        """
        raw = task.get(field) or []
        if isinstance(raw, (str, int)):
            values = [raw]
        elif isinstance(raw, (list, tuple, set)):
            values = list(raw)
        else:
            values = []
        allowed = set(allowed_ids) if allowed_ids is not None else set(self.eggs)
        out, invalid = [], []
        for value in values:
            ref = self._resolve_egg_reference(value, allowed)
            if ref is None:
                invalid.append(str(value))
            elif ref not in out:
                out.append(ref)
        if field == "depends_on":
            previous = list(task.get("_dependency_errors") or [])
            task["_dependency_errors"] = list(dict.fromkeys(previous + invalid))
            if task["_dependency_errors"]:
                print(f"[commander] invalid explicit dependencies for {task.get('egg_id')}: "
                      f"{task['_dependency_errors']!r}")
        return out

    def _find_dependency_cycle(self, tasks):
        """Return node ids participating in a cycle of explicit pending dependencies."""
        graph = {}
        for task in tasks:
            eid = task.get("egg_id")
            if eid:
                graph.setdefault(eid, set()).update(task.get("depends_on") or [])
        color = {node: 0 for node in graph}
        stack, cycle = [], set()

        def visit(node):
            color[node] = 1
            stack.append(node)
            for dep in graph.get(node, ()):
                if dep not in graph:
                    continue
                if color[dep] == 0:
                    visit(dep)
                elif color[dep] == 1:
                    try:
                        cycle.update(stack[stack.index(dep):])
                    except ValueError:
                        cycle.add(dep)
            stack.pop()
            color[node] = 2

        for node in graph:
            if color[node] == 0:
                visit(node)
        return sorted(cycle)

    def _resolve_dependencies(self, tasks, boxes=None, eligible_ids=None):
        """Normalize ONLY explicitly declared dependencies; never infer graph edges.

        ``depends_on`` controls execution order. ``deliver`` controls authorized input.
        A valid dependency on an unselected egg is retained as a blocker rather than
        silently removed. Role names, prompts, titles, and canvas edges create no
        implicit execution dependencies.
        """
        boxes = boxes if boxes is not None else {}
        all_ids = set(self.eggs)
        deps = {}
        for task in tasks:
            eid = task.get("egg_id")
            if not eid:
                continue
            task_deps = self._normalize_task_refs(task, "depends_on", all_ids)
            authorized = set(self._normalize_task_refs(task, "deliver", all_ids))
            deps[eid] = set(task_deps)
            # An explicit dependency authorizes reading its conclusion once available.
            # Do not erase the declared deliver list while the source is still running.
            authorized.update(dep for dep in task_deps if dep in boxes and dep != eid)
            task["deliver"] = sorted(authorized)
            task["depends_on"] = sorted(set(task_deps))
        return deps

    def _split_by_dependency(self, tasks, boxes=None, eligible_ids=None, running_ids=None):
        """Split only by explicit dependencies; cycles and missing refs remain blocked."""
        boxes = boxes if boxes is not None else {}
        running_ids = set(running_ids or ())
        self._resolve_dependencies(tasks, boxes, eligible_ids=eligible_ids)
        ids = {task.get("egg_id") for task in tasks if task.get("egg_id")}
        runnable, deferred = [], []
        for task in tasks:
            eid = task.get("egg_id")
            unresolved = [f"无效依赖引用:{err}" for err in (task.get("_dependency_errors") or [])]
            for dep in task.get("depends_on") or []:
                # A self-dependency is a cycle and can never be treated as satisfied.
                if dep == eid:
                    unresolved.append(dep)
                # A producer in this batch must finish this run before its consumer.
                elif dep in ids or dep in running_ids or dep not in boxes:
                    unresolved.append(dep)
            if unresolved:
                task["_pending_reason"] = "等待/校验依赖：" + ", ".join(dict.fromkeys(unresolved))
                deferred.append(task)
            else:
                runnable.append(task)
        return runnable, deferred

    # ---- background dispatch + completion notification ----
    def _start_runs(self, plan_steps, boxes, channels, inbox, events, running, bg,
                    started_ids=None, goal_context=None, extra_peers=None):
        """Start tasks that are not already running; record inbox items for ack-on-success."""
        started = 0
        started_ids = started_ids if started_ids is not None else []
        scheduled_ids = {s.get("egg_id") for s in plan_steps}
        for step in plan_steps:
            eid = step.get("egg_id")
            egg = self.eggs.get(eid)
            if not egg or eid in running or bool(getattr(egg, "busy", False)):
                continue
            # Keep runtime-only acknowledgement metadata off the UI plan object.
            run_step = dict(step)
            run_step["goal_context"] = run_step.get("goal_context") or goal_context or ""
            run_step["_consumed_inbox"] = list(inbox.get(eid, []) or [])
            # The delivery contract is explicit: the egg must know exactly which
            # artifact to produce and how it will be verified. When the commander
            # did not state it, the backend appends the standard contract.
            message = str(run_step.get("message") or "")
            if ("交付标准" not in message
                    and not self._has_named_deliverable(message)):
                message = message + self._delivery_standard(egg, message)
            # Coordination context: tell this egg where its peers' deliverables
            # live so it can read them. The judge/finalizer especially need to
            # consult the other participants' actual output (not just their own
            # workspace), which is impossible without knowing the peer directories.
            peer_ids = scheduled_ids | set(extra_peers or ())
            if len(peer_ids) > 1:
                block = self._peer_directories_block(peer_ids)
                if block:
                    message = message + block
            run_step["message"] = message
            events.put_nowait({"k": "instruct", "egg": eid,
                               "text": "🎯 总指挥指令：" + message})
            events.put_nowait({"k": "status", "egg": eid, "status": "running"})
            active_sources = set(running) | scheduled_ids
            ctx = self._task_context(run_step, boxes, channels, inbox,
                                     active_ids=active_sources)
            q = asyncio.Queue()
            pump = asyncio.create_task(_pump(egg.run(message, context=ctx), q))
            rec = {"step": run_step, "tokens": [], "done": False, "reply": "",
                   "failed": False, "pump": pump, "egg": eid,
                   "started_at": time.time()}
            running[eid] = rec
            consumer = asyncio.create_task(self._consume_run(eid, q, rec, events))
            rec["consumer"] = consumer
            bg.extend([pump, consumer])
            # Keep the last successful result available while a rerun is in flight.
            # Explicit depends_on checks also require the dependency not to be running,
            # so stale results cannot satisfy a dependency on the current run.
            started_ids.append(eid)
            started += 1
        return started

    async def _consume_run(self, eid, q, rec, events):
        """Collect one run's stream and preserve failures as failures, not conclusions."""
        try:
            while True:
                item = await q.get()
                if item is _PUMP_DONE:
                    rec["reply"] = "".join(str(x) for x in rec["tokens"])
                    rec["done"] = True
                    return
                if isinstance(item, tuple) and item and item[0] == "__err__":
                    rec["failed"] = True
                    txt = f"\n（执行出错：{item[1]}）"
                    rec["tokens"].append(txt)
                    await events.put({"k": "egg", "egg": eid, "text": txt})
                    continue
                rec["tokens"].append(item)
                await events.put({"k": "egg", "egg": eid, "text": item})
        except asyncio.CancelledError:
            # Cancellation must also be a terminal state. The debate phase loop
            # waits for rec["done"], so leaving it False here can deadlock forever.
            rec["failed"] = True
            rec["cancelled"] = True
            rec["error"] = "run consumer cancelled"
            rec["reply"] = "".join(str(x) for x in rec.get("tokens", []))
            rec["done"] = True
            pump = rec.get("pump")
            if pump is not None and not pump.done():
                pump.cancel()
            try:
                events.put_nowait({
                    "k": "egg", "egg": eid,
                    "text": "\n（任务流被取消；该任务不会计为成功）",
                })
            except Exception:
                pass
            raise
        except Exception as ex:  # pragma: no cover
            rec["failed"] = True
            rec["error"] = str(ex)
            rec["reply"] = "".join(str(x) for x in rec["tokens"])
            rec["done"] = True
            await events.put({"k": "egg", "egg": eid,
                              "text": f"\n（执行出错：{ex}）"})

    def _file_result(self, rec, boxes, channels, cmd_hist, inbox, notified,
                     all_results):
        """Record a run, acknowledge consumed notifications only on success, and notify listeners.

        Everything that leaves this run (the conclusion box, the channels, the
        commander's history, the completion notices) carries ONLY the
        assistant's own text: the 💭 thinking, the 🔧 tool calls / ↳ results
        and the ⚠ warnings stay local.
        """
        step = rec["step"]
        eid = step["egg_id"]
        raw_reply = rec.get("reply") or ""
        egg = self.eggs.get(eid)
        if not egg:
            return

        failed = bool(rec.get("failed"))
        # What may be passed on: the assistant's text only. A failed run has no
        # conclusion — pass the REASON, never the thinking / tool noise that was
        # streamed before it broke.
        reply = self._assistant_only(egg, rec)
        if not reply and failed:
            reply = str(rec.get("error") or "（本轮失败，没有产出结论）")
        all_results.append((egg.title, step.get("message", ""), reply))
        # Objective delivery check: did this egg really hand over its artifact?
        delivery = self._verify_delivery(egg, step.get("message"),
                                         rec.get("started_at"))
        rec["delivery"] = delivery
        self._last_delivery[eid] = delivery
        if not failed:
            boxes[eid] = {"result": reply}
            # A successful run acknowledges only the inbox prefix it actually saw.
            consumed = list(step.get("_consumed_inbox") or [])
            current = list(inbox.get(eid, []) or [])
            if consumed and current[:len(consumed)] == consumed:
                current = current[len(consumed):]
            elif consumed:
                # Defensive fallback when another operation reordered the inbox.
                remaining = list(current)
                for msg in consumed:
                    try:
                        remaining.remove(msg)
                    except ValueError:
                        pass
                current = remaining
            if current:
                inbox[eid] = current
            else:
                inbox.pop(eid, None)
        status_text = "执行失败" if failed else "完成"
        task_id = str(step.get("task_id") or f"run-{time.time_ns()}-{eid}")
        self._task_results[task_id] = {
            "task_id": task_id,
            "egg_id": eid,
            "title": egg.title,
            "phase": str(step.get("phase") or "commander"),
            "message": str(step.get("message") or ""),
            "status": "failed" if failed else "completed",
            "result": reply,
            "error": str(rec.get("error") or ""),
            "delivered": bool(delivery.get("delivered")),
            "deliverable": delivery.get("path"),
            "delivery_detail": delivery.get("detail"),
            "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        ch = channels.setdefault(("commander", eid), [])
        ch.append({"role": "user", "text": step.get("message", "")})
        ch.append({"role": "assistant", "text": reply})

        # Peer history must contain the source's actual conclusion, not the consumer's reply
        # mislabeled as if it had come from the source.
        for src in (step.get("deliver") or []):
            source_box = boxes.get(src)
            if not source_box or src == eid:
                continue
            source_title = self.eggs[src].title if src in self.eggs else src
            pch = channels.setdefault((src, eid), [])
            pch.append({"role": "user",
                        "text": f"来自蛋「{source_title}」的结论：\n{source_box.get('result', '')}"})
            pch.append({"role": "assistant",
                        "text": f"当前任务执行{status_text}：\n{reply}"})

        cmd_hist.append({
            "role": "user",
            "text": (f"【子蛋「{egg.title}」{status_text}】\n"
                     f"结论/错误：{reply[:2000]}\n"
                     f"交付核验：{delivery.get('detail')}"
                     + ("" if delivery.get("delivered")
                        else " —— 该结论尚未落盘，若它是本轮必需的成果，"
                             "请重新派活并明确要求写盘交付。")),
        })
        notified.append({"egg": eid, "title": egg.title, "reply": reply[:2000],
                         "failed": failed,
                         "delivery": delivery.get("detail"),
                         "delivered": bool(delivery.get("delivered"))})

        # Notifications persist across commander_run calls and session restarts.
        for target in (step.get("notify") or []):
            if target not in self.eggs or target == eid:
                continue
            prefix = "执行失败" if failed else "已完成"
            message = (f"【完成通知：蛋「{egg.title}」{prefix}并通知你】\n"
                       f"{reply[:2000]}")
            inbox.setdefault(target, []).append(message)
            notify_channel = channels.setdefault((eid, target), [])
            notify_channel.append({"role": "user",
                                   "text": f"来自蛋「{egg.title}」的完成通知"})
            notify_channel.append({"role": "assistant", "text": reply})
        self.autosave(force=True)

    async def _commander_decide(self, text, summary, targets, boxes, channels,
                                cmd_hist, round_no, watch=None, notified=None,
                                sink=None, deferred=None):
        """Ask the commander LLM for the NEXT round only (reactive adjudication).

        The commander sees:
          * the original instruction
          * the eggs it may drive (id / title / role)
          * the boxes collected so far (each egg's latest conclusion)
          * the MONITOR panel (``watch``): per egg — running?, conversation
            length and how much it grew since the last decision, silence
            duration, elapsed runtime, stuck/overdue flags, pending
            notifications; plus the completion notifications it received
            (``notified``)
          * the user+assistant conversation history of its channels
        and answers with either a set of tasks to run NOW (1 = serial, N =
        parallel) or ``action="finish"`` when the instruction is complete.

        Each task may name ``deliver`` — the boxes the commander AUTHORIZES
        this egg to read. That is the only channel through which one egg's
        message reaches another egg; anything not delivered stays with the
        commander for arbitration.

        Each task may name ``notify`` — the eggs that must be told WHEN this
        egg finishes (the commander is always told).

        ``sink`` is an optional callback that receives the planner's OWN
        output (its text / thinking / tool calls) so the commander panel can
        show what the commander actually said — not just the synthesized plan.
        """
        if not SDK_OK:
            return self._fake_commander_decide(text, targets, boxes, round_no)
        try:
            target_desc = "\n".join(
                f"- id={self.eggs[t].id} 标题={self.eggs[t].title} "
                f"职责={self.eggs[t].prompt}"
                for t in targets
            )
            boxes_desc = "\n".join(
                f"- 来自蛋「{self.eggs[b].title if b in self.eggs else b}」的结论：\n"
                f"{v.get('result', '')[:800]}"
                for b, v in boxes.items()
            ) or "（目前还没有任何蛋的结论）"
            chan_desc = "\n".join(
                f"### {self._chan_label(s)} → {self._chan_label(d)}\n"
                f"{self._fmt_channel(ch)}"
                for (s, d), ch in channels.items()
            ) or "（暂无通讯记录）"
            if len(chan_desc) > 4000:
                chan_desc = "…（更早的通讯记录略去）\n" + chan_desc[-4000:]
            watch_desc = self._fmt_watch(watch or {}, notified or [])
            sys_instr = types.CustomSystemInstructions(text=(
                "你是总指挥（Commander），以【持续循环 + 实时监控】方式驱动若干子蛋"
                "完成用户指令。\n"
                "重要机制：你【不预先规划完整流程】，而是【循环】工作。每一轮你都会收到"
                "【子蛋实时监控】面板（会话长度/增量、静默时长、已运行时长、超时与静默"
                "告警、待处理的完成通知），你据此决定【下一步】。\n"
                "规则：\n"
                "0) 【先说后决】先用 1-3 句中文说明你此刻的判断与下一步计划"
                "（这段话会直接显示给用户看），然后再输出结构化 decision。"
                "说明不超过 3 句，不要写长篇大论，也不要重复已有结论。\n"
                "1) 你是一个【循环】：只要用户指令还没真正完成，就继续给下一步指令；"
                "只有当所有子蛋都交付了结论、指令确实完成时，才 action='finish' 并在 "
                "summary 里给出最终总结。不要因为某轮没有现成任务就提前结束。"
                "特别注意：子蛋还没有产出结论，正说明现在应该【派活】而不是收尾；"
                "只有在你亲自确认了所有子蛋的结论都已在【已有的结论盒子】里、"
                "且用户指令确实达成了，才可以 finish。\n"
                "2) 每轮先读【子蛋实时监控】：\n"
                "   · 会话长度/增量：增量>0 说明该蛋还在推进；增量=0 且静默很久说明它"
                "卡住了 —— 你可以重新给它派活、换一只蛋接手，或直接收尾。\n"
                "   · 【待处理的完成通知】：子蛋跑完后会主动通知你（你总会收到）。\n"
                "   · 【监控告警】：运行超时/长时间无输出的蛋需要你裁决。\n"
                "3) 一次只给本轮的指令。若本轮这些蛋互不依赖，可同时给多个（并行）；"
                "若需要等某个结果才能定下一步，就只给一个（串行）。\n"
                "3c) 如果【上一轮被推迟的任务】里还有蛋在等待，说明还有活没干完"
                "（例如裁判还没裁决）——必须先派活给它，不能直接 finish。\n"
                "3a) 同一只蛋在一轮里只给一个任务；已经产出结论的蛋，除非指令明确要求"
                "它继续推进，否则不要重复派它。\n"
                "3b) 【时序/依赖】凡是必须等待另一个蛋本轮产出后才能执行的任务，必须在"
                "depends_on 中填写来源蛋的真实 id。系统只把 depends_on 当作明确的等待关系；"
                "不要因为指令里提到了正方/反方、使用了‘回应’或‘复核’就推断等待关系。"
                "例如辩论：正方与反方立论互不依赖，同轮并行；主持人依赖双方时写"
                "depends_on=[\"e2\", \"e4\"]；总结陈词还应等待主持人交锋结束。"
                "不要依赖基于角色名称、节点标题、画布连线或自然语言动词的自动推断；"
                "调度器只执行你显式写入 depends_on 的等待关系。对先立论、再主持交锋、"
                "最后总结的多阶段任务，你必须自己写清依赖：主持人 depends_on 包含双方立论 id，"
                "最终总结 depends_on 包含主持人 id。\n"
                "3d) 【交付标准】每个任务的 message 都必须写清交付标准，四项缺一不可：\n"
                "   · 交付物：一份 Markdown 文件，写在该蛋【自己的工作目录根目录】下——"
                "就是用户在「审阅代码」里看到的那个目录；路径统一用 <角色名>.md"
                "（例如 正方一辩(支持方).md）；禁止写进 agent_data/（SDK 内部目录，"
                "用户在「审阅代码」里看不到）；\n"
                "   · 内容：只写本节点负责的结论，不复述其它蛋的内容；\n"
                "   · 验收：写完后读回该文件，确认已写入且非空；\n"
                "   · 通知：在回复里写明交付物路径，并在 notify 里写清要通知的蛋。\n"
                "只在聊天里说“已完成”而不落盘，一律视为【未交付】：系统会核验文件、"
                "在监控面板里标出【未交付】，你也必须据此重新派活并要求写盘。\n"
                "3e) depends_on 写在每个任务里；若整轮任务都等待同一批上游，"
                "也可以把 depends_on 写在 decision 顶层（会套用到本轮每个任务）。\n"
                "4) 【授权与等待分开】deliver 只表示允许读取某只蛋已经生成的结论，不表示"
                "必须等待它；只有需要它本轮的新结果时，才同时写 depends_on。来源结论尚未"
                "产生时，deliver 授权应保留到任务实际执行，不可提前丢弃。\n"
                "5) 子蛋之间【默认不能互相通信】。只有你（总指挥）有权让某个结论从一只蛋"
                "传给另一只蛋：在 deliver 中填写来源蛋 id（或无歧义的完整标题，系统会规范化为 id）。"
                "未授权的结论只留在你这里由你裁决。\n"
                "6) notify 指定完成后要通知的其它蛋；可以填真实 id 或无歧义的完整标题。\n"
                "7) 只读工具（view_file / list_directory / search_directory / find_file）"
                "【只能】在下面列出的目录里查看。【不要】去翻其它目录，也不要反复探索"
                "目录结构；一次最多调用 2 次工具，除非确实需要核实子蛋产出。\n"
                "8) 通讯只传【assistant 的正文结论】：子蛋的 💭思考、🔧工具调用与结果"
                "不会传递给你，你也不需要它们——你看到的就是各蛋的实际结论。\n"
                "9) 任务完成时 action='finish'，并在 summary 里给出最终总结；"
                "否则 action='dispatch' 给出本轮 tasks。\n"
                "返回结构化 decision（action + "
                "tasks[egg_id, message, deliver, notify, depends_on] + "
                "summary + note）。note 用一句中文说明本轮意图。"
            ))
            caps = types.CapabilitiesConfig(
                # The commander may INSPECT what the eggs produced (read-only
                # tools) before deciding the next round — and the tool calls
                # then show up in the 总指挥's 轨迹, like the eggs'.
                enabled_tools=[
                    types.BuiltinTools.VIEW_FILE,
                    types.BuiltinTools.LIST_DIR,
                    types.BuiltinTools.SEARCH_DIR,
                    types.BuiltinTools.FIND_FILE,
                ],
                enable_subagents=False,
                agent_behavior=types.AgentBehavior.MINIMAL,
                # Tool output MUST stay small: the planner once read an 800-line
                # transcript and blew the agent's context window, which killed
                # the whole round (the decision silently fell back).
                tool_output_truncation_config=types.ToolOutputTruncationConfig(
                    max_tokens=2000),
            )
            # The commander may READ what the eggs produced — but only inside
            # the eggs' own worktrees. Without this the read-only tools could
            # pull in a huge file (e.g. a session log) and blow the agent's
            # context window, killing the whole round.
            workspaces = []
            for t in targets:
                wt = getattr(self.eggs.get(t), "worktree", None)
                if wt:
                    workspaces.append(str(wt))
            ws_desc = "\n".join(f"- {w}" for w in workspaces) or "（本轮没有可检视的目录）"
            deferred_desc = ""
            if deferred:
                lines = []
                for item in deferred:
                    deps = item.get("depends_on") or []
                    missing = [dep for dep in deps if dep not in boxes]
                    status = (f"尚缺结论：{', '.join(missing)}；请先派发缺失的上游蛋。"
                              if missing else "依赖已满足；本轮必须继续派发该待办任务。")
                    if item.get("_pending_reason"):
                        status += f" 状态：{item['_pending_reason']}"
                    lines.append(
                        f"- 蛋「{item.get('title')}」(id={item.get('egg_id')})："
                        f"任务={item.get('message', '')}；依赖={deps}；{status}"
                    )
                deferred_desc = "【尚未完成的待办任务】\n" + "\n".join(lines) + "\n\n"
            history_desc = "\n".join(
                f"{entry.get('role', 'user')}：{str(entry.get('text', ''))[:700]}"
                for entry in (cmd_hist or [])[-10:]
            )
            if len(history_desc) > 5000:
                history_desc = history_desc[-5000:]
            prompt = (
                f"总指挥近几轮历史（包括调度反馈、空转提醒和完成通知）：\n"
                f"{history_desc or '（暂无）'}\n\n"
                f"可选子蛋：\n{target_desc}\n\n"
                f"{deferred_desc}"
                f"已有的结论盒子：\n{boxes_desc}\n\n"
                f"子蛋实时监控：\n{watch_desc}\n\n"
                f"通讯历史（user+assistant）：\n{chan_desc}\n\n"
                f"画布状态：\n{summary}\n\n"
                f"可检视的目录（只读，仅限这些）：\n{ws_desc}\n\n"
                f"用户指令：{text}\n\n"
                f"这是第 {round_no} 轮。请只决定【本轮】的动作。"
            )
            # The planner agent is created fresh每 round, so a corrupt history is
            # unlikely — but the model can still return an EMPTY turn, which the
            # server rejects with HTTP 400 "Assistant message must contain either
            # 'content' or 'tool_calls'!". Retry once with a brand-new agent
            # before giving up on the round.
            data = None
            fallback_texts = []
            format_hint = ""
            # Three escalating attempts:
            #   1) structured output (the `finish` tool carries the JSON);
            #   2) structured output + an explicit format reminder;
            #   3) plain mode — the model just writes the JSON as text, which
            #      _parse_decision_text() can recover.
            # A local model that answers with PROSE instead of a decision must
            # never cost the user the whole round.
            for attempt in (1, 2, 3):
                try:
                    async with Agent(build_agent_config(
                        system_instructions=sys_instr, capabilities=caps,
                        response_schema=(CommanderDecision if attempt < 3 else None),
                        workspaces=(workspaces or None),
                    )) as planner:
                        resp = await planner.chat(prompt + format_hint)
                        # Consume the FULL semantic stream (resp.chunks) so the
                        # commander's 💭 thinking and 🔧 tool calls are recorded
                        # into its 轨迹 — `async for chunk in resp` would filter
                        # them out. structured_output() then reads the parsed
                        # decision from the conversation.
                        #
                        # The planner answers in STRUCTURED-OUTPUT mode, i.e. it
                        # calls the special `finish` tool whose args carry the
                        # raw JSON. Small local models sometimes fail to make
                        # the SDK parse that envelope, and structured_output()
                        # then returns None — so we keep the raw JSON as a
                        # fallback and parse it ourselves.
                        raw_json = None
                        text_chunks = []
                        async for chunk in resp.chunks:
                            self._record_planner_chunk(chunk)
                            if sink is not None:
                                shown = self._planner_chunk_text(chunk)
                                if shown:
                                    sink(shown)
                            # A decision may arrive as a Text chunk, as a bare
                            # string chunk (the local harness emits both), OR as a
                            # Thought chunk — local models often hide the JSON
                            # decision inside their reasoning instead of emitting a
                            # `finish` tool call or an answer. Collect ALL of them,
                            # or the JSON fallback silently loses the whole round
                            # ("决策输出无法解析或不可用"). The `finish` envelope is
                            # captured separately via raw_json below; a plain
                            # ToolCall/ToolResult is never a decision, so skip it.
                            if isinstance(chunk, types.Text):
                                chunk_text = getattr(chunk, "text", None)
                                if chunk_text:
                                    text_chunks.append(str(chunk_text))
                            elif isinstance(chunk, types.Thought):
                                chunk_text = getattr(chunk, "text", None)
                                if chunk_text:
                                    text_chunks.append(str(chunk_text))
                            elif isinstance(chunk, str):
                                text_chunks.append(str(chunk))
                            if isinstance(chunk, types.ToolCall):
                                nm = getattr(chunk, "name", None)
                                if nm == "finish" and raw_json is None:
                                    raw_json = self._finish_payload(chunk)
                        data = await resp.structured_output()
                        if isinstance(data, str):
                            # The harness may hand the decision back as RAW JSON
                            # text instead of a parsed object.
                            parsed = self._parse_decision_text(data)
                            data = parsed if parsed is not None else None
                        if data is None and raw_json:
                            print("[commander-decide] structured_output empty; parsing raw finish payload")
                            data = self._parse_decision_text(raw_json)
                        if data is None and text_chunks:
                            # Some local models emit JSON as text instead of invoking the
                            # structured-output tool. Recover it if a JSON object exists.
                            data = self._parse_decision_text("\n".join(text_chunks))
                        # Keep the raw carriers as a last-resort source: if the
                        # structured output turns out to be empty/unusable, the
                        # decision can still be recovered from them.
                        fallback_texts = [t for t in (raw_json,
                                                      "\n".join(text_chunks)) if t]
                        if data is None and attempt < 3:
                            # The model answered with prose instead of a decision.
                            # Ask it again, more strictly, before giving up.
                            print("[commander-decide] no structured decision; "
                                  f"retrying (attempt {attempt + 1})")
                            format_hint = (
                                "\n\n【格式纠正】你上一轮的输出无法解析为结构化 decision。"
                                "本轮请只输出一个 JSON 对象，不要写任何解释文字："
                                '{"action":"dispatch","tasks":[{"egg_id":"...",'
                                '"message":"给该蛋的完整指令（含交付物路径与验收方式）",'
                                '"deliver":[],"notify":[],"depends_on":[]}],'
                                '"summary":"本轮意图","note":"一句话说明"}'
                                "\n只有在所有子蛋都已交付结论时才能 action='finish'；"
                                "还有蛋没干活时必须 action='dispatch' 并派活。"
                            )
                            continue
                    break
                except asyncio.CancelledError:
                    raise
                except Exception as ex:
                    if attempt == 1 and is_corrupt_conversation_error(ex):
                        print(f"[commander-decide] empty/corrupt turn, "
                              f"retrying with a fresh agent: {ex}")
                        continue
                    raise
            if data is None:
                raise RuntimeError("commander decision produced no output")
            if isinstance(data, dict):
                data = self._normalize_decision(data) or data
            if isinstance(data, dict):
                action = data.get("action") or "dispatch"
                raw_tasks = data.get("tasks") or []
                summary_txt = data.get("summary") or ""
                note = data.get("note") or ""
            else:
                action = getattr(data, "action", "dispatch")
                raw_tasks = getattr(data, "tasks", []) or []
                summary_txt = getattr(data, "summary", "") or ""
                note = getattr(data, "note", "") or ""
            valid = set(targets)
            out_tasks = self._decision_tasks(raw_tasks, valid)
            if not out_tasks and action != "finish":
                # The structured output may come back empty/unusable while the
                # model's own text still carries a full decision. Never lose a
                # whole round over that — recover it from the raw output.
                for cand in fallback_texts:
                    alt = self._parse_decision_text(cand)
                    if not isinstance(alt, dict):
                        continue
                    alt = self._normalize_decision(alt)
                    alt_tasks = self._decision_tasks(alt.get("tasks") or [], valid)
                    if not alt_tasks:
                        continue
                    out_tasks = alt_tasks
                    action = alt.get("action") or action
                    summary_txt = alt.get("summary") or summary_txt
                    note = alt.get("note") or note
                    print("[commander-decide] recovered the decision from the "
                          "raw model output")
                    break
            if action == "finish":
                return {"action": "finish", "tasks": [],
                        "summary": summary_txt, "note": note}
            if action not in ("dispatch", "finish"):
                print(f"[commander-decide] invalid action={action!r}; refusing to guess")
                return self._fake_commander_decide(text, targets, boxes, round_no)
            if not out_tasks:
                # Never turn an invalid/empty dispatch into a false finish.
                return self._fake_commander_decide(text, targets, boxes, round_no)
            return {"action": "dispatch", "tasks": out_tasks,
                    "summary": summary_txt, "note": note}
        except Exception as ex:  # pragma: no cover
            print(f"[commander-decide] failed, using fallback: {ex}")
            return self._fake_commander_decide(text, targets, boxes, round_no)

    def _decision_tasks(self, raw_tasks, valid):
        """Turn the raw decision tasks into validated, normalized task dicts."""
        out = []
        for t in raw_tasks or []:
            if isinstance(t, dict):
                raw_eid = t.get("egg_id"); message = t.get("message")
                deliver = t.get("deliver") or []
                notify = t.get("notify") or []
                depends = t.get("depends_on") or []
            else:
                raw_eid = getattr(t, "egg_id", None)
                message = getattr(t, "message", None)
                deliver = getattr(t, "deliver", []) or []
                notify = getattr(t, "notify", []) or []
                depends = getattr(t, "depends_on", []) or []
            eid = self._resolve_egg_reference(raw_eid, valid)
            if eid in valid and message:
                dependency_holder = {"egg_id": eid, "depends_on": depends}
                normalized_dependencies = self._normalize_task_refs(
                    dependency_holder, "depends_on", set(self.eggs))
                out.append({
                    "egg_id": eid, "message": str(message),
                    "deliver": [d for d in self._normalize_egg_refs(
                        deliver, set(self.eggs), "deliver") if d != eid],
                    "notify": [n for n in self._normalize_egg_refs(
                        notify, set(self.eggs), "notify") if n != eid],
                    "depends_on": normalized_dependencies,
                    "_dependency_errors": list(dependency_holder.get("_dependency_errors") or []),
                })
        return self._dedupe_commander_tasks(out)

    def _normalize_decision(self, data):
        """Tolerate the schema deviations local models actually produce.

        A usable decision must not be thrown away over a cosmetic deviation, so
        these shapes are normalized instead of rejected:
          * notify / deliver / depends_on written as a bare string;
          * depends_on written at the TOP level (it applies to every task);
          * tasks given as a single object instead of a list.
        """
        if not isinstance(data, dict):
            return None
        top_deps = data.get("depends_on") or []
        if isinstance(top_deps, (str, int)):
            top_deps = [top_deps]
        elif not isinstance(top_deps, (list, tuple)):
            top_deps = []
        tasks = data.get("tasks") or []
        if isinstance(tasks, dict):
            tasks = [tasks]
        out_tasks = []
        for t in tasks:
            if not isinstance(t, dict):
                continue
            t = dict(t)
            for field in ("deliver", "notify", "depends_on"):
                val = t.get(field)
                if isinstance(val, (str, int)):
                    t[field] = [val]
                elif isinstance(val, (list, tuple, set)):
                    t[field] = [v for v in val]
                else:
                    t[field] = []
            # A top-level depends_on means "these tasks wait for these eggs".
            if not t.get("depends_on") and top_deps:
                t["depends_on"] = list(top_deps)
            out_tasks.append(t)
        return {
            "action": data.get("action") or "dispatch",
            "tasks": out_tasks,
            "summary": data.get("summary") or "",
            "note": data.get("note") or "",
        }

    # ---- what counts as a finished run (not just a finished stream) ----
    def _run_has_output(self, rec):
        """True when THIS run produced real answer text.

        The egg's transcript tells the two apart exactly: the model's answer is
        recorded as kind="text", its thinking as kind="thought". A run that
        produced only thinking / tool chatter has NOT concluded anything, so the
        scheduler must never treat it as a finished stage (that is how a cut-off
        turn used to be reported as "阶段完成" while the egg was still talking
        to the LLM).
        """
        egg = self.eggs.get((rec.get("step") or {}).get("egg_id"))
        if egg is not None:
            facts = getattr(egg, "last_completion", None) or {}
            try:
                # Authoritative for the run that just finished.
                return int(facts.get("reply_len") or 0) > 0
            except Exception:
                pass
        # Fallback for eggs without completion facts: strip the activity lines.
        for raw in str(rec.get("reply") or "").splitlines():
            s = raw.strip()
            if s and not s.startswith(_ACTIVITY_PREFIXES):
                return True
        return False

    def _run_was_cut_off(self, rec):
        """True when the turn was aborted mid-flight (stall / upstream error)."""
        reply = str(rec.get("reply") or "")
        return any(marker in reply for marker in _ABORT_MARKERS)

    # ---- what may be PASSED ON to another agent ----
    def _assistant_text(self, egg):
        """The egg's assistant-only text (its own answers, no thinking/tools)."""
        lines = []
        for r in (getattr(egg, "transcript", None) or []):
            if r.get("kind") == "text" and r.get("role") == "assistant":
                lines.append(r.get("text") or "")
        return "\n".join(lines).strip()

    def _assistant_only(self, egg, rec):
        """The run's assistant-only text: the ONLY thing that may be passed on.

        A conversation carries three different things: the model's 💭 thinking,
        its 🔧 tool calls / ↳ results, and its actual answer. Only the answer
        travels to another egg or to the commander — the rest is local process
        noise that would pollute the receiver's context (and blow up its
        context window, which is what used to make the runs fail).

        The egg's transcript knows the difference exactly: the answer is
        recorded as kind="text", the thinking as kind="thought".
        """
        mark = getattr(egg, "_run_mark", None)
        if egg is not None and isinstance(mark, int):
            texts = [r.get("text") or "" for r in (egg.transcript or [])[mark:]
                     if r.get("kind") == "text" and r.get("role") == "assistant"]
            if texts:
                return "\n".join(texts).strip()
        # Fallback (eggs without a transcript): drop the activity lines.
        lines = []
        for raw in str(rec.get("reply") or "").splitlines():
            s = raw.strip()
            if not s or s.startswith(_ACTIVITY_PREFIXES):
                continue
            lines.append(s)
        return "\n".join(lines).strip()

    # ---- delivery contract (what an egg must hand over) ----
    def _delivery_path(self, egg, message):
        """The ONE deliverable file of a task.

        The commander usually names it (``正方一辩(支持方).md``); when it does
        not, a stable default is derived from the egg's title so every task
        still has exactly one unambiguous artifact to verify.

        The path is always relative to the egg's OWN worktree root — that root
        IS the directory shown in the 「审阅代码」 panel, so a deliverable there
        is one click away from the user. Any ``agent_data/`` prefix is stripped:
        that folder is the SDK's internal app-data dir and is NOT listed in
        「审阅代码」, so a file written there would be invisible to the user.
        """
        text = str(message or "")
        m = re.search(r"agent_data/[^\s`'\"]+?\.md", text)
        if m:
            # Normalize "agent_data/x.md" -> "x.md" (the reviewable location).
            return m.group(0).split("agent_data/", 1)[1]
        # A relative path named by the commander (e.g. 结论.md / docs/结论.md).
        m = re.search(r"(?<![\w.\\-])((?:[^\s`'\"/\\]+/)*[^\s`'\"/\\]+\.md)", text)
        if m and m.group(1) != "README.md":
            return m.group(1)
        title = str(getattr(egg, "title", "") or "").strip()
        safe = re.sub(r'[\\/:*?"<>|\s]+', "_", title).strip("_")
        return f"{safe or '交付物'}.md"

    def _has_named_deliverable(self, message):
        """True when the commander already named the deliverable file itself."""
        text = str(message or "")
        if re.search(r"agent_data/[^\s`'\"]+?\.md", text):
            return True
        m = re.search(r"(?<![\w.\\-])((?:[^\s`'\"/\\]+/)*[^\s`'\"/\\]+\.md)", text)
        return bool(m and m.group(1) != "README.md")

    def _delivery_standard(self, egg, message):
        """The explicit delivery contract handed to every dispatched egg."""
        path = self._delivery_path(egg, message)
        return (
            "\n\n【交付标准】（本任务必须逐条满足）\n"
            f"1) 交付物：把本轮成果写成 Markdown 文件 {path}，"
            "放在你自己的工作目录【根目录】下——那正是用户点「审阅代码」时看到的目录"
            "（不要写进 agent_data/，那是内部目录，用户在「审阅代码」里看不到）；\n"
            "2) 内容：只写你本节点负责的结论，不复述其它蛋的内容。\n"
            "3) 验收：写完后用读取工具读回该文件，确认已写入且内容非空。\n"
            "4) 通知：在最终回复里写明交付物路径与要点，"
            "并通知 notify 中列出的蛋。\n"
            "只在聊天里说“已完成”而不落盘，视为【未交付】——"
            "系统会核验文件并在监控面板里标出【未交付】。"
        )

    def _peer_directories_block(self, egg_ids):
        """A compact list of each participant egg's deliverable directory.

        A sub-egg (e.g. the debate judge) must READ what its peers actually
        produced, but it only knows its OWN workspace. We hand it the other
        participants' worktree roots — that root IS the directory the user sees
        in 「审阅代码」, where every deliverable .md lands. The block is appended
        to dispatched task messages so any egg can ``view_file`` / ``list_directory``
        the others' artifacts instead of guessing or fabricating.
        """
        rows = []
        seen = set()
        for eid in egg_ids:
            if not eid or eid in seen:
                continue
            egg = self.eggs.get(eid)
            if egg is None:
                continue
            wt = getattr(egg, "worktree", None)
            if not wt:
                continue
            seen.add(eid)
            rows.append(f"- {egg.title}（{eid}）：{wt}")
        if not rows:
            return ""
        return (
            "\n\n【协作蛋交付物目录】需要参照其他蛋的真实产出时，可用 view_file / "
            "list_directory 读取以下工作目录中的交付物 .md 文件（这些目录即用户在"
            "「审阅代码」中看到的目录，交付物都落在其根目录下）：\n" + "\n".join(rows)
        )

    def _verify_delivery(self, egg, message, started_at=None):
        """Check that the task's deliverable really landed on disk.

        The commander must be able to tell “delivered” from “only talked about
        it”, so the answer is always objective: does the file exist, is it
        non-empty, and was it written during this run? A file written outside
        the agreed path still counts as delivered (the path drift is reported),
        because the CONTENT is what the debate needs.
        """
        expected = self._delivery_path(egg, message)
        base = None
        try:
            base = Path(egg.worktree).resolve()
        except Exception:
            base = None

        def _read(rel):
            """Return {path, chars, bytes, mtime} for a worktree-relative file."""
            if base is None:
                return None
            try:
                fp = (base / rel).resolve()
                if base not in fp.parents or not fp.is_file():
                    return None
                text = fp.read_text(encoding="utf-8", errors="replace")
                stat = fp.stat()
                return {"path": rel, "chars": len(text.strip()),
                        "bytes": stat.st_size, "mtime": stat.st_mtime}
            except Exception:
                return None

        def _fresh(mtime):
            if not started_at:
                return ""
            return ("，本轮写入" if mtime >= float(started_at) - 5
                    else "，此前轮次已写入")

        hit = _read(expected)
        if hit:
            if hit["chars"] == 0:
                return {"path": expected, "delivered": False, "bytes": hit["bytes"],
                        "chars": 0,
                        "detail": f"未交付：交付物 {expected} 内容为空"}
            return {"path": expected, "delivered": True, "bytes": hit["bytes"],
                    "chars": hit["chars"],
                    "detail": f"已交付 {expected}（{hit['chars']} 字{_fresh(hit['mtime'])}）"}
        # The egg may have written its markdown somewhere else in its worktree.
        # Only files the user can actually SEE in 「审阅代码」 count: anything
        # under agent_data/ is SDK-internal and is not listed there, and the
        # auto-generated README.md placeholder is not a deliverable.
        best = None
        try:
            for p in sorted(base.rglob("*.md")) if base else []:
                if not p.is_file():
                    continue
                rel = str(p.relative_to(base))
                parts = Path(rel).parts
                if parts and parts[0] == "agent_data":
                    continue
                if Path(rel).name.lower() == "readme.md":
                    continue
                cand = _read(rel)
                if not cand or cand["chars"] == 0:
                    continue
                if started_at and cand["mtime"] < float(started_at) - 5:
                    continue
                if best is None or cand["mtime"] > best["mtime"]:
                    best = cand
        except Exception:
            best = None
        if best:
            return {"path": best["path"], "delivered": True, "bytes": best["bytes"],
                    "chars": best["chars"],
                    "detail": f"已交付 {best['path']}（{best['chars']} 字"
                              f"{_fresh(best['mtime'])}；与约定路径 {expected} 不一致）"}
        return {"path": expected, "delivered": False, "bytes": 0, "chars": 0,
                "detail": f"未交付：找不到交付物 {expected}"}

    def _fake_commander_decide(self, text, targets, boxes, round_no):
        """Return a controlled error; never invent a task when the planner fails.

        The scheduler cannot infer phase/role semantics from a failed model response.
        Returning an explicit error is safer than repeatedly dispatching the first
        missing egg and falsely making progress.
        """
        return {
            "action": "error",
            "tasks": [],
            "summary": (
                f"总指挥第 {round_no} 轮决策输出无法解析或不可用。"
                "已停止自动调度以避免重复派活或越过阶段；当前未完成任务不会被报告为成功。"
                "请重试，或检查总指挥的结构化输出。"
            ),
            "note": "决策失败：未生成回退任务。",
        }

    def _planner_chunk_text(self, chunk):
        """Render a planner chunk as display text for the commander panel.

        The planner runs in STRUCTURED-OUTPUT mode, so its decision travels in
        a special tool call named ``finish`` whose args carry the raw JSON —
        that must NOT be shown to the user (they should read the commander's
        own words, not the JSON envelope). Everything else — its text, its
        thinking, its real tool calls and their results — is shown.
        """
        if isinstance(chunk, types.Text):
            # The model usually appends its structured decision as a ```json
            # fence (or a "结构化决策" section) at the end of its answer. That
            # JSON is parsed separately by _parse_decision_text and must NOT be
            # shown verbatim — keep only the commander's natural-language plan.
            return self._strip_structured_decision(chunk.text or "")
        if isinstance(chunk, types.Thought):
            return f"💭 {self._strip_structured_decision(chunk.text or '')}"
        if isinstance(chunk, types.ToolCall):
            name = getattr(chunk, "name", None)
            if not name or name == "finish":
                return ""          # the structured-output envelope, not a tool
            return f"🔧 调用工具：{name}"
        if isinstance(chunk, types.ToolResult):
            res = getattr(chunk, "result", None)
            if res is None:
                res = getattr(chunk, "error", None)
            res = str(res or "")
            if len(res) > 300:
                res = res[:300] + "…"
            return f"↳ 工具结果：{res}"
        if isinstance(chunk, str):
            return chunk
        return ""

    def _strip_structured_decision(self, text):
        """Drop the structured decision from text shown in the commander panel.

        Local models emit their decision as a trailing ```json fence (often
        preceded by a "结构化决策" / "结构化输出" header) right after their
        natural-language plan. That JSON is recovered by ``_parse_decision_text``
        and should never be shown to the user verbatim — truncate at the first
        such marker so only the commander's readable plan reaches the panel.
        """
        if not text:
            return text
        cut = len(text)
        for m in re.finditer(r"```", text):
            cut = min(cut, m.start())
        for pat in (r"结构化决策", r"结构化输出"):
            m = re.search(pat, text)
            if m:
                cut = min(cut, m.start())
        # Drop the bold/header opener ("**") and trailing punctuation that
        # preceded the structured-decision marker.
        return re.sub(r"[*#\s：:]+$", "", text[:cut].rstrip())

    def _finish_payload(self, chunk):
        """Extract the raw decision JSON carried by the `finish` tool call."""
        args = getattr(chunk, "args", None)
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                return None
        if not isinstance(args, dict):
            return None
        for key in ("output_string", "output", "text", "result"):
            val = args.get(key)
            if isinstance(val, str) and val.strip():
                return val
            if isinstance(val, dict):
                return json.dumps(val, ensure_ascii=False)
        return None

    def _iter_balanced_json(self, text):
        """Yield every top-level, brace-balanced JSON object slice in ``text``.

        A top-level object starts at a `{` whose matching `}` (respecting string
        literals and escapes) closes it. The scan resumes right after each
        captured object, so nested braces are not re-yielded as candidates. An
        unterminated (truncated) object is skipped instead of breaking the scan.
        """
        out = []
        n = len(text)
        idx = 0
        while True:
            start = text.find("{", idx)
            if start == -1:
                break
            depth = 0
            in_str = False
            esc = False
            closed = -1
            for i in range(start, n):
                ch = text[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        closed = i
                        break
            if closed == -1:
                # Unterminated object (e.g. truncated stream) — skip this brace.
                idx = start + 1
            else:
                out.append(text[start:closed + 1])
                idx = closed + 1
        return out

    def _parse_decision_text(self, text):
        """Parse a commander decision out of raw model output.

        Local models produce messy output, so this tolerates and RECOVERS from:
          * a bare JSON object;
          * JSON wrapped in prose or ```json fences;
          * SEVERAL JSON objects emitted in one turn (duplicated decisions — the
            model sometimes restates the plan as a fenced object and then emits
            the same decision again as bare JSON);
          * a leading JSON that is truncated/invalid while a later one is whole.

        It never returns the first thing it finds — it scores every candidate
        and returns the BEST usable decision (a valid ``action`` with the most
        ``tasks``, preferring one that also carries ``summary``/``note``), so a
        good decision is never lost to a stray or repeated fragment.
        """
        if not text:
            return None
        text = text.strip()

        candidates = []  # list of JSON strings / already-parsed dicts

        # 1) the whole thing may be JSON
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                candidates.append(obj)
        except Exception:
            pass

        # 2) ```json ... ``` (or bare ``` ... ```) fences — their content is a
        #    complete JSON object to consider.
        for m in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.S):
            block = m.group(1).strip()
            if block:
                candidates.append(block)

        # 3) every balanced {...} object found in the raw text.
        candidates.extend(self._iter_balanced_json(text))

        best = None
        best_score = (-1,)
        for cand in candidates:
            obj = cand if isinstance(cand, dict) else None
            if obj is None:
                try:
                    obj = json.loads(cand)
                except Exception:
                    continue
            if not isinstance(obj, dict):
                continue
            action = obj.get("action")
            tasks = obj.get("tasks") or []
            if isinstance(tasks, dict):
                tasks = [tasks]
            elif not isinstance(tasks, list):
                tasks = []
            # Higher is better: valid action, more tasks, has summary, has note.
            score = (
                1 if action in ("dispatch", "finish") else 0,
                len(tasks),
                1 if obj.get("summary") else 0,
                1 if obj.get("note") else 0,
            )
            if score > best_score:
                best_score = score
                best = obj
        return best

    def _record_planner_chunk(self, chunk):
        """Record a planner (round-decision) chunk into the commander's trace.

        The planner agent is what actually THINKS each round, so its 💭
        thinking and 🔧 tool calls belong in the 总指挥's 轨迹 too.
        """
        cmd = self.commander
        if isinstance(chunk, types.Thought):
            cmd._record_thought(chunk.text)
        elif isinstance(chunk, types.ToolCall):
            cmd._record("assistant", "tool_call",
                        name=getattr(chunk, "name", None),
                        args=getattr(chunk, "args", None))
        elif isinstance(chunk, types.ToolResult):
            cmd._record("tool", "tool_result",
                        name=getattr(chunk, "name", None),
                        result=getattr(chunk, "result", None),
                        error=getattr(chunk, "error", None))
        elif isinstance(chunk, types.Text):
            cmd._record_text(chunk.text)
        elif isinstance(chunk, str):
            # The local harness sometimes emits a bare string instead of a Text
            # chunk; without this the decision would be invisible in the 轨迹.
            cmd._record_text(chunk)

    def _chan_label(self, eid):
        """Human label for a channel endpoint ("总指挥" or the egg's title)."""
        if eid == "commander":
            return "总指挥"
        if eid in self.eggs:
            return f"蛋「{self.eggs[eid].title}」"
        return str(eid)

    def _fmt_channel(self, entries):
        """Render a channel's user+assistant history as readable text."""
        out = []
        for e in entries:
            role = e.get("role")
            who = "用户" if role == "user" else "助手"
            out.append(f"{who}：{e.get('text', '')}")
        return "\n".join(out)

    def _task_context(self, task, boxes, channels, inbox=None, active_ids=None):
        """Build task context; inbox messages are acknowledged only after successful completion."""
        eid = task["egg_id"]
        blocks = []
        if task.get("goal_context"):
            blocks.append("【用户总指令（背景上下文）】\n" + str(task["goal_context"]))
        ch = channels.get(("commander", eid))
        if ch:
            blocks.append("【你与总指挥的通讯历史（user+assistant）】\n" + self._fmt_channel(ch))
        active_ids = set(active_ids or ())
        for src in task.get("deliver") or []:
            # A source that is active now may still have a prior, completed result.
            # It is safe to deliver that version unless depends_on explicitly waits for
            # the source's current run (such a task will be deferred by the scheduler).
            box = boxes.get(src)
            if src in active_ids and not box:
                continue
            if not box:
                continue
            source_title = self.eggs[src].title if src in self.eggs else src
            block = (f"【总指挥授权转交：蛋「{source_title}」的结论】\n"
                     f"{box.get('result', '')}")
            peer_history = channels.get((src, eid))
            if peer_history:
                block += "\n\n【与该蛋的通讯历史（user+assistant）】\n" + self._fmt_channel(peer_history)
            blocks.append(block)
        pending = list((inbox or {}).get(eid, []) or [])
        task["_consumed_inbox"] = pending
        if pending:
            blocks.append("【其它蛋发给你的完成通知（本次成功完成后才确认已读）】\n"
                          + "\n\n".join(pending))
        if not blocks:
            return None
        return "\n\n".join(blocks)

    # ---- one-click run of selected eggs (parallel / serial by graph) ----
    def _levels(self, targets):
        """Return topological levels; fail explicitly if the selected graph has a cycle."""
        targets = list(dict.fromkeys(targets))
        target_set = set(targets)
        preds = {target: [] for target in targets}
        for edge in self.edges:
            source, dest = edge.get("from"), edge.get("to")
            if source in target_set and dest in target_set and source != dest:
                if source not in preds[dest]:
                    preds[dest].append(source)
        levels = []
        done = set()
        remaining = set(targets)
        while remaining:
            ready = [target for target in targets
                     if target in remaining and set(preds[target]) <= done]
            if not ready:
                cycle_nodes = ", ".join(sorted(remaining))
                raise ValueError(f"所选节点之间存在循环依赖：{cycle_nodes}；请先删除造成循环的连线。")
            levels.append(ready)
            done.update(ready)
            remaining.difference_update(ready)
        return levels, preds

    def _run_message(self, egg, upstream_ids, outputs):
        """Build an egg's run prompt.

        A serial (downstream) egg inherits each upstream egg's CONVERSATION —
        but only the assistant's own answers: the 💭 thinking and the 🔧 tool
        calls / results are NOT passed on, they stay in the upstream's 轨迹.
        Falls back to the captured output text if an upstream has no transcript.
        """
        up_blocks = []
        for u in upstream_ids:
            ue = self.eggs.get(u)
            if not ue:
                continue
            hist = ue.history_text(mode=self.history_mode, limit=4000).strip()
            if not hist:
                hist = (outputs.get(u, "") or "").strip()
            if hist:
                up_blocks.append(
                    f"### 上游蛋「{ue.title}」（职责：{ue.prompt}）的会话历史：\n{hist}"
                )
        parts = []
        if up_blocks:
            parts.append("以下是你的上游（被依赖）蛋的会话历史，请基于它们完成你的任务：")
            parts.extend(up_blocks)
            parts.append("")
        parts.append(f"你的任务：{egg.prompt}")
        return "\n".join(parts)

    async def run_selected(self, targets):
        """Run selected eggs topologically, cancelling all workers if the caller exits early."""
        targets = list(dict.fromkeys(t for t in (targets or []) if t in self.eggs))
        if not targets:
            yield {"k": "end"}
            return
        try:
            levels, preds = self._levels(targets)
        except ValueError as ex:
            yield {"k": "say", "text": f"无法执行：{ex}", "role": "system"}
            yield {"k": "end"}
            return

        outputs = {}
        active_workers = []
        total_levels = len(levels)
        for target in targets:
            yield {"k": "status", "egg": target, "status": "planned"}

        try:
            for level_index, level in enumerate(levels, 1):
                names = "、".join(self.eggs[target].title for target in level)
                mode = "并行" if len(level) > 1 else "串行"
                yield {"k": "say", "text": f"▶ 第 {level_index}/{total_levels} 层（{mode}）：{names}\n", "role": "system"}
                for target in level:
                    yield {"k": "status", "egg": target, "status": "running"}
                queue = asyncio.Queue()

                async def worker(eid):
                    egg = self.eggs[eid]
                    message = self._run_message(egg, preds.get(eid, []), outputs)
                    await queue.put({"k": "instruct", "egg": eid,
                                     "text": "▶ 运行指令：" + message})
                    buffer = []
                    inner = None

                    async def drain():
                        generator = egg.run(message)
                        try:
                            async for token in generator:
                                buffer.append(str(token))
                                await queue.put({"k": "egg", "egg": eid, "text": token})
                        finally:
                            try:
                                await generator.aclose()
                            except Exception:
                                pass

                    try:
                        inner = asyncio.create_task(drain())
                        await inner
                    except asyncio.CancelledError:
                        if inner is not None and not inner.done():
                            inner.cancel()
                        if inner is not None:
                            await asyncio.gather(inner, return_exceptions=True)
                        raise
                    except Exception as ex:  # pragma: no cover
                        await queue.put({"k": "egg", "egg": eid,
                                         "text": f"\n（执行出错：{ex}）"})
                    finally:
                        # Only the assistant's own text may travel downstream:
                        # the 💭 thinking / 🔧 tool chatter stays local.
                        outputs[eid] = self._assistant_text(egg)
                        await queue.put({"k": "status", "egg": eid, "status": "idle"})
                        await queue.put({"k": "end_egg", "egg": eid})

                workers = [asyncio.create_task(worker(target)) for target in level]
                active_workers.extend(workers)
                finished_count = 0
                while finished_count < len(level):
                    event = await queue.get()
                    if event.get("k") == "end_egg":
                        finished_count += 1
                    yield event
                await asyncio.gather(*workers, return_exceptions=True)
                active_workers = [worker_task for worker_task in active_workers
                                  if not worker_task.done()]
        finally:
            for worker_task in active_workers:
                if not worker_task.done():
                    worker_task.cancel()
            if active_workers:
                await asyncio.gather(*active_workers, return_exceptions=True)
        yield {"k": "end"}

    def _commander_context(self, targets, mode=None):
        """Graph summary PLUS the selected eggs' historical conversations.

        Lets the Commander (总指挥) inherit what each selected egg has actually
        done/decided — not just its prompt — when advising or driving them.
        Only the assistant's own text is transmitted: the 💭 thinking and the
        🔧 tool calls / results stay in each egg's local 轨迹.
        """
        mode = mode or self.history_mode
        base = self._graph_summary(targets[0] if targets else None)
        if not targets:
            return base
        blocks = []
        for t in targets:
            e = self.eggs.get(t)
            if not e:
                continue
            hist = e.history_text(mode=mode, limit=6000).strip()
            blocks.append(
                f"### 蛋「{e.title}」（{e.type}）职责：{e.prompt}\n"
                f"会话历史：\n{hist if hist else '（暂无会话记录）'}"
            )
        return base + "\n\n【选中蛋的会话历史】\n" + "\n\n".join(blocks)

    def _graph_summary(self, target=None):
        lines = ["当前画布上的蛋与连线（仅用户连线可互通）："]
        for e in self.eggs.values():
            lines.append(f"- [{e.id}] {e.title} ({e.type}): {e.prompt[:80]}")
        for ed in self.edges:
            a = self.eggs.get(ed["from"])
            b = self.eggs.get(ed["to"])
            lines.append(
                f"  连线: {a.title if a else ed['from']} -> {b.title if b else ed['to']}"
            )
        if target and target in self.eggs:
            lines.append(f"协调目标：{self.eggs[target].title}")
        return "\n".join(lines)
