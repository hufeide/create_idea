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

# Verbs that mean "this task must READ another egg's conclusion" (a real
# dependency).  A bare mention ("预判反方的主要反驳") is NOT one of them.
_CONSUME_VERBS = ("综合", "评判", "比较", "阅读", "基于", "依据", "参考",
                   "权衡", "裁决", "总结", "结合", "汇总", "整合", "评估",
                   "归纳", "复核", "批改", "审阅")

# An egg whose ROLE is one of these consumes other eggs' output (a judge, an
# aggregator, a reviewer).  Such an egg must never run in the same round as
# the producers it is supposed to weigh — the structure of the task, not the
# wording of the instruction, decides that.
_CONSUMER_ROLES = ("裁判", "评审", "评判", "评委", "仲裁", "汇总", "整合",
                   "总结", "对比", "比较", "协调", "监督", "审核", "验收",
                   "分析", "决策", "裁定", "权衡", "复盘", "汇报", "主持")


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
    deliver: list[str] = []
    # Who this egg must NOTIFY once it finishes. The commander is ALWAYS
    # notified (that is the default channel); ids listed here are ADDITIONAL
    # eggs the commander authorizes to receive this egg's conclusion.
    notify: list[str] = []
    # Eggs whose CONCLUSION this task depends on. A task that depends on
    # another task OF THE SAME ROUND cannot start yet — the commander defers
    # it to the next round (e.g. a judge that must weigh both debaters).
    depends_on: list[str] = []


class CommanderDecision(pydantic.BaseModel):
    """The commander's decision for ONE round.

    ``action``:
      - "dispatch" -> run ``tasks`` now (1 task = serial, N tasks = parallel)
      - "finish"   -> the instruction is complete; ``summary`` is the recap
    """
    action: str = "dispatch"
    tasks: list[CommanderTask] = []
    summary: str = ""
    note: str = ""


class Canvas:
    def __init__(self):
        self.eggs = {}
        self.edges = []
        self._id = 0
        self.commander = Commander()
        self.history_mode = "brief"
        # Session (canvas) persistence. The canvas boots EMPTY; a session is created
        # on first mutation (add_egg) or on every generate(). All of a session's
        # eggs + their SDK conversation state live under one dated directory.
        self.session_id = None
        self.session_dir = None
        self.created_at = None
        self.label = None
        # Throttle for autosave() so drags/rapid edits don't storm the disk.
        self._last_save = None

    # ---- id helpers ----
    def _new_id(self, p="e"):
        self._id += 1
        return f"{p}{self._id}"

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
        # Persist the current session before tearing agents down.
        try:
            self.save()
        except Exception:  # pragma: no cover
            pass
        for e in self.eggs.values():
            await e.stop()
        await self.commander.stop()

    # ---- session (canvas) persistence ----
    def _new_session(self, label=None):
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
        # A fresh canvas gets a FRESH commander conversation (not the previous
        # canvas's id/transcript); the commander keeps its own sub-directory
        # (session_dir/commander_data) just like each egg has its worktree.
        self.commander.bind_session(base, None)
        self.commander.transcript = []
        return sid

    def ensure_session(self):
        """Create a session if this canvas is still unbound (empty / fresh)."""
        if self.session_id is None:
            self._new_session()

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
        """Resume a saved session: rebuild eggs/edges/transcripts and reattach each
        egg's conversation_id + session save_dir so its LLM chat truly RESUMES."""
        base = SESSIONS_DIR / session_id
        path = base / "canvas.json"
        if not path.exists():
            raise FileNotFoundError(f"session not found: {session_id}")
        data = json.loads(path.read_text(encoding="utf-8"))
        # Stop any live agents from the previously in-memory session.
        await self.commander.stop()
        for e in list(self.eggs.values()):
            await e.stop()
        self.eggs.clear()
        self.edges.clear()
        self.session_id = data.get("session_id", session_id)
        self.session_dir = base
        self.created_at = data.get("created_at")
        self.label = data.get("label")
        self.history_mode = data.get("history_mode", "brief")
        for ed in data.get("eggs", []):
            egg = EggAgent(
                ed["id"], ed["title"], ed["prompt"], ed["type"],
                (ed.get("x", 0), ed.get("y", 0)),
                subagents=ed.get("subagents"),
                parent=ed.get("parent"),
                session_dir=str(base),
                conversation_id=ed.get("conversation_id"),
            )
            egg.transcript = ed.get("transcript", [])
            self.eggs[ed["id"]] = egg
        for e in data.get("edges", []):
            if e.get("from") in self.eggs and e.get("to") in self.eggs:
                self.edges.append(e)
        self.commander.transcript = data.get("commander", [])
        self.commander.bind_session(base, data.get("commander_conversation_id"))
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
            session_dir=self.session_dir, conversation_id=None,
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
                    "one part. Return concise titles and clear prompts."
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
            return [
                {
                    "title": e.get("title") if isinstance(e, dict) else e.title,
                    "prompt": e.get("prompt") if isinstance(e, dict) else e.prompt,
                    "type": (e.get("type") if isinstance(e, dict) else getattr(e, "type", "module")) or "module",
                }
                for e in eggs
            ]
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
        ids = [i for i in ids if i in self.eggs]
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
            from google.antigravity import types
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
        self.autosave()

    async def delete_edge(self, edge_id):
        self.edges = [ed for ed in self.edges if ed["id"] != edge_id]
        self.autosave()

    async def add_edge(self, from_, to):
        if from_ == to or from_ not in self.eggs or to not in self.eggs:
            return False
        if any(ed["from"] == from_ and ed["to"] == to for ed in self.edges):
            return False
        self.edges.append({"id": self._new_id("edge"), "from": from_, "to": to})
        self.autosave()
        return True

    async def move(self, eid, x, y):
        e = self.eggs.get(eid)
        if e:
            e.x, e.y = x, y
        self.autosave()

    # ---- commander ----
    async def commander_run(self, text, targets=None):
        """总指挥以【持续循环 + 实时监控】方式驱动选中的子蛋。

        The commander is a LOOP, not a one-shot planner:

          * 【持续监控】子蛋在【后台】运行，总指挥因此可以在它们工作期间持续
            观察：会话有没有变长（delta）、有没有待处理的完成通知、有没有超时
            或长时间静默 —— 并据此决定下一步动作。
          * 【主动通知】每只子蛋跑完后会主动通知总指挥（默认通道）；总指挥也
            可以在任务的 ``notify`` 字段里指定【完成后要通知的其它子蛋】，那
            只蛋下次运行时就会收到这条完成通知。
          * 【只有完成才跳出】循环只在决策返回 action="finish"（指令真的完成）
            时退出。两个兜底阀门防止无限空转：连续 _MAX_IDLE 轮没有可执行的
            任务、或达到 _MAX_ROUNDS 轮上限。

        Yields UI events:
          {"k":"say","text":..}      -> stream into the commander chat
          {"k":"instruct","egg":id,"text":..} -> push the instruction into an egg
          {"k":"egg","egg":id,"text":..}      -> stream an egg's reply tokens
          {"k":"status","egg":id,"status":..} -> egg status (planned/running/idle)
          {"k":"end_egg","egg":id}            -> finalize an egg's stream
          {"k":"plan","round":n,"steps":[..]} -> the current round's plan
          {"k":"end"}                          -> finalize the commander stream
        """
        targets = [t for t in (targets or []) if t in self.eggs]
        summary = self._commander_context(targets)

        # No targets selected -> plain coordination advice (unchanged behavior).
        if not targets:
            async for tok in self.commander.run(text, summary):
                yield {"k": "say", "text": tok}
            yield {"k": "end"}
            return

        # ---- control mode: a monitoring LOOP over the selected eggs ----
        # The commander/egg channels carry user+assistant history only.
        summary = self._commander_context(targets, mode="brief")
        boxes = {}        # egg_id -> {"result": str}   (the 消息盒子)
        channels = {}     # (src, dst) -> [{"role","text"}]  user+assistant only
        cmd_hist = []     # commander-side user+assistant history
        cmd_hist.append({"role": "user", "text": text})
        inbox = {}        # egg_id -> [完成通知] 等待该蛋下次运行时读取
        notified = []     # 自上次裁决以来收到的【完成通知】
        running = {}      # egg_id -> 本次指挥中在后台运行的蛋
        events = asyncio.Queue()   # 后台产生的 UI 事件
        watch_base = {}   # egg_id -> 上次裁决时的会话长度（用来算增量）
        postponed = []    # 上一轮因依赖被推迟的任务（下一轮要派活给它们）
        reported = set()  # 已上报过的告警，避免每轮重复刷屏
        all_results = []  # (title, message, reply) for the final recap
        final_summary = ""
        round_no = 0
        idle_rounds = 0
        bg = []           # 后台任务（裁决 / 子蛋运行），退出时统一清理
        normal_exit = False   # 总指挥正常收尾（finish）时不该打断还在跑的蛋

        try:
            while True:
                # ---- 1) 监控：把后台事件转发给 UI ----
                while not events.empty():
                    yield events.get_nowait()

                # ---- 2) 监控：收集已经跑完的子蛋（它们会主动通知总指挥） ----
                finished = [eid for eid, rec in running.items() if rec["done"]]
                for eid in finished:
                    rec = running.pop(eid)
                    self._file_result(rec, boxes, channels, cmd_hist, inbox,
                                      notified, all_results)
                    yield {"k": "status", "egg": eid, "status": "idle"}
                    yield {"k": "end_egg", "egg": eid}

                # ---- 3) 监控：会话有没有变长 / 卡住 / 超时 ----
                watch = self._watch_eggs(targets, watch_base, inbox)
                for eid, st in watch.items():
                    if st["delta"] > 0:
                        # 又在推进了 -> 之前的告警作废
                        reported.discard((eid, "stuck"))
                        reported.discard((eid, "overdue"))
                alerts = self._watch_alerts(watch, reported, cmd_hist)

                # ---- 4) 裁决（只有子蛋跑完 / 出现异常时才去问总指挥） ----
                if finished or alerts or not running:
                    round_no += 1
                    if round_no > _MAX_ROUNDS:
                        # 兜底阀门：轮数用尽，用现有结果收尾（走总结流程）。
                        print(f"[commander] round cap {_MAX_ROUNDS} reached")
                        break
                    # 新的一段总指挥发言：让前端封存上一条消息。
                    events.put_nowait({"k": "say_break"})
                    dec = asyncio.create_task(self._commander_decide(
                        text, summary, targets, boxes, channels, cmd_hist,
                        round_no, watch=watch, notified=list(notified),
                        deferred=list(postponed),
                        sink=lambda t: events.put_nowait(
                            {"k": "say", "text": t})))
                    bg.append(dec)
                    # 裁决期间 UI 仍然能刷新子蛋的输出（不冻结面板）。
                    while not dec.done():
                        ev = await self._next_event(events, 0.5)
                        if ev is not None:
                            yield ev
                    decision = await dec
                    if dec in bg:
                        bg.remove(dec)
                    tasks = [t for t in (decision.get("tasks") or [])
                             if t.get("egg_id") in self.eggs]
                    # One task per egg per round: the model sometimes emits
                    # several variants for the same egg (e.g. three "裁判任务").
                    _seen, _uniq = set(), []
                    for t in tasks:
                        if t["egg_id"] in _seen:
                            continue
                        _seen.add(t["egg_id"])
                        _uniq.append(t)
                    tasks = _uniq
                    if decision.get("action") == "finish":
                        # A task that was POSTPONED for a dependency and whose
                        # dependencies are now satisfied is still OUTSTANDING
                        # work -> the instruction is NOT done yet. The commander
                        # may not wrap up while such a task is pending; we
                        # dispatch it instead (a judge that never runs could
                        # not judge anything).
                        ready = [d for d in postponed
                                 if all(dep in boxes
                                        for dep in (d.get("depends_on") or []))]
                        if ready:
                            names = "、".join(d["title"] for d in ready)
                            cmd_hist.append({
                                "role": "user",
                                "text": (f"【调度】{names} 仍在等待被派活"
                                         f"（依赖已满足），指令尚未完成；"
                                         f"请先派活给它再收尾。"),
                            })
                            decision = {
                                "action": "dispatch",
                                "tasks": [{
                                    "egg_id": d["egg_id"],
                                    "message": d["message"],
                                    # it must be able to read the conclusions it
                                    # was waiting for
                                    "deliver": [b for b in boxes
                                                if b != d["egg_id"]],
                                    "notify": [],
                                    "depends_on": d.get("depends_on") or [],
                                } for d in ready],
                                "summary": decision.get("summary") or "",
                                "note": "继续完成被推迟的任务",
                            }
                        else:
                            # 只有真正完成才跳出循环。
                            final_summary = decision.get("summary") or ""
                            normal_exit = True
                            break

                    started = 0
                    deferred = []
                    if tasks:
                        # --- dependency split: a task that waits for another task
                        # of THIS round cannot start yet (e.g. a judge that must
                        # weigh both debaters) — it is deferred to the next round.
                        runnable, deferred = self._split_by_dependency(tasks, boxes)
                        if deferred:
                            names = "、".join(
                                f"蛋「{self.eggs[d['egg_id']].title}」"
                                for d in deferred if d["egg_id"] in self.eggs)
                            cmd_hist.append({
                                "role": "user",
                                "text": (f"【调度】{names} 依赖本轮其它蛋的产出，"
                                         f"已推迟到下一轮执行；请在它们完成后"
                                         f"再派活。"),
                            })
                        tasks = runnable
                        # Remember them: the NEXT round must dispatch them.
                        postponed = [{
                            "egg_id": d["egg_id"],
                            "title": self.eggs[d["egg_id"]].title
                            if d["egg_id"] in self.eggs else d["egg_id"],
                            "message": d.get("message") or "",
                            "depends_on": d.get("depends_on") or [],
                        } for d in deferred if d["egg_id"] in self.eggs]
                    if tasks:
                        # --- this round's plan (1 task = serial, N = parallel) ---
                        mode = "parallel" if len(tasks) > 1 else "serial"
                        plan_steps = [
                            {"egg_id": t["egg_id"],
                             "title": self.eggs[t["egg_id"]].title,
                             "message": t["message"],
                             "deliver": t.get("deliver") or [],
                             "notify": t.get("notify") or [],
                             "depends_on": t.get("depends_on") or [],
                             "mode": mode}
                            for t in tasks
                        ]
                        # Deferred tasks are shown too (grey) so the user can see
                        # WHY they did not run yet.
                        for d in deferred:
                            if d["egg_id"] not in self.eggs:
                                continue
                            plan_steps.append({
                                "egg_id": d["egg_id"],
                                "title": self.eggs[d["egg_id"]].title,
                                "message": d["message"],
                                "deliver": d.get("deliver") or [],
                                "notify": d.get("notify") or [],
                                "depends_on": d.get("depends_on") or [],
                                "mode": "deferred",
                            })
                        note = decision.get("note") or ""
                        narration = (f"【第 {round_no} 轮 · {mode}】{note}\n" + "\n".join(
                            f"→ [{s['title']}] {s['message']}"
                            + (f"  (授权转交: {', '.join(s['deliver'])})"
                               if s["deliver"] else "")
                            + (f"  (完成后通知: {', '.join(s['notify'])})"
                               if s["notify"] else "")
                            + (f"  (等待依赖: {', '.join(s['depends_on'])})"
                               if s["mode"] == "deferred" else "")
                            for s in plan_steps))
                        monitor_line = self._fmt_watch_line(watch)
                        if monitor_line:
                            narration += "\n📡 监控：" + monitor_line
                        if alerts:
                            narration += "\n⚠ " + "\n⚠ ".join(alerts)
                        self.commander._record("system", "plan", text=narration)
                        self.commander._record("assistant", "text", text=narration)
                        cmd_hist.append({"role": "assistant", "text": narration})
                        # 规划说明另起一条消息（前面是总指挥自己的话）。
                        yield {"k": "say_break"}
                        yield {"k": "say", "text": narration}
                        yield {"k": "plan", "round": round_no, "steps": plan_steps}
                        for s in plan_steps:
                            if s["mode"] == "deferred":
                                continue
                            yield {"k": "status", "egg": s["egg_id"],
                                   "status": "planned"}
                        # 子蛋在【后台】运行，总指挥因此可以持续监控它们。
                        started = self._start_runs(
                            [s for s in plan_steps if s["mode"] != "deferred"],
                            boxes, channels, inbox, events, running, bg)
                    if started == 0:
                        if running:
                            # 总指挥把活派给了一只【正在运行】的蛋 -> 没什么
                            # 新东西可启动，等它跑完即可（不算空转）。
                            pass
                        else:
                            # 本轮没有可执行的子任务：记一笔并继续循环（不退出），
                            # 但连续多轮空转会强制收尾，避免无限 loop。
                            idle_rounds += 1
                            cmd_hist.append({
                                "role": "user",
                                "text": (f"（第 {round_no} 轮没有可执行的子任务。"
                                         f"请给出下一步指令；若指令确实已完成，"
                                         f"请用 action='finish' 给出最终总结。）"),
                            })
                            if idle_rounds >= _MAX_IDLE:
                                final_summary = (decision.get("summary")
                                                 or "（总指挥连续多轮没有可执行的任务，"
                                                    "指挥结束。）")
                                break
                            await asyncio.sleep(_MONITOR_INTERVAL)
                    else:
                        idle_rounds = 0
                    notified.clear()
                    watch_base = {k: v["len"] for k, v in watch.items()}
                    continue

                # ---- 5) 还在跑且没有异常：等下一个事件 / 监控 tick ----
                ev = await self._next_event(events, _MONITOR_INTERVAL)
                if ev is not None:
                    yield ev
        finally:
            if not normal_exit:
                # 异常收场（客户端断开 / 被取消 / 兜底阀门触发）：收掉后台任务，
                # 别让子蛋继续空跑。
                for t in bg:
                    if not t.done():
                        t.cancel()
                for rec in list(running.values()):
                    pump = rec.get("pump")
                    if pump is not None and not pump.done():
                        pump.cancel()
            # 正常收尾（action=finish）：仍在工作的子蛋继续跑完，不打断它。

        # Recap: let the commander summarize completion (fallback to plain text).
        try:
            if final_summary:
                self.commander._record("assistant", "text", text=final_summary)
                yield {"k": "say_break"}
                yield {"k": "say", "text": final_summary}
            else:
                res_txt = "\n\n".join(
                    f"[{title}] 指令：{msg}\n结果：{reply[:600]}"
                    for (title, msg, reply) in all_results
                )
                recap_prompt = (
                    f"用户指令：{text}\n\n各蛋执行情况：\n{res_txt}\n\n"
                    "请用 2-4 句中文总结该指令的完成情况与后续建议。"
                )
                async for tok in self.commander.run(recap_prompt, summary):
                    yield {"k": "say", "text": tok}
        except Exception as ex:  # pragma: no cover
            yield {"k": "say", "text": f"（总结失败：{ex}）"}
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
                parts.append(f"{eid} 空闲(会话{st['len']}字{pend})")
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
            lines.append(f"- id={eid} 「{st['title']}」{state} | 会话长度 "
                         f"{st['len']} 字，{growth}{extra}")
        if notified:
            lines.append("【待处理的完成通知（子蛋已完成并通知了你）】")
            for n in notified[-3:]:
                lines.append(f"- 蛋「{n.get('title')}」的结论："
                             f"{(n.get('reply') or '')[:300]}")
        return "\n".join(lines)

    def _role_hint(self, eid):
        """A short, distinctive keyword for an egg (from its title).

        "正方辩手-运气是资产" -> "正方";  "裁判角色" -> "裁判".  Used to spot
        that one egg's instruction REFERENCES another egg of the same round.
        """
        title = self.eggs[eid].title if eid in self.eggs else str(eid)
        head = re.split(r"[-—_、,，\s:：()（）]", title)[0] or title
        return head[:2] if len(head) >= 2 else head

    def _references(self, msg, keyword, egg_id=None):
        """True when ``msg`` tells this egg to READ another egg's conclusion.

        Two ways the commander points at a peer:
          * explicitly — "正方(e1)与反方(e2)" (strong signal);
          * by role name next to a consuming verb — "综合正反双方的论证".
        A mere mention ("预判反方的主要反驳") is NOT a dependency, and a
        self-reference ("你是正方辩手，…") never is either.
        """
        if not keyword:
            return False
        if egg_id:
            # An EXPLICIT id reference is a strong signal on its own:
            #   "正方(e1)"  "正方 e1"  "来自 e1"  "蛋e1"  "e1 的论证"
            pat = r"[（(]\s*" + re.escape(egg_id) + r"\s*[)）]"
            if re.search(pat, msg):
                return True
            near = r"[^。！？；\n]{0,24}?"
            if re.search(re.escape(keyword) + near + re.escape(egg_id), msg):
                return True
            if re.search(re.escape(egg_id) + near + re.escape(keyword), msg):
                return True
            if re.search(r"(蛋|源|来自|由)\s*[:：]?\s*" + re.escape(egg_id), msg):
                return True
        idx = msg.find(keyword)
        if idx < 0:
            return False
        m = re.match(r"\s*你是([^，。：:,\n]{0,14})", msg)
        if m and keyword in m.group(1):
            return False      # it is talking about ITSELF
        # The consuming verb must live in the SAME clause: "需综合正方与反方
        # 双方论证" covers both roles, while "预判反方的主要反驳" does not.
        clause = self._clause(msg, idx)
        return any(v in clause for v in _CONSUME_VERBS)

    def _clause(self, msg, idx):
        """The sentence-ish span of ``msg`` that contains character ``idx``."""
        start = 0
        for sep in "。！？；\n":
            p = msg.rfind(sep, 0, idx)
            if p != -1:
                start = max(start, p + 1)
        end = len(msg)
        for sep in "。！？；\n":
            p = msg.find(sep, idx)
            if p != -1:
                end = min(end, p)
        return msg[start:end]

    def _is_consumer(self, eid):
        """True when this egg's ROLE consumes other eggs' output."""
        egg = self.eggs.get(eid)
        if not egg:
            return False
        blob = f"{egg.title} {getattr(egg, 'prompt', '')}"
        return any(k in blob for k in _CONSUMER_ROLES)

    def _resolve_dependencies(self, tasks, boxes=None):
        """Fill in the dependencies the commander did not declare.

        Small local models often ignore the ``depends_on`` field, but they DO
        name the other roles in the instruction: the judge says "综合正方与反方
        双方论证".  Such a task obviously needs that egg's conclusion first,
        so we infer the edge from the text — otherwise everything runs in
        parallel and the judge has nothing to judge.

        The same reading also AUTHORIZES the box: an egg that is told to
        weigh another egg's conclusion must be allowed to read it.
        """
        boxes = boxes or {}
        ids = {t["egg_id"] for t in tasks}
        deps = {t["egg_id"]: set(t.get("depends_on") or []) for t in tasks}
        for t in tasks:
            msg = t.get("message") or ""
            refs = set()
            for eid in self.eggs:
                if eid == t["egg_id"]:
                    continue
                if self._references(msg, self._role_hint(eid), eid):
                    refs.add(eid)
            for u in tasks:
                if u["egg_id"] in refs:
                    deps[t["egg_id"]].add(u["egg_id"])
            # STRUCTURAL rule: a judge/aggregator must not run alongside the
            # producers it is supposed to weigh — it waits for their results.
            if self._is_consumer(t["egg_id"]):
                for u in tasks:
                    if u["egg_id"] == t["egg_id"]:
                        continue
                    if not self._is_consumer(u["egg_id"]):
                        deps[t["egg_id"]].add(u["egg_id"])
            # 授权读取它被要求参考的、已经有结论的蛋
            deliver = set(t.get("deliver") or [])
            for r in refs:
                if r in boxes and r != t["egg_id"]:
                    deliver.add(r)
            t["deliver"] = sorted(deliver)
        # Break dependency cycles (A waits for B and B waits for A): the LATER
        # task keeps waiting for the earlier one, the earlier one just runs.
        for i, t in enumerate(tasks):
            for j, u in enumerate(tasks):
                if i == j:
                    continue
                if (u["egg_id"] in deps[t["egg_id"]]
                        and t["egg_id"] in deps[u["egg_id"]]
                        and j > i):
                    deps[t["egg_id"]].discard(u["egg_id"])
        for t in tasks:
            t["depends_on"] = [d for d in deps[t["egg_id"]] if d in ids]
        return deps

    def _split_by_dependency(self, tasks, boxes=None):
        """Split this round's tasks into (runnable, deferred).

        A task whose ``depends_on`` names ANOTHER task of the SAME round has
        to wait — its dependency has not produced anything yet. Such a task is
        deferred (it is NOT run this round; the commander picks it up in the
        next round, once the dependency's conclusion is in its box).

        ``depends_on`` entries that point at an egg with an ALREADY collected
        conclusion are satisfied and do not defer anything.
        """
        self._resolve_dependencies(tasks, boxes)
        ids = {t["egg_id"] for t in tasks}
        boxes = boxes or {}
        runnable, deferred = [], []
        for t in tasks:
            # A dependency that already has a conclusion (and is not part of
            # THIS round) is satisfied -> it does not defer the task.
            deps = [d for d in (t.get("depends_on") or [])
                    if d in ids and d != t["egg_id"] and d not in boxes]
            if deps:
                deferred.append(t)
            else:
                runnable.append(t)
        # Safety net: if EVERYTHING ended up deferred (a cycle slipped through),
        # let the first task run so the loop keeps moving.
        if tasks and not runnable:
            first = tasks[0]
            first["depends_on"] = []
            return [first], tasks[1:]
        return runnable, deferred

    # ---- background dispatch + completion notification ----
    def _start_runs(self, plan_steps, boxes, channels, inbox, events, running, bg):
        """Start this round's eggs in the BACKGROUND.

        Running the eggs in the background is what makes the commander's
        monitoring possible: the eggs keep streaming while the commander
        watches their conversation length / silence / duration.
        """
        started = 0
        for step in plan_steps:
            eid = step["egg_id"]
            egg = self.eggs.get(eid)
            if not egg or eid in running:
                continue
            events.put_nowait({"k": "instruct", "egg": egg.id,
                                "text": "🎯 总指挥指令：" + step["message"]})
            events.put_nowait({"k": "status", "egg": egg.id, "status": "running"})
            ctx = self._task_context(step, boxes, channels, inbox)
            q: asyncio.Queue = asyncio.Queue()
            pump = asyncio.create_task(_pump(egg.run(step["message"], context=ctx), q))
            rec = {"step": step, "tokens": [], "done": False, "reply": "",
                   "pump": pump, "egg": egg.id}
            running[eid] = rec
            consumer = asyncio.create_task(self._consume_run(eid, q, rec, events))
            bg.extend([pump, consumer])
            started += 1
        return started

    async def _consume_run(self, eid, q, rec, events):
        """Drain one egg's token stream into the shared UI event queue."""
        try:
            while True:
                item = await q.get()
                if item is _PUMP_DONE:
                    rec["reply"] = "".join(rec["tokens"])
                    rec["done"] = True
                    return
                if isinstance(item, tuple) and item and item[0] == "__err__":
                    txt = f"\n（执行出错：{item[1]}）"
                    rec["tokens"].append(txt)
                    await events.put({"k": "egg", "egg": eid, "text": txt})
                    continue
                rec["tokens"].append(item)
                await events.put({"k": "egg", "egg": eid, "text": item})
        except asyncio.CancelledError:
            raise
        except Exception as ex:  # pragma: no cover
            rec["reply"] = "".join(rec["tokens"])
            rec["done"] = True
            await events.put({"k": "egg", "egg": eid,
                              "text": f"\n（执行出错：{ex}）"})

    def _file_result(self, rec, boxes, channels, cmd_hist, inbox, notified,
                     all_results):
        """File a finished egg's conclusion and NOTIFY the right listeners.

        Notifications:
          * the 总指挥 is ALWAYS notified (its own conversation history gets a
            "子蛋完成" turn, so the next decision is informed by it);
          * every egg listed in the task's ``notify`` field receives the
            conclusion too (it waits in that egg's inbox and is handed over as
            context the next time that egg runs).
        """
        step = rec["step"]
        eid = step["egg_id"]
        reply = rec["reply"] or ""
        egg = self.eggs.get(eid)
        if not egg:
            return
        all_results.append((egg.title, step["message"], reply))
        boxes[eid] = {"result": reply}
        # Commander <-> egg channel: the instruction and the reply.
        ch = channels.setdefault(("commander", eid), [])
        ch.append({"role": "user", "text": step["message"]})
        ch.append({"role": "assistant", "text": reply})
        # Any egg this one was AUTHORIZED to hear from keeps a peer-to-peer
        # channel too, so the delivered context persists.
        for src in (step.get("deliver") or []):
            pch = channels.setdefault((src, eid), [])
            pch.append({"role": "user",
                        "text": f"来自蛋「{self.eggs[src].title if src in self.eggs else src}」的结论"})
            pch.append({"role": "assistant", "text": reply})
        # ---- 完成通知 ----
        cmd_hist.append({
            "role": "user",
            "text": f"【子蛋「{egg.title}」完成】\n结论：{reply[:2000]}",
        })
        notified.append({"egg": eid, "title": egg.title, "reply": reply[:2000]})
        # 总指挥指定的其它蛋：结论进入它们的收件箱，下次运行时作为上下文带上。
        for tgt in (step.get("notify") or []):
            if tgt not in self.eggs or tgt == eid:
                continue
            msg = (f"【完成通知：蛋「{egg.title}」已完成并通知你】\n"
                   f"{reply[:2000]}")
            inbox.setdefault(tgt, []).append(msg)
            nch = channels.setdefault((eid, tgt), [])
            nch.append({"role": "user",
                        "text": f"来自蛋「{egg.title}」的完成通知"})
            nch.append({"role": "assistant", "text": reply})

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
                "3b) 【时序/依赖】有先后关系的任务必须用 depends_on 声明依赖的蛋 id，"
                "系统会自动把它推迟到依赖完成后的下一轮执行。例如辩论：正方与反方"
                "互不依赖，同轮并行；裁判依赖双方，depends_on 写 [\"正方id\", \"反方id\"]，"
                "它不会在本轮跑（此时双方还没产出，裁判无从裁决），会在双方完成后"
                "的下一轮才被派活。裁判/评审/汇总/整合/对比这类角色一律都要写"
                "depends_on，不要和被它依赖的角色放在同一轮。\n"
                "4) 子蛋之间【默认不能互相通信】。只有你（总指挥）有权让某个结论从一只蛋"
                "传给另一只蛋：若本轮某只蛋需要参考另一只蛋的结论，在该任务的 deliver "
                "字段里写上来源蛋的 id，即为【授权转发】；未授权的结论只留在你这里由你裁决。\n"
                "5) 每个任务的 notify 字段指定【这只蛋完成后要通知谁】（填子蛋 id 数组，"
                "可为画布上任一子蛋；你总会收到通知）。例如你希望 A 的结论直接提醒 B "
                "继续干活，就写 notify=[\"B_id\"]。\n"
                "6) 只读工具（view_file / list_directory / search_directory / find_file）"
                "【只能】在下面列出的目录里查看。【不要】去翻其它目录，也不要反复探索"
                "目录结构；一次最多调用 2 次工具，除非确实需要核实子蛋产出。\n"
                "7) 通讯需要携带上下文：user+assistant 的历史消息即可（暂不需要 tool 和 thinking）。\n"
                "8) 任务完成时 action='finish'，并在 summary 里给出最终总结；"
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
                deferred_desc = ("【上一轮被推迟的任务（依赖已满足，该派活了）】\n"
                                 + "\n".join(
                                     f"- 蛋「{d.get('title')}」(id={d.get('egg_id')}) "
                                     f"等待 {', '.join(d.get('depends_on') or [])} 的产出，"
                                     f"本轮请派活给它"
                                     for d in deferred)
                                 + "\n\n")
            prompt = (
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
            for attempt in (1, 2):
                try:
                    async with Agent(build_agent_config(
                        system_instructions=sys_instr, capabilities=caps,
                        response_schema=CommanderDecision,
                        workspaces=(workspaces or None),
                    )) as planner:
                        resp = await planner.chat(prompt)
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
                        async for chunk in resp.chunks:
                            self._record_planner_chunk(chunk)
                            if sink is not None:
                                shown = self._planner_chunk_text(chunk)
                                if shown:
                                    sink(shown)
                            if isinstance(chunk, types.ToolCall):
                                nm = getattr(chunk, "name", None)
                                if nm == "finish" and raw_json is None:
                                    raw_json = self._finish_payload(chunk)
                        data = await resp.structured_output()
                        if data is None and raw_json:
                            # The SDK could not read the envelope, but the model
                            # DID produce a valid decision -> use it.
                            print("[commander-decide] structured_output empty, "
                                  "using the raw finish payload instead")
                            data = self._parse_decision_text(raw_json)
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
            out_tasks = []
            for t in raw_tasks:
                if isinstance(t, dict):
                    eid = t.get("egg_id"); message = t.get("message")
                    deliver = t.get("deliver") or []
                    notify = t.get("notify") or []
                    depends = t.get("depends_on") or []
                else:
                    eid = t.egg_id; message = t.message
                    deliver = getattr(t, "deliver", []) or []
                    notify = getattr(t, "notify", []) or []
                    depends = getattr(t, "depends_on", []) or []
                if eid in valid and message:
                    # only deliver boxes that actually exist
                    out_tasks.append({
                        "egg_id": eid, "message": message,
                        "deliver": [d for d in deliver
                                    if d in boxes and d != eid],
                        # 完成后通知谁：必须是画布上真实存在的另一只蛋
                        "notify": [n for n in notify
                                   if n in self.eggs and n != eid],
                        # 依赖：只保留画布上真实存在的蛋（自己不能依赖自己）
                        "depends_on": [d for d in depends
                                       if d in self.eggs and d != eid],
                    })
            if action != "dispatch":
                return {"action": "finish", "tasks": [],
                        "summary": summary_txt, "note": note}
            # NOTE: an empty task list is NOT a reason to finish — the commander
            # stays in its loop and is asked again (the idle counter in
            # commander_run is the only guard against a no-op loop).
            return {"action": "dispatch", "tasks": out_tasks,
                    "summary": summary_txt, "note": note}
        except Exception as ex:  # pragma: no cover
            print(f"[commander-decide] failed, using fallback: {ex}")
            return self._fake_commander_decide(text, targets, boxes, round_no)

    def _fake_commander_decide(self, text, targets, boxes, round_no):
        """Fallback when the planner produced nothing usable.

        The model may well be available — what usually fails is the
        STRUCTURED-OUTPUT envelope (small local models often answer with
        prose the SDK cannot parse). So we do NOT claim "no model"; we say the
        decision could not be parsed, and dispatch CONSERVATIVELY: only the
        first target, so a fallback never invents a wrong parallel structure.
        """
        if boxes or round_no > 1:
            return {"action": "finish", "tasks": [],
                    "summary": "（总指挥本轮未能给出可解析的决策，指挥结束。）",
                    "note": ""}
        if not targets:
            return {"action": "finish", "tasks": [],
                    "summary": "（没有可指挥的子蛋。）", "note": ""}
        return {"action": "dispatch",
                "tasks": [{"egg_id": targets[0], "message": text, "deliver": [],
                           "notify": [], "depends_on": []}],
                "summary": "", "note": "（总指挥输出无法解析，先派一个目标蛋）"}

    def _planner_chunk_text(self, chunk):
        """Render a planner chunk as display text for the commander panel.

        The planner runs in STRUCTURED-OUTPUT mode, so its decision travels in
        a special tool call named ``finish`` whose args carry the raw JSON —
        that must NOT be shown to the user (they should read the commander's
        own words, not the JSON envelope). Everything else — its text, its
        thinking, its real tool calls and their results — is shown.
        """
        if isinstance(chunk, types.Text):
            return chunk.text or ""
        if isinstance(chunk, types.Thought):
            return f"💭 {chunk.text or ''}"
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

    def _parse_decision_text(self, text):
        """Parse a commander decision out of raw model output.

        Used when the SDK's structured-output parser rejects the envelope.
        Tolerates: a bare JSON object, JSON wrapped in prose/markdown fences,
        and a JSON object embedded in a longer answer.
        """
        if not text:
            return None
        text = text.strip()
        # 1) the whole thing may be JSON
        for candidate in (text,):
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
        # 2) ```json ... ``` fences
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
        if m:
            try:
                obj = json.loads(m.group(1))
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
        # 3) the first balanced {...} block in the text
        start = text.find("{")
        while start != -1:
            depth = 0
            in_str = False
            esc = False
            for i in range(start, len(text)):
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
                        try:
                            obj = json.loads(text[start:i + 1])
                            if isinstance(obj, dict):
                                return obj
                        except Exception:
                            break
            start = text.find("{", start + 1)
        return None

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

    def _task_context(self, task, boxes, channels, inbox=None):
        """Build the context handed to a driven egg.

        Contains ONLY user+assistant history (no tool calls / thinking):
          * the commander<->egg channel history (what the commander and this egg
            have already said to each other),
          * the AUTHORIZED peer boxes this egg is allowed to read,
          * the peer channel history for each authorized source,
          * the completion notifications other eggs sent to THIS egg (its
            inbox) — they are consumed here, so each notification is handed
            over exactly once.
        """
        eid = task["egg_id"]
        blocks = []
        ch = channels.get(("commander", eid))
        if ch:
            blocks.append(
                "【你与总指挥的通讯历史（user+assistant）】\n"
                + self._fmt_channel(ch))
        for src in task.get("deliver") or []:
            box = boxes.get(src)
            if not box:
                continue
            stitle = self.eggs[src].title if src in self.eggs else src
            blk = (f"【总指挥授权转交：蛋「{stitle}」的结论】\n"
                   f"{box.get('result', '')}")
            pch = channels.get((src, eid))
            if pch:
                blk += ("\n\n【与该蛋的通讯历史（user+assistant）】\n"
                        + self._fmt_channel(pch))
            blocks.append(blk)
        if inbox:
            # Consume the notifications addressed to this egg.
            pend = inbox.pop(eid, None)
            if pend:
                blocks.append("【其它蛋发给你的完成通知（请据此继续）】\n"
                              + "\n\n".join(pend))
        if not blocks:
            return None
        return "\n\n".join(blocks)

    # ---- one-click run of selected eggs (parallel / serial by graph) ----
    def _levels(self, targets):
        """Group selected eggs into dependency levels using edges.

        An edge ``from -> to`` means ``to`` depends on ``from`` (``to`` receives
        ``from``'s output), so ``from`` must run first. Eggs with no dependency
        among the selection share a level and can run in PARALLEL; dependent
        eggs land in later levels (SERIAL). Returns (levels, preds).
        """
        tset = set(targets)
        preds = {t: [] for t in targets}
        for ed in self.edges:
            a, b = ed["from"], ed["to"]
            if a in tset and b in tset and a != b and a not in preds[b]:
                preds[b].append(a)
        levels, done, remaining = [], set(), set(targets)
        while remaining:
            ready = [t for t in remaining if set(preds[t]) <= done]
            if not ready:  # cycle fallback: run the rest together
                ready = list(remaining)
            levels.append(ready)
            done |= set(ready)
            remaining -= set(ready)
        return levels, preds

    def _run_message(self, egg, upstream_ids, outputs):
        """Build an egg's run prompt.

        A serial (downstream) egg inherits each upstream egg's full CONVERSATION
        HISTORY (commands, replies, thoughts, tool calls/results) — not just a
        truncated reply — so it can build on what upstream actually did. Falls
        back to the captured output text if an upstream has no transcript.
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
        """Run the selected eggs level by level: each level's eggs run in
        PARALLEL (streamed concurrently), later levels run after earlier ones
        (SERIAL), feeding upstream outputs downstream.

        Emits status events so the UI shows 计划中/运行中/空闲; a running egg can
        be interrupted (its inner run task is cancelled) without aborting the
        plan — the next level still runs after this one finishes."""
        targets = [t for t in (targets or []) if t in self.eggs]
        if not targets:
            yield {"k": "end"}
            return
        levels, preds = self._levels(targets)
        outputs = {}
        total = len(levels)
        # Everything selected is queued ("计划中") until its level starts.
        for t in targets:
            yield {"k": "status", "egg": t, "status": "planned"}
        for li, level in enumerate(levels, 1):
            names = "、".join(self.eggs[t].title for t in level)
            mode = "并行" if len(level) > 1 else "串行"
            yield {"k": "say", "text": f"▶ 第 {li}/{total} 层（{mode}）：{names}\n"}
            for t in level:
                yield {"k": "status", "egg": t, "status": "running"}
            q: asyncio.Queue = asyncio.Queue()

            async def worker(eid, q=q):
                egg = self.eggs[eid]
                msg = self._run_message(egg, preds.get(eid, []), outputs)
                await q.put({"k": "instruct", "egg": eid,
                             "text": "▶ 运行指令：" + msg})
                buf = []

                async def drain():
                    gen = egg.run(msg)
                    try:
                        async for tok in gen:
                            buf.append(tok)
                            await q.put({"k": "egg", "egg": eid, "text": tok})
                    finally:
                        # aclose so interrupting this egg releases its locks now.
                        try:
                            await gen.aclose()
                        except Exception:  # pragma: no cover
                            pass

                # egg.run() runs on its own task, so interrupt() cancels only it.
                inner = asyncio.create_task(drain())
                try:
                    await inner
                except asyncio.CancelledError:
                    await q.put({"k": "egg", "egg": eid, "text": "\n⏹ 已打断。"})
                except Exception as ex:  # pragma: no cover
                    await q.put({"k": "egg", "egg": eid,
                                 "text": f"\n（执行出错：{ex}）"})
                outputs[eid] = "".join(buf)
                await q.put({"k": "status", "egg": eid, "status": "idle"})
                await q.put({"k": "end_egg", "egg": eid})

            tasks = [asyncio.create_task(worker(t)) for t in level]
            finished = 0
            while finished < len(level):
                ev = await q.get()
                if ev.get("k") == "end_egg":
                    finished += 1
                yield ev
            await asyncio.gather(*tasks)
        yield {"k": "end"}

    def _commander_context(self, targets, mode=None):
        """Graph summary PLUS the selected eggs' historical conversations.

        Lets the Commander (总指挥) inherit what each selected egg has actually
        done/decided — not just its prompt — when advising or driving them.
        ``mode`` overrides the canvas-wide history mode; the commander uses
        "brief" (user+assistant only, no tool calls / thinking).
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
