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
import time

import pydantic

from .egg import EggAgent, reasoning_capture_state
from .commander import Commander
from .config import build_agent_config, BACKEND, SESSIONS_DIR

try:  # pragma: no cover
    from google.antigravity import Agent, types, CapabilitiesConfig
    SDK_OK = True
except Exception:  # pragma: no cover
    SDK_OK = False


_PUMP_DONE = object()   # sentinel: an egg's run generator finished / was cancelled


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
        """Commander drives the selected eggs REACTIVELY, one round at a time.

        The commander no longer plans the whole serial/parallel graph up front.
        Every round it looks at what came back and decides the NEXT move:
          * one task   -> serial: that egg runs alone, its conclusion returns
          * many tasks -> parallel: several eggs run at once, conclusions return
                          together
        Each egg conclusion lands in a "box" (消息盒子) held by the commander.
        The commander may AUTHORIZE a box to be delivered to another egg — that
        authorization is the ONLY way two eggs talk to each other; an
        unauthorized message stays with the commander, who arbitrates it.
        All commander<->egg traffic carries the user+assistant conversation
        history (tool calls / thinking are left out for now).

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

        # ---- control mode: reactive, round by round ----
        # The commander/egg channels carry user+assistant history only.
        summary = self._commander_context(targets, mode="brief")
        boxes = {}        # egg_id -> {"result": str}   (the 消息盒子)
        channels = {}     # (src, dst) -> [{"role","text"}]  user+assistant only
        cmd_hist = []     # commander-side user+assistant history
        cmd_hist.append({"role": "user", "text": text})
        all_results = []  # (title, message, reply) for the final recap
        final_summary = ""
        max_rounds = 12
        round_no = 0

        while round_no < max_rounds:
            round_no += 1
            decision = await self._commander_decide(
                text, summary, targets, boxes, channels, cmd_hist, round_no)
            tasks = [t for t in (decision.get("tasks") or [])
                     if t.get("egg_id") in self.eggs]
            if decision.get("action") == "finish" or not tasks:
                final_summary = decision.get("summary") or ""
                break

            # --- this round's plan (1 task = serial, N tasks = parallel) ---
            mode = "parallel" if len(tasks) > 1 else "serial"
            plan_steps = [
                {"egg_id": t["egg_id"],
                 "title": self.eggs[t["egg_id"]].title,
                 "message": t["message"],
                 "deliver": t.get("deliver") or [],
                 "mode": mode}
                for t in tasks
            ]
            note = decision.get("note") or ""
            narration = (f"【第 {round_no} 轮 · {mode}】{note}\n" + "\n".join(
                f"→ [{s['title']}] {s['message']}"
                + (f"  (授权转交: {', '.join(s['deliver'])})" if s["deliver"] else "")
                for s in plan_steps))
            self.commander._record("system", "plan", text=narration)
            self.commander._record("assistant", "text", text=narration)
            cmd_hist.append({"role": "assistant", "text": narration})
            yield {"k": "say", "text": narration}
            yield {"k": "plan", "round": round_no, "steps": plan_steps}
            for s in plan_steps:
                yield {"k": "status", "egg": s["egg_id"], "status": "planned"}

            # ---- execute this round ----
            # A round with ONE task is serial (that egg alone, then its result
            # comes back for the commander's next decision); a round with SEVERAL
            # tasks runs them in parallel and returns all their results at once.
            # Either way the results are handed back to the commander, who then
            # decides the next round — no pre-baked multi-step graph.
            round_results = []
            if len(plan_steps) == 1:
                async for ev in self._run_task(plan_steps[0], boxes, channels,
                                              round_results):
                    yield ev
            else:
                async for ev in self._run_tasks_parallel(plan_steps, boxes,
                                                         channels, round_results):
                    yield ev

            # ---- file the results into boxes + channel history ----
            for r in round_results:
                eid, msg, reply = r["egg_id"], r["message"], r["reply"]
                egg = self.eggs.get(eid)
                if not egg:
                    continue
                all_results.append((egg.title, msg, reply))
                boxes[eid] = {"result": reply}
                # Commander <-> egg channel: the instruction and the reply.
                ch = channels.setdefault(("commander", eid), [])
                ch.append({"role": "user", "text": msg})
                ch.append({"role": "assistant", "text": reply})
                # Any egg this one was AUTHORIZED to hear from keeps a
                # peer-to-peer channel too, so the delivered context persists.
                for src in (r.get("deliver") or []):
                    pch = channels.setdefault((src, eid), [])
                    pch.append({"role": "user",
                                "text": f"来自蛋「{self.eggs[src].title if src in self.eggs else src}」的结论"})
                    pch.append({"role": "assistant", "text": reply})
                # Feed the result back to the commander as a user turn so its
                # next decision is informed by what actually came out.
                cmd_hist.append({
                    "role": "user",
                    "text": f"【子蛋「{egg.title}」完成】\n结论：{reply[:2000]}",
                })
                yield {"k": "status", "egg": eid, "status": "idle"}
                yield {"k": "end_egg", "egg": eid}

        # Recap: let the commander summarize completion (fallback to plain text).
        try:
            if final_summary:
                self.commander._record("assistant", "text", text=final_summary)
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

    async def _commander_decide(self, text, summary, targets, boxes, channels,
                                cmd_hist, round_no):
        """Ask the commander LLM for the NEXT round only (reactive adjudication).

        The commander sees:
          * the original instruction
          * the eggs it may drive (id / title / role)
          * the boxes collected so far (each egg's latest conclusion)
          * the user+assistant conversation history of its channels
        and answers with either a set of tasks to run NOW (1 = serial, N =
        parallel) or ``action="finish"`` when the instruction is complete.

        Each task may name ``deliver`` — the boxes the commander AUTHORIZES
        this egg to read. That is the only channel through which one egg's
        message reaches another egg; anything not delivered stays with the
        commander for arbitration.
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
            sys_instr = types.CustomSystemInstructions(text=(
                "你是总指挥（Commander），以【逐轮反应式】方式驱动若干子蛋完成用户指令。\n"
                "重要机制：你【不预先规划完整流程】。每一轮你只决定【下一步】做什么，"
                "等这一轮的结果回来后，再根据实际结果决定下一轮。\n"
                "规则：\n"
                "1) 一次只给本轮的指令。若本轮这些蛋互不依赖，可同时给多个（并行）；"
                "若需要等某个结果才能定下一步，就只给一个（串行）。\n"
                "2) 子蛋的结论会回传给你。你据此裁决：还需要谁继续做、还是已经完成。\n"
                "3) 子蛋之间【默认不能互相通信】。只有你（总指挥）有权让某个结论从一只蛋"
                "传给另一只蛋：若本轮某只蛋需要参考另一只蛋的结论，在该任务的 deliver "
                "字段里写上来源蛋的 id，即为【授权转发】；未授权的结论只留在你这里由你裁决。\n"
                "4) 通讯需要携带上下文：user+assistant 的历史消息即可（暂不需要 tool 和 thinking）。\n"
                "5) 任务完成时 action='finish'，并在 summary 里给出最终总结；"
                "否则 action='dispatch' 给出本轮 tasks。\n"
                "返回结构化 decision（action + tasks[egg_id, message, deliver] + "
                "summary + note）。note 用一句中文说明本轮意图。"
            ))
            caps = types.CapabilitiesConfig(
                enabled_tools=[], enable_subagents=False,
                agent_behavior=types.AgentBehavior.MINIMAL,
            )
            prompt = (
                f"可选子蛋：\n{target_desc}\n\n"
                f"已有的结论盒子：\n{boxes_desc}\n\n"
                f"通讯历史（user+assistant）：\n{chan_desc}\n\n"
                f"画布状态：\n{summary}\n\n"
                f"用户指令：{text}\n\n"
                f"这是第 {round_no} 轮。请只决定【本轮】的动作。"
            )
            async with Agent(build_agent_config(
                system_instructions=sys_instr, capabilities=caps,
                response_schema=CommanderDecision,
            )) as planner:
                resp = await planner.chat(prompt)
                data = await resp.structured_output()
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
                else:
                    eid = t.egg_id; message = t.message
                    deliver = getattr(t, "deliver", []) or []
                if eid in valid and message:
                    # only deliver boxes that actually exist
                    out_tasks.append({
                        "egg_id": eid, "message": message,
                        "deliver": [d for d in deliver
                                    if d in boxes and d != eid],
                    })
            if action != "dispatch" or not out_tasks:
                return {"action": "finish", "tasks": [],
                        "summary": summary_txt, "note": note}
            return {"action": "dispatch", "tasks": out_tasks,
                    "summary": summary_txt, "note": note}
        except Exception as ex:  # pragma: no cover
            print(f"[commander-decide] failed, using fallback: {ex}")
            return self._fake_commander_decide(text, targets, boxes, round_no)

    def _fake_commander_decide(self, text, targets, boxes, round_no):
        """No model / planner failed: broadcast once, then finish."""
        if boxes or round_no > 1:
            return {"action": "finish", "tasks": [],
                    "summary": "（无可用模型，无法继续指挥子蛋。）", "note": ""}
        return {"action": "dispatch",
                "tasks": [{"egg_id": t, "message": text, "deliver": []}
                          for t in targets],
                "summary": "", "note": ""}

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

    def _task_context(self, task, boxes, channels):
        """Build the context handed to a driven egg.

        Contains ONLY user+assistant history (no tool calls / thinking):
          * the commander<->egg channel history (what the commander and this egg
            have already said to each other),
          * the AUTHORIZED peer boxes this egg is allowed to read,
          * the peer channel history for each authorized source.
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
        if not blocks:
            return None
        return "\n\n".join(blocks)

    async def _run_task(self, task, boxes, channels, out):
        """Run ONE task (serial round). Streams the egg's reply tokens.

        The task's result is appended to ``out`` (a caller-owned list) so
        concurrent commander runs never clobber each other's state.
        """
        eid = task["egg_id"]
        egg = self.eggs.get(eid)
        if not egg:
            return
        yield {"k": "instruct", "egg": egg.id,
               "text": "🎯 总指挥指令：" + task["message"]}
        yield {"k": "status", "egg": egg.id, "status": "running"}
        ctx = self._task_context(task, boxes, channels)
        collected = []
        q: asyncio.Queue = asyncio.Queue()
        pump = asyncio.create_task(
            _pump(egg.run(task["message"], context=ctx), q))
        try:
            while True:
                item = await q.get()
                if item is _PUMP_DONE:
                    break
                if isinstance(item, tuple) and item and item[0] == "__err__":
                    yield {"k": "egg", "egg": egg.id,
                           "text": f"\n（执行出错：{item[1]}）"}
                    continue
                collected.append(item)
                yield {"k": "egg", "egg": egg.id, "text": item}
        finally:
            await pump
        out.append({
            "egg_id": eid, "message": task["message"],
            "reply": "".join(collected), "deliver": task.get("deliver") or [],
        })

    async def _run_tasks_parallel(self, tasks, boxes, channels, out):
        """Run SEVERAL tasks at once (parallel round); results come back together."""
        active = {}
        pumps = []
        for task in tasks:
            eid = task["egg_id"]
            egg = self.eggs.get(eid)
            if not egg:
                continue
            yield {"k": "instruct", "egg": egg.id,
                   "text": "🎯 总指挥指令：" + task["message"]}
            yield {"k": "status", "egg": egg.id, "status": "running"}
            q: asyncio.Queue = asyncio.Queue()
            ctx = self._task_context(task, boxes, channels)
            pumps.append(asyncio.create_task(
                _pump(egg.run(task["message"], context=ctx), q)))
            active[egg.id] = (q, task)
        if not active:
            return
        merge: asyncio.Queue = asyncio.Queue()
        async def _consumer(eid, q):
            while True:
                item = await q.get()
                if item is _PUMP_DONE:
                    await merge.put((eid, _PUMP_DONE))
                    return
                await merge.put((eid, item))
        consumers = [asyncio.create_task(_consumer(eid, q))
                     for eid, (q, _t) in active.items()]
        # NOTE: asyncio.gather returns a Future, not a coroutine — create_task()
        # would reject it ("a coroutine was expected"); ensure_future accepts both.
        monitor = asyncio.ensure_future(
            asyncio.gather(*pumps, *consumers, return_exceptions=True))
        collected = {eid: [] for eid in active}
        finished = 0
        try:
            while finished < len(active):
                eid, item = await merge.get()
                if item is _PUMP_DONE:
                    yield {"k": "status", "egg": eid, "status": "idle"}
                    yield {"k": "end_egg", "egg": eid}
                    finished += 1
                    continue
                if isinstance(item, tuple) and item and item[0] == "__err__":
                    yield {"k": "egg", "egg": eid,
                           "text": f"\n（执行出错：{item[1]}）"}
                    continue
                collected[eid].append(item)
                yield {"k": "egg", "egg": eid, "text": item}
        finally:
            await monitor
        for eid, (_q, task) in active.items():
            out.append({
                "egg_id": eid, "message": task["message"],
                "reply": "".join(collected[eid]),
                "deliver": task.get("deliver") or [],
            })

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
