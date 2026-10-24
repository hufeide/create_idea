"""Canvas state: eggs, edges, and the operations that mutate them.

This is the only module that knows about the Antigravity SDK's higher-level
features:
  * ``generate`` uses structured output (response_schema) to plan eggs.
  * ``combine`` wires the source eggs in as SubagentConfigs of the new egg,
    so the combined egg can delegate to its parts.
"""
import asyncio

import pydantic

from .egg import EggAgent, reasoning_capture_state
from .commander import Commander
from .config import build_agent_config, BACKEND

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


class CommanderStep(pydantic.BaseModel):
    egg_id: str
    message: str


class CommanderPlan(pydantic.BaseModel):
    steps: list[CommanderStep]


class Canvas:
    def __init__(self):
        self.eggs = {}
        self.edges = []
        self._id = 0
        self.commander = Commander()
        # How much conversation history is shared with downstream eggs / the
        # commander when they inherit upstream/selected eggs' sessions:
        #   "brief" (default) = 用户+助手 text only
        #   "full"            = everything (thoughts, tool calls/results, …)
        self.history_mode = "brief"

    # ---- id helpers ----
    def _new_id(self, p="e"):
        self._id += 1
        return f"{p}{self._id}"

    # ---- lifecycle ----
    async def start(self):
        # Eggs and the commander connect to the model lazily on first use, so the
        # canvas always boots cleanly even with no model file / server present.
        await self._seed()

    async def shutdown(self):
        for e in self.eggs.values():
            await e.stop()
        await self.commander.stop()

    def view(self):
        return {
            "eggs": [e.spec() for e in self.eggs.values()],
            "edges": [dict(e) for e in self.edges],
            "history_mode": self.history_mode,
            "reasoning": reasoning_capture_state(),
        }

    # ---- spawning ----
    async def _spawn(self, title, prompt, type_, x, y, subagents=None, parent=None):
        eid = self._new_id()
        egg = EggAgent(eid, title, prompt, type_, (x, y), subagents=subagents, parent=parent)
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
        if x is None:
            x = 120 + (len(self.eggs) % 4) * 60
        if y is None:
            y = 120 + (len(self.eggs) % 4) * 60
        return await self._spawn(title, prompt, "module", x, y)

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

    async def delete_edge(self, edge_id):
        self.edges = [ed for ed in self.edges if ed["id"] != edge_id]

    async def add_edge(self, from_, to):
        if from_ == to or from_ not in self.eggs or to not in self.eggs:
            return False
        if any(ed["from"] == from_ and ed["to"] == to for ed in self.edges):
            return False
        self.edges.append({"id": self._new_id("edge"), "from": from_, "to": to})
        return True

    async def move(self, eid, x, y):
        e = self.eggs.get(eid)
        if e:
            e.x, e.y = x, y

    # ---- commander ----
    async def commander_run(self, text, targets=None):
        """Commander either advises (no targets) or actively DRIVES the selected
        eggs: it plans per-egg messages, sends them (streaming each egg's real
        reply), then recaps how the user's instruction was completed.

        Yields UI events:
          {"k":"say","text":..}      -> stream into the commander chat
          {"k":"instruct","egg":id,"text":..} -> push the instruction into an egg
          {"k":"egg","egg":id,"text":..}      -> stream an egg's reply tokens
          {"k":"status","egg":id,"status":..} -> egg status (planned/running/idle)
          {"k":"end_egg","egg":id}            -> finalize an egg's stream
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

        # Control mode: plan, drive each egg, then recap.
        plan = await self._commander_plan(text, summary, targets)
        if not plan:
            plan = [{"egg_id": t, "message": text} for t in targets]
        plan_txt = "\n".join(
            f"→ [{self.eggs[s['egg_id']].title}] {s['message']}"
            for s in plan if s["egg_id"] in self.eggs
        )
        narration = f"我来指挥 {len(targets)} 个蛋完成该指令，共 {len(plan)} 步：\n{plan_txt}\n\n"
        self.commander._record("system", "plan", text=plan_txt)
        self.commander._record("assistant", "text", text=narration)
        yield {"k": "say", "text": narration}

        # Eggs in the plan are queued ("计划中"); they run one-by-one (serial),
        # each waiting for the previous to finish.
        for s in plan:
            if s["egg_id"] in self.eggs:
                yield {"k": "status", "egg": s["egg_id"], "status": "planned"}

        results = []
        for s in plan:
            egg = self.eggs.get(s["egg_id"])
            if not egg:
                continue
            yield {"k": "instruct", "egg": egg.id,
                   "text": "🎯 总指挥指令：" + s["message"]}
            yield {"k": "status", "egg": egg.id, "status": "running"}
            collected = []
            q: asyncio.Queue = asyncio.Queue()
            pump = asyncio.create_task(_pump(egg.run(s["message"]), q))
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
            yield {"k": "status", "egg": egg.id, "status": "idle"}
            yield {"k": "end_egg", "egg": egg.id}
            results.append((egg.title, s["message"], "".join(collected)))

        # Recap: let the commander summarize completion (fallback to plain text).
        try:
            res_txt = "\n\n".join(
                f"[{title}] 指令：{msg}\n结果：{reply[:600]}"
                for (title, msg, reply) in results
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

    async def _commander_plan(self, text, summary, targets):
        """Ask the commander LLM to break the instruction into per-egg messages."""
        if not SDK_OK:
            return self._fake_commander_plan(text, targets)
        try:
            target_desc = "\n".join(
                f"- id={self.eggs[t].id} 标题={self.eggs[t].title} "
                f"职责={self.eggs[t].prompt}"
                for t in targets
            )
            sys_instr = types.CustomSystemInstructions(text=(
                "你是总指挥，负责把用户指令拆解为对各个子蛋的具体消息。"
                "只能使用下面给定的子蛋 id；为每个需要行动的蛋写一条明确、可执行的中文消息；"
                "无需行动的蛋不要列入。返回结构化 steps（egg_id + message）。"
            ))
            caps = types.CapabilitiesConfig(
                enabled_tools=[], enable_subagents=False,
                agent_behavior=types.AgentBehavior.MINIMAL,
            )
            prompt = (
                f"可选子蛋：\n{target_desc}\n\n画布状态：\n{summary}\n\n"
                f"用户指令：{text}\n请给出执行步骤。"
            )
            async with Agent(build_agent_config(
                system_instructions=sys_instr, capabilities=caps,
                response_schema=CommanderPlan,
            )) as planner:
                resp = await planner.chat(prompt)
                data = await resp.structured_output()
            steps = data.get("steps") if isinstance(data, dict) else getattr(data, "steps", [])
            valid = set(targets)
            out = []
            for s in steps:
                eid = s.get("egg_id") if isinstance(s, dict) else s.egg_id
                message = s.get("message") if isinstance(s, dict) else s.message
                if eid in valid and message:
                    out.append({"egg_id": eid, "message": message})
            return out
        except Exception as ex:  # pragma: no cover
            print(f"[commander-plan] failed, using fallback: {ex}")
            return self._fake_commander_plan(text, targets)

    def _fake_commander_plan(self, text, targets):
        # No model / planner failed: broadcast the instruction to every target.
        return [{"egg_id": t, "message": text} for t in targets]

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

    def _commander_context(self, targets):
        """Graph summary PLUS the selected eggs' historical conversations.

        Lets the Commander (总指挥) inherit what each selected egg has actually
        done/decided — not just its prompt — when advising or driving them.
        """
        base = self._graph_summary(targets[0] if targets else None)
        if not targets:
            return base
        blocks = []
        for t in targets:
            e = self.eggs.get(t)
            if not e:
                continue
            hist = e.history_text(mode=self.history_mode, limit=6000).strip()
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
