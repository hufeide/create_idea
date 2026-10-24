"""Temporary: real two-round debate — judge must be deferred, then dispatched."""
import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.graph import Canvas


class FakeEgg:
    def __init__(self, eid, title):
        self.id = eid
        self.title = title
        self.prompt = "参与「运气是不是一种资产」的辩论，给出本方论证。"
        self.type = "module"
        self.worktree = Path("/tmp/_repro_wt")
        self.busy = False
        self.transcript = []
        self.last_activity = time.monotonic()
        self.run_started_at = None
        self.last_completion = None

    @property
    def transcript_len(self):
        return 0

    def history_text(self, mode="brief", limit=None):
        return ""


async def main():
    Path("/tmp/_repro_wt").mkdir(parents=True, exist_ok=True)
    c = Canvas()
    for eid, t in (("e1", "正方辩手-运气是资产"),
                   ("e2", "反方辩手-运气非资产"),
                   ("e3", "裁判角色")):
        c.eggs[eid] = FakeEgg(eid, t)

    boxes, channels, cmd_hist = {}, {}, [
        {"role": "user", "text": "开始一轮辩论：运气是不是一种资产？"}]

    d1 = await c._commander_decide(
        "开始一轮辩论：运气是不是一种资产？", "", ["e1", "e2", "e3"],
        boxes, channels, cmd_hist, 1)
    tasks = d1.get("tasks") or []
    runnable, deferred = c._split_by_dependency(tasks, boxes)
    print("R1 action:", d1.get("action"))
    for t in (d1.get("tasks") or []):
        print("  MSG", t["egg_id"], "=>", t["message"][:400])
        print("       deliver=", t.get("deliver"))
    print("R1 runnable :", [t["egg_id"] for t in runnable])
    print("R1 deferred :", [(t["egg_id"], t["depends_on"]) for t in deferred])
    assert "e3" not in [t["egg_id"] for t in runnable], "judge ran too early!"
    assert "e3" in [t["egg_id"] for t in deferred]

    # The two debaters finish -> their conclusions are in the boxes.
    for t in runnable:
        eid = t["egg_id"]
        reply = f"【{c.eggs[eid].title}的论证】" + ("论证与反驳的详细论述。" * 40)
        boxes[eid] = {"result": reply}
        ch = channels.setdefault(("commander", eid), [])
        ch.append({"role": "user", "text": t["message"]})
        ch.append({"role": "assistant", "text": reply})
        cmd_hist.append({"role": "user",
                         "text": f"【子蛋「{eid}」完成】\n结论：{reply[:2000]}"})
    cmd_hist.append({"role": "user", "text": "【调度】蛋「裁判角色」依赖本轮其它蛋的产出，已推迟到下一轮执行；请在它们完成后再派活。"})

    print("--- ROUND 2 ---", flush=True)
    d2 = await c._commander_decide(
        "开始一轮辩论：运气是不是一种资产？", "", ["e1", "e2", "e3"],
        boxes, channels, cmd_hist, 2)
    print("R2 action:", d2.get("action"))
    for t in (d2.get("tasks") or []):
        print("   task", t["egg_id"], "deliver=", t.get("deliver"),
              "depends_on=", t.get("depends_on"))
    print("--- R2 transcript tail ---")
    for r in c.commander.transcript[-6:]:
        if r.get("kind") == "tool_call":
            print("[tool_call]", r.get("name"), repr(r.get("args"))[:900])
        else:
            print("[%s]" % r.get("kind"), repr(r.get("text"))[:400])
    ids2 = [t["egg_id"] for t in (d2.get("tasks") or [])]
    assert "e3" in ids2, f"judge was NOT dispatched in round 2: {ids2}"
    judge = [t for t in d2["tasks"] if t["egg_id"] == "e3"][0]
    assert set(judge.get("deliver") or []) == {"e1", "e2"}, judge
    print("PASS: 裁判在第 2 轮被派活，且获授权读取正方+反方的结论")


asyncio.run(main())
