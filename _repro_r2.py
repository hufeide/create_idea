"""Temporary: reproduce the round-2 commander failure with a realistic payload."""
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

    async def run(self, text, context=None):
        self.busy = True
        try:
            for _ in range(5):
                yield "本方观点与论据。" * 40
                await asyncio.sleep(0.01)
        finally:
            self.busy = False


async def main():
    Path("/tmp/_repro_wt").mkdir(parents=True, exist_ok=True)
    c = Canvas()
    for eid, t in (("e1", "正方辩手-运气是资产"),
                   ("e2", "反方辩手-运气非资产"),
                   ("e3", "裁判角色")):
        c.eggs[eid] = FakeEgg(eid, t)

    boxes = {}
    channels = {}
    cmd_hist = [{"role": "user", "text": "开始一轮辩论：运气是不是一种资产？"}]

    print("=== ROUND 1 ===", flush=True)
    d1 = await c._commander_decide(
        "开始一轮辩论：运气是不是一种资产？", "", ["e1", "e2", "e3"],
        boxes, channels, cmd_hist, 1)
    print("R1 decision:", d1, flush=True)
    print("=== R1 commander transcript ===", flush=True)
    for r in c.commander.transcript:
        k = r.get("kind")
        if k == "text":
            print("[text]", repr(r.get("text"))[:1500])
        elif k == "tool_call":
            print("[tool_call]", r.get("name"), repr(r.get("args"))[:400])
        elif k == "thought":
            print("[thought]", repr(r.get("text"))[:300])
        else:
            print("[%s]" % k, repr(r.get("text"))[:200])

    # Simulate the three eggs finishing (long replies -> boxes + channels).
    for t in (d1.get("tasks") or []):
        eid = t["egg_id"]
        reply = "【结论】" + ("论证与反驳的详细论述。" * 60)
        boxes[eid] = {"result": reply}
        ch = channels.setdefault(("commander", eid), [])
        ch.append({"role": "user", "text": t["message"]})
        ch.append({"role": "assistant", "text": reply})
        cmd_hist.append({"role": "user",
                         "text": f"【子蛋「{eid}」完成】\n结论：{reply[:2000]}"})

    print("=== ROUND 2 ===", flush=True)
    d2 = await c._commander_decide(
        "开始一轮辩论：运气是不是一种资产？", "", ["e1", "e2", "e3"],
        boxes, channels, cmd_hist, 2)
    print("R2 decision:", d2, flush=True)


asyncio.run(main())
