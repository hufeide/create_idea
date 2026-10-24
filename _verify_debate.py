"""Temporary: real-model debate round 1 -> does the judge get deferred?"""
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
    print("R1:", d1.get("action"), flush=True)
    for t in (d1.get("tasks") or []):
        print("   task", t["egg_id"], "depends_on=", t.get("depends_on"),
              flush=True)
    runnable, deferred = c._split_by_dependency(d1.get("tasks") or [])
    print("R1 runnable:", [t["egg_id"] for t in runnable],
          "deferred:", [t["egg_id"] for t in deferred], flush=True)


asyncio.run(main())
