"""Temporary: dependency inference from the real model's debate wording."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.graph import Canvas


class E:
    def __init__(self, eid, title):
        self.id = eid
        self.title = title
        self.prompt = "p"
        self.type = "module"


c = Canvas()
for eid, t in (("e1", "正方辩手-运气是资产"),
               ("e2", "反方辩手-运气非资产"),
               ("e3", "裁判角色")):
    c.eggs[eid] = E(eid, t)

# The exact wording the real model produced for the debate round.
tasks = [
    {"egg_id": "e1",
     "message": "你是正方辩手，主张『运气是一种资产』。请给出你的核心论证："
                "从哲学与经济学角度说明运气为何应被视为一种可持有、可增值/贬值的"
                "财富形态，并回应潜在的批评。"},
    {"egg_id": "e2",
     "message": "你是反方辩手，主张『运气不是资产』。请给出你的核心论证：以经济学"
                "与会计学标准（资产需受控、可预期收益、具有交换价值等）说明运气为何"
                "不符合资产的定义，并回应潜在的批评。"},
    {"egg_id": "e3",
     "message": "你是裁判，需综合正方（运气可增值财富形态）与反方（不符合资产定义）"
                "双方论证，最终给出『运气是不是一种资产』的裁决结论，并说明理由。"},
]
runnable, deferred = c._split_by_dependency(tasks)
print("runnable :", [t["egg_id"] for t in runnable])
print("deferred :", [(t["egg_id"], t.get("depends_on")) for t in deferred])
assert [t["egg_id"] for t in runnable] == ["e1", "e2"], runnable
assert [t["egg_id"] for t in deferred] == ["e3"], deferred
assert set(deferred[0]["depends_on"]) == {"e1", "e2"}, deferred[0]
print("PASS: 裁判被推迟到下一轮，且依赖 = 正方 + 反方")

# Self-reference must NOT create a dependency (正方 talks about 正方).
t2 = [{"egg_id": "e1", "message": "你是正方辩手，主张运气是一种资产，请论证。"},
      {"egg_id": "e2", "message": "你是反方辩手，请反驳。"}]
r2, d2 = c._split_by_dependency(t2)
assert not d2 and len(r2) == 2, (r2, d2)
print("PASS: 自称不产生依赖（双方仍可并行）")

# A cycle must not deadlock everything.
t3 = [{"egg_id": "e1", "message": "你是正方，请结合反方的论证来回应。"},
      {"egg_id": "e2", "message": "你是反方，请结合正方的论证来回应。"}]
r3, d3 = c._split_by_dependency(t3)
assert len(r3) >= 1, (r3, d3)
print("PASS: 互相引用不会死锁，runnable =", [t["egg_id"] for t in r3])
