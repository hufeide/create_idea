"""Temporary: dependency inference on the REAL model's debate wording."""
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

# Verbatim output of the real model (round 1).
tasks = [
    {"egg_id": "e1",
     "message": "你是正方辩手（运气是资产）。请给出完整的立论论证：明确定义“资产”的"
                "核心要素，并论证运气满足这些要素——可从风险对冲、稀缺性、可资本化或"
                "带来超额回报等角度切入。输出应包含：本方核心论点、支持逻辑链、以及预判"
                "反方的主要反驳以便回应。"},
    {"egg_id": "e2",
     "message": "你是反方辩手（运气非资产）。请给出完整的立论论证：剖析“资产”的本质"
                "属性（如受控性、可预期回报、法律产权界定），并论证运气不具备这些属性"
                "而更接近风险或机会，不能归为资产。输出应包含：本方核心论点、支持逻辑链、"
                "以及对正方‘风险对冲’类说法的反驳预案。"},
    {"egg_id": "e3",
     "message": "你是裁判角色，负责对正方(e1)与反方(e2)的立论进行评判并给出初步立场"
                "倾向。请阅读你被授权的结论后，比较两方的论证强度、逻辑自洽性、可证伪性"
                "与现实解释力，最终给出偏向哪一方的裁决及理由。注意：你的评判依赖正反双方"
                "输出，因此仅在本轮这两只蛋完成后才会被执行。"},
]
runnable, deferred = c._split_by_dependency(tasks, {})
print("runnable:", [t["egg_id"] for t in runnable])
print("deferred:", [(t["egg_id"], sorted(t["depends_on"])) for t in deferred])
assert [t["egg_id"] for t in runnable] == ["e1", "e2"], runnable
assert set(deferred[0]["depends_on"]) == {"e1", "e2"}, deferred[0]
print("PASS: 辩手互提但不算依赖；裁判依赖双方 -> 正确")

# The judge variant WITHOUT explicit ids but with a consuming verb.
tasks2 = [
    {"egg_id": "e1", "message": "你是正方辩手，请给出本方论证。"},
    {"egg_id": "e2", "message": "你是反方辩手，请给出本方论证。"},
    {"egg_id": "e3", "message": "你是裁判，请综合正方与反方双方论证后给出裁决结论。"},
]
r2, d2 = c._split_by_dependency(tasks2, {})
assert [t["egg_id"] for t in r2] == ["e1", "e2"], r2
assert set(d2[0]["depends_on"]) == {"e1", "e2"}, d2[0]
print("PASS: 无显式 id 时，同句消费动词也能识别依赖")

# A dependency that already has a conclusion is SATISFIED -> not deferred.
tasks3 = [{"egg_id": "e3", "message": "你是裁判，请综合正方(e1)与反方(e2)的论证。"}]
r3, d3 = c._split_by_dependency(tasks3, {"e1": {"result": "x"},
                                           "e2": {"result": "y"}})
assert len(r3) == 1 and not d3, (r3, d3)
assert set(r3[0]["deliver"]) == {"e1", "e2"}, r3[0]
print("PASS: 已有结论的依赖不推迟，且自动授权读取（deliver）")
