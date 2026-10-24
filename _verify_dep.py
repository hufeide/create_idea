"""Temporary: verify dependency split + raw-JSON fallback parsing."""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.graph import Canvas

c = Canvas()

# --- 1) dependency split: the judge must NOT run in the same round ---
tasks = [
    {"egg_id": "e1", "message": "正方论证", "depends_on": []},
    {"egg_id": "e2", "message": "反方论证", "depends_on": []},
    {"egg_id": "e3", "message": "裁判裁决", "depends_on": ["e1", "e2"]},
]
runnable, deferred = c._split_by_dependency(tasks)
assert [t["egg_id"] for t in runnable] == ["e1", "e2"], runnable
assert [t["egg_id"] for t in deferred] == ["e3"], deferred
print("PASS 1: 依赖本轮其它蛋的任务被推迟（裁判不进本轮）")

# --- 2) a dependency on an ALREADY finished egg does not defer ---
tasks2 = [{"egg_id": "e3", "message": "裁判", "depends_on": ["e1"]}]
runnable2, deferred2 = c._split_by_dependency(tasks2)
assert len(runnable2) == 1 and not deferred2, (runnable2, deferred2)
print("PASS 2: 依赖已有结论的蛋不推迟")

# --- 3) raw JSON fallback parsing (structured_output returned None) ---
cases = [
    '{"action":"dispatch","tasks":[{"egg_id":"e1","message":"m"}],"summary":"","note":""}',
    '好的，下面是决策：\n```json\n{"action":"finish","tasks":[],"summary":"done","note":"n"}\n```\n',
    '这是我的判断 {"action":"dispatch","tasks":[{"egg_id":"e2","message":"go"}],"summary":"s","note":"n"} 完毕',
]
for raw in cases:
    obj = c._parse_decision_text(raw)
    assert obj and obj.get("action") in ("dispatch", "finish"), raw
print("PASS 3: 三种形态的原始 JSON 都能兜底解析")

# --- 4) finish tool payload extraction ---
class FakeChunk:
    name = "finish"
    args = {"output_string": '{"action":"dispatch","tasks":[]}'}
assert c._finish_payload(FakeChunk()) == '{"action":"dispatch","tasks":[]}'
class FakeChunk2:
    name = "finish"
    args = '{"output_string": "{\\"action\\":\\"finish\\"}"}'
assert c._finish_payload(FakeChunk2()) == '{"action":"finish"}'
print("PASS 4: finish 工具的 JSON 载荷能取回")

# --- 5) fallback must NOT invent a parallel structure ---
c.eggs["e1"] = type("E", (), {"id": "e1", "title": "A", "prompt": "p", "type": "m"})()
c.eggs["e2"] = type("E", (), {"id": "e2", "title": "B", "prompt": "p", "type": "m"})()
fb = c._fake_commander_decide("go", ["e1", "e2"], {}, 1)
assert len(fb["tasks"]) == 1 and fb["tasks"][0]["egg_id"] == "e1", fb
assert "无可用模型" not in fb.get("summary", ""), fb
fb2 = c._fake_commander_decide("go", ["e1", "e2"], {"e1": {"result": "x"}}, 2)
assert fb2["action"] == "finish" and "无可用模型" not in fb2["summary"], fb2
print("PASS 5: 兜底只派一个目标 + 文案不再谎称『无可用模型』")
