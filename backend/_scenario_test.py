import asyncio, os, sys, time, re, shutil, tempfile
sys.path.insert(0, '/home/administator/workspace/create_idea')
import backend.graph as G

TIMINGS = []  # (eid, 'start'/'end', monotonic)

def log(eid, kind):
    TIMINGS.append((eid, kind, time.monotonic()))

class FakeEgg:
    def __init__(self, eid, title, worktree, prompt=None, type_="egg"):
        self.id = eid; self.title = title; self.worktree = worktree
        self.prompt = prompt or ("职责：%s" % title); self.type = type_
        self.busy = False; self.run_started_at = None
        self.last_activity = time.monotonic(); self.last_completion = None
        self.transcript = []; self.transcript_len = 0; self._run_mark = None; self._run_started = 0
        self._deliver_path = None
    def _record(self, *a, **k): pass
    def stop(self): pass
    def interrupt(self): pass
    def history_text(self, mode=None, limit=None):
        return ''
    def spec(self):
        return {'id': self.id, 'title': self.title, 'prompt': self.prompt,
                'type': self.type, 'x': 0, 'y': 0, 'parent': None,
                'conversation_id': None, 'running': self.busy}
    async def run(self, message, context=None):
        self.busy = True; self.run_started_at = time.monotonic()
        log(self.id, 'start')
        # simulate work
        await asyncio.sleep(0.15)
        self.last_activity = time.monotonic()
        # figure out deliverable filename
        m = re.search(r'\S+\.md', message)
        name = m.group(0) if m else (self.title + '.md')
        self._deliver_path = name
        try:
            os.makedirs(self.worktree, exist_ok=True)
            with open(os.path.join(self.worktree, name), 'w', encoding='utf-8') as f:
                f.write('结论 by %s: %s' % (self.title, message[:50]))
        except Exception: pass
        self.transcript_len += len(message)
        self.last_activity = time.monotonic(); self.last_completion = time.monotonic()
        self.busy = False
        log(self.id, 'end')
        yield '这是%s的结论。' % self.title

class Harness:
    def __init__(self, titles):
        self.canvas = G.Canvas()
        self.tmpdir = tempfile.mkdtemp()
        self.eggs = []
        for i, t in enumerate(titles):
            eid = 'e%d' % (i+1)
            wt = os.path.join(self.tmpdir, eid)
            os.makedirs(wt, exist_ok=True)
            eg = FakeEgg(eid, t, wt)
            self.canvas.eggs[eid] = eg
            self.eggs.append(eg)
        self.canvas._last_delivery = {}
        self.script = []
        self.canvas._commander_decide = self._scripted
    async def _scripted(self, *a, **k):
        if self.script:
            return self.script.pop(0)
        return {'action':'finish','tasks':[],'summary':'done','note':''}
    def decision(self, tasks, action='dispatch', note=''):
        self.script.append({'action':action,'tasks':tasks,'summary':note,'note':note})
    async def run(self, text, targets=None):
        tgt = targets or [e.id for e in self.eggs]
        events = []
        async for ev in self.canvas.commander_run(text, tgt):
            events.append(ev)
        return events
    def cleanup(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

def mk_tasks(h, specs):
    # specs: list of (eid, message, depends_on, deliver, notify)
    return [{'egg_id':s[0],'message':s[1],'depends_on':s[2] if len(s)>2 else [],
            'deliver':s[3] if len(s)>3 else [],'notify':s[4] if len(s)>4 else []}
            for s in specs]

def overlaps(eid1, eid2):
    s1=[t for t in TIMINGS if t[0]==eid1]; s2=[t for t in TIMINGS if t[0]==eid2]
    a=[t for t in s1 if t[1]=='start'][0][2]; b=[t for t in s1 if t[1]=='end'][0][2]
    c=[t for t in s2 if t[1]=='start'][0][2]; d=[t for t in s2 if t[1]=='end'][0][2]
    return a < d and c < b

async def scenario_parallel():
    TIMINGS.clear()
    h = Harness(['研究员A','研究员B','研究员C'])
    h.decision(mk_tasks(h, [
        ('e1','请产出研究报告.md，写明结论。'),
        ('e2','请产出研究报告.md，写明结论。'),
        ('e3','请产出研究报告.md，写明结论。')]))
    await h.run('并行研究三个主题')
    ok = overlaps('e1','e2') and overlaps('e1','e3') and overlaps('e2','e3')
    print('SCENARIO parallel: eggs ran concurrently =', ok)
    h.cleanup()
    return ok

async def scenario_dependent():
    TIMINGS.clear()
    h = Harness(['写手','评审','汇总'])
    h.decision(mk_tasks(h, [('e1','请产出初稿.md，写明内容。')]))
    h.decision(mk_tasks(h, [
        ('e2','请产出评审.md，依据初稿。',['e1']),
        ('e3','请产出汇总.md，综合初稿与评审。',['e1','e2'])]))
    await h.run('先写初稿，再评审，最后汇总')
    # e2 and e3 must start AFTER e1 ended
    e1end=[t for t in TIMINGS if t[0]=='e1' and t[1]=='end'][0][2]
    e2start=[t for t in TIMINGS if t[0]=='e2' and t[1]=='start'][0][2]
    e3start=[t for t in TIMINGS if t[0]=='e3' and t[1]=='start'][0][2]
    ok = e2start > e1end and e3start > e1end
    print('SCENARIO dependent: e2/e3 started after e1 finished =', ok)
    h.cleanup()
    return ok

async def scenario_empty_dispatch_recovery():
    TIMINGS.clear()
    h = Harness(['写手','评审'])
    # commander emits an EMPTY dispatch while e1 not done -> must not halt; forced dispatch kicks in
    h.decision([])
    h.decision([])
    events = await h.run('完成初稿与评审')
    # verify e1 actually got dispatched (forced) by checking it produced a deliverable / ran
    ran = any(t[0]=='e1' for t in TIMINGS)
    ok = ran
    print('SCENARIO empty-dispatch recovery: missing egg still dispatched =', ok)
    h.cleanup()
    return ok

async def scenario_idempotent():
    TIMINGS.clear()
    h = Harness(['写手','评审'])
    h.decision(mk_tasks(h, [('e1','请产出初稿.md，写明内容。')]))
    # after e1 done, commander re-dispatches e1 (completed) plus e2 -> e1 must be skipped
    h.decision(mk_tasks(h, [('e1','请产出初稿.md，再写一次。'),('e2','请产出评审.md。')]))
    await h.run('写初稿后评审，重复派写手应被跳过')
    # count e1 starts: should be exactly 1
    e1starts = sum(1 for t in TIMINGS if t[0]=='e1' and t[1]=='start')
    e2starts = sum(1 for t in TIMINGS if t[0]=='e2' and t[1]=='start')
    ok = (e1starts == 1) and (e2starts == 1)
    print('SCENARIO idempotent: completed egg not re-run, new egg ran =', ok, '(e1starts=%d e2starts=%d)'%(e1starts,e2starts))
    h.cleanup()
    return ok

async def scenario_no_delivery_flag():
    # unit test of the no_delivery detection without running the loop
    c = G.Canvas()
    eid='e7'
    e = FakeEgg(eid,'卡住的蛋','/tmp/none')
    e.busy = True
    e.run_started_at = time.monotonic() - 500  # started 500s ago
    e.last_activity = time.monotonic() - 200   # silent 200s
    c.eggs[eid]=e
    c._last_delivery = {eid: {'delivered': False, 'detail':'本轮运行中，尚未交付', 'path':None,'bytes':0,'chars':0}}
    watch = c._watch_eggs([eid])
    flag = watch[eid]['no_delivery']
    alerts = c._watch_alerts(watch, set(), [])
    no_deliv_alert = any('长期运行未交付' in a for a in alerts)
    ok = bool(flag) and no_deliv_alert
    print('SCENARIO no_delivery flag: no_delivery=',bool(flag),' alert=',no_deliv_alert)
    return ok

async def scenario_fake_recover():
    c = G.Canvas()
    c.eggs = {'e1': FakeEgg('e1','a','/tmp/x'), 'e2': FakeEgg('e2','b','/tmp/y')}
    # simulate a parse failure recovery: boxes empty (nothing delivered yet)
    d = c._fake_commander_decide('开始', ['e1','e2'], {}, 1)
    ok = (d['action']=='dispatch' and d['tasks']==[])
    print('SCENARIO fake-commander recovery: action=',d['action'],' (no hard halt) =',ok)
    return ok

async def scenario_mixed():
    # One decision carries BOTH independent (parallel) and dependent tasks.
    TIMINGS.clear()
    h = Harness(['调研A', '调研B', '汇总'])
    h.decision(mk_tasks(h, [
        ('e1', '请产出报告A.md，写明结论。'),
        ('e2', '请产出报告B.md，写明结论。'),
        ('e3', '请产出汇总.md，综合A与B。', ['e1', 'e2'])]))
    await h.run('并行调研A、B，再汇总')
    a = overlaps('e1', 'e2')
    e1end = [t for t in TIMINGS if t[0] == 'e1' and t[1] == 'end'][0][2]
    e2end = [t for t in TIMINGS if t[0] == 'e2' and t[1] == 'end'][0][2]
    e3start = [t for t in TIMINGS if t[0] == 'e3' and t[1] == 'start'][0][2]
    ok = a and e3start > e1end and e3start > e2end
    print('SCENARIO mixed (parallel+dependent in one decision): e1/e2 concurrent & e3 after both =', ok)
    h.cleanup()
    return ok

async def scenario_finish_forced():
    # Commander says finish, but a selected target never got a task -> forced.
    TIMINGS.clear()
    h = Harness(['写手', '评审'])
    h.decision(mk_tasks(h, [('e1', '请产出初稿.md，写明内容。')]))
    h.decision([], action='finish')
    await h.run('先写初稿，然后结束')
    ran1 = any(t[0] == 'e1' for t in TIMINGS)
    ran2 = any(t[0] == 'e2' for t in TIMINGS)
    ok = ran1 and ran2
    print('SCENARIO finish-forced: unfinished selected target still dispatched =', ok)
    h.cleanup()
    return ok

async def main():
    results = []
    results.append(await scenario_parallel())
    results.append(await scenario_dependent())
    results.append(await scenario_empty_dispatch_recovery())
    results.append(await scenario_idempotent())
    results.append(await scenario_no_delivery_flag())
    results.append(await scenario_fake_recover())
    results.append(await scenario_mixed())
    results.append(await scenario_finish_forced())
    print('\nALL PASS =', all(results))

asyncio.run(main())
