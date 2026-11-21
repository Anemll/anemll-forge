"""21 Nov 2026 (review fixes): V8Engine._restore is all or nothing. CPU only: the named methods of v8_server.py run
on fakes (as results/22nov_review/cache_checks.py does). A failure after the base restore (the KV write-back of a
'sysk' restore, the ring write-back of a 'system' restore) must leave a clean cold state: no tokens, model reset,
drafter reset, the failed snapshot dropped, and the next prefill reusing nothing.
    PY -B restore_fault_unit.py V8_SERVER.py BASE_QWEN38_SERVER.py"""
import ast, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np

def selected(path, name, methods, env):
    tree = ast.parse(Path(path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    assert {n.name for n in cls.body} == set(methods), {n.name for n in cls.body}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), env)
    return env[name]

logs = []
rt = SimpleNamespace(log=logs.append)
env = {"np": np, "S": rt, "time": time, "re": __import__("re"), "os": __import__("os"), "RING_SNAP": "fix",
       "RING_KEYS": ("system",), "COLD_SCRUB": set()}
rt.Engine = selected(sys.argv[2], "Engine", ["_snapshot", "_restore"], env)
V8 = selected(sys.argv[1], "V8Engine", ["_snapshot", "_ring_save", "_restore", "_ring_restore", "_restore_cold", "_cold_scrub"], env)
fails = 0

def fixture():
    e = object.__new__(V8)
    st = {"resets": 0}
    def reset(**k): st["resets"] += 1
    e.model = SimpleNamespace(snapshot=lambda: {}, restore=lambda s: None, fit=lambda n: True, hi=0, reset=reset)
    e.v2, e.snaps, e.fed, e.v8 = True, {}, [], None
    e.drafter = SimpleNamespace(ring=[(np.zeros((1, 4, 2)), np.zeros((1, 4, 2)))], slot_pos=np.full(4, -1), pending=[])
    e.drafter.reset = lambda: st.__setitem__("dreset", True)
    return e, st

def check(name, ok, msg=""):
    global fails
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'} {name} {msg}")

# 1. 'sysk' restore whose KV write-back fails
for ring in ("empty", "nonempty"):
    e, st = fixture()
    if ring == "nonempty":
        e.drafter.slot_pos[:2] = [0, 1]
    e._snapshot("system", ["A"] * 12)
    e.snaps["sysk"] = {**e.snaps.pop("system"), "kv": {"owner": "A"}}
    def boom(n, put=None): raise RuntimeError("injected KV write-back failure")
    e._kv_rows = boom
    e.fed = ["B"] * 5
    e._restore(e.snaps["sysk"])
    check(f"sysk + failed KV write-back ({ring} ring) -> cold", e.fed == [] and "sysk" not in e.snaps and st["resets"] == 1
          and st.get("dreset"), f"(fed {len(e.fed)}, snaps {sorted(e.snaps)}, model resets {st['resets']})")
# 2. 'sysk' restore that succeeds (empty ring, the reviewer's case)
e, st = fixture()
e._snapshot("system", ["A"] * 6)
e.snaps["sysk"] = {**e.snaps.pop("system"), "kv": {"owner": "A"}}
owner = ["B"]
e._kv_rows = lambda n, put=None: owner.__setitem__(0, put["owner"])
e._restore(e.snaps["sysk"])
check("sysk, empty ring, success", owner == ["A"] and e.fed == ["A"] * 6 and "system" in e.snaps and "sysk" not in e.snaps)
# 3. 'system' restore whose ring write-back fails
e, st = fixture()
e.drafter.slot_pos[:2] = [0, 1]
e._snapshot("system", ["A"] * 12)
e.snaps["system"]["draft_ring"] = (np.array([0, 1]), [])   # a corrupt copy: the write-back fails
real = e._ring_restore
def bad(snap, cur): raise ValueError("injected ring failure")
e._ring_restore = bad
e._restore(e.snaps["system"])
check("system + failed ring write-back -> cold", e.fed == [] and "system" not in e.snaps and st["resets"] == 1,
      f"(fed {len(e.fed)}, snaps {sorted(e.snaps)})")
print("logged:", [l for l in logs if "not restored" in l][:2])
print("ALL PASS" if not fails else f"{fails} FAILED"); sys.exit(1 if fails else 0)
