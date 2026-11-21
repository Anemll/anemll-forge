"""21 Nov 2026 (review fixes): Tree.call's split path (the 96K entry) with a fake GPU driver. The plan must be built
under the chain mask, the runs must see the tree mask, and the chain mask must be back on exit: normally, when the
first-use plan bind fails, when a run fails, and under python -O (no assert side effects).
    PY -B tree_mask_unit.py TREE_VERIFY.py [optimize 0|1]"""
import sys
from types import SimpleNamespace
import numpy as np

src, opt = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0
ns = {"__name__": "tree_verify_under_test"}
exec(compile(open(src, "rb").read(), src, "exec", optimize=opt), ns)
fails = 0

def case(name, bad_plan=False, bad_run=False):
    global fails
    mask, at_plan, at_run = [(0, 0)], [], []
    def set_tree(lo, hi):
        mask.append((lo, hi)); return 0
    def run():
        at_run.append(mask[-1])
        if bad_run:
            raise RuntimeError("injected run failure")
    def plan(par):
        at_plan.append(mask[-1])
        if bad_plan:
            raise ValueError("injected bind failure")
        P = SimpleNamespace(run=run)
        return [(P, P, P)], SimpleNamespace(run=run)
    mb = SimpleNamespace(pos=0, ctx=98304, fit=lambda n: True, cap=lambda c: c, par=0, _flip=lambda: None,
                         stats={"calls": 0}, logits=np.zeros((8, 4)))
    g = SimpleNamespace(dec_set_tree=set_tree, dec_run=lambda *a: 1.0)
    split = SimpleNamespace(ka=False, use=lambda p: True, _sync_kv=lambda: None, g=g,
                            files=[{"Ha": 0, "Hba": 0}], st=None, stats={"steps": 0, "gpu_ms": 0.0, "wall_ms": 0.0})
    t = SimpleNamespace(mb=mb, _inputs=lambda *a: None, split_ok=True, _split_plan=plan, split=split)
    err = None
    try:
        ns["Tree"].call(t, None, None, None, None, 5, 9)
    except Exception as e:  # noqa: BLE001
        err = type(e).__name__
    ok = at_plan == [(0, 0)] and mask[-1] == (0, 0) and (bad_plan or all(m == (5, 9) for m in at_run) and at_run)
    if not bad_plan and not bad_run:
        ok = ok and split.stats["steps"] == 1 and split.stats["gpu_ms"] > 0 and err is None
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'} {name} (-O{opt}): mask at plan {at_plan}, during runs {sorted(set(at_run))}, "
          f"on exit {mask[-1]}, error {err}, steps {split.stats['steps']} gpu_ms {split.stats['gpu_ms']}")

case("normal")
case("plan bind fails", bad_plan=True)
case("a run fails", bad_run=True)
print("ALL PASS" if not fails else f"{fails} FAILED"); sys.exit(1 if fails else 0)
