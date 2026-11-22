"""21 Nov 2026: unit check of LOOKUP_GATE=think:T:BASE (pipe/tree_verify.py): the reasoning-span test (_in_think, incremental
per request) and the gate's decisions. No model, no server.
    PY think_gate_unit.py"""
import sys
from types import SimpleNamespace

sys.path[:0] = ["/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe",
                "/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v02/scripts"]
import ngram_draft as NG  # noqa: E402
import tree_verify as TV  # noqa: E402

O, Cl = 248068, 248069
ok = True


def check(name, got, want):
    global ok
    ok &= got == want
    print(f"{'ok ' if got == want else 'BAD'} {name}: {got} (want {want})")


t = TV.Tree.__new__(TV.Tree)
t._thc = None; t._think_ids = None
# 1. thinking on: the prompt ends with <think>\n
h = NG.TokenHistory([1, 2, 3, O, 198])
eng = SimpleNamespace(tok=None, think_close=Cl, tokens=h)
check("prompt ends in <think>", t._in_think(eng, 5), True)
h.extend([10, 11, 12])
check("still reasoning (incremental)", t._in_think(eng, 7), True)
check("anchor is </think>", t._in_think(eng, Cl), False)
h.extend([Cl, 198])
check("after </think>", t._in_think(eng, 9), False)
# 2. thinking off: <think>\n\n</think>\n\n in the prompt
eng.tokens = NG.TokenHistory([1, O, 198, Cl, 198, 5])
check("new request, thinking off", t._in_think(eng, 9), False)
# 3. earlier turns closed, this turn open
eng.tokens = NG.TokenHistory([O, 4, Cl, 7, O, 4, Cl, 8, O, 198])
check("history of closed turns, open now", t._in_think(eng, 3), True)
# 4. anchor is <think>
eng.tokens = NG.TokenHistory([1, 2])
check("anchor is <think>", t._in_think(eng, O), True)

# gate decisions (yield references set: e(L) / (cyc - draft) vs 1.25 x yield)
t._cyc_ema, t._draft_ema, t._yield_ema = 120.0, 13.0, 0.030   # 3.6 tok per 120 ms cycle
for spec, L, think, want in (("think:24:yield:1.25", 20, True, False), ("think:24:yield:1.25", 30, True, True),
                             ("think:24:yield:1.25", 8, False, True), ("think:off:yield:1.25", 64, True, False),
                             ("think:off:ctx:32768:48:1.25", 64, False, True), ("think:off:ctx:32768:48:1.25", 20, False, False),
                             ("think:24:", 10, True, False), ("think:24:", 10, False, True), ("yield:1.25", 8, True, True)):
    TV.LOOKUP_GATE = spec
    check(f"gate {spec} L={L} think={think}", t._gate(L, 1.0, 10000, think), want)
t._yield_ema = 0.070   # the tree is doing well: lookup must beat it
TV.LOOKUP_GATE = "think:24:yield:1.25"
check("gate think:24:yield:1.25 L=30 think, strong tree", t._gate(30, 1.0, 10000, True), False)
print("ALL OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
