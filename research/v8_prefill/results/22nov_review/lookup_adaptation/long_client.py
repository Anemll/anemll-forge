"""Send synthetic fixtures sequentially, writing only counts, timestamps, arm metadata and SHA digests."""
import hashlib
import json
import re
import sys
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import window_bench as bench
def run(fixture, output, seed, power):
    f=Path(fixture)
    doc=json.loads(f.read_text())
    if "source_files" in doc:
        from holdout import materialize
        tasks=materialize(doc)
    else:
        tasks=doc["rungs"]["long"]
    if len(tasks)<10: raise ValueError("incomplete long-session fixture")
    digest=hashlib.sha256(f.read_bytes()).hexdigest()
    out={"fixture_sha":digest,"seed":seed,"expected_turns":len(tasks),"turns":[],"valid":False}
    log=bench.LOG; ino=log.stat().st_ino; pos=log.stat().st_size
    def ready():
        if log.stat().st_ino!=ino: raise RuntimeError("service log rotated")
        with log.open("rb") as s:
            s.seek(pos); return s.read()
    if bench.START.search(log.read_bytes()): raise RuntimeError("foreign request before fresh arm")
    def send(q,cap,temp,think,sd):
        nonlocal pos
        if bench.START.search(ready()): raise RuntimeError("foreign service traffic")
        r=bench.request(q,cap,temp,think,sd)
        interval=r.pop("_power_window",None)
        if interval:
            r["decode_power"]=power.window(*interval)
        deadline=time.monotonic()+5
        while True:
            block=ready()
            a=bench.START.findall(block); b=list(bench.END.finditer(block))
            if len(a)>1 or len(b)>1: raise RuntimeError("unexpected service traffic")
            if a and b: break
            if time.monotonic()>deadline: raise RuntimeError("incomplete server markers")
            time.sleep(.05)
        end=b[0]
        n,cached,generated=map(int,end.groups())
        if generated!=r["completion_tokens"]: raise RuntimeError("server generation mismatch")
        m=bench.METRIC.search(end.group())
        if not m: raise RuntimeError("missing timing")
        r.update(internal_prompt=n, internal_cached=cached, cycle_ms=float(m[4]),draft_ms=float(m[5]),tv=float(m[2]))
        look=bench.LOOK.search(end.group());r["lookup_cycles"]=int(look[1]) if look else 0
        if look:
            r["lookup_tokens_per_verify"]=float(look[3])
        tree=bench.TREE.search(block)
        if tree:
            r["tree_rounds"],r["branched_rounds"],r["chain_rounds"]=map(int,tree.groups())
        pos+=len(block)
        return r
    out["warmup"]=send({"messages":[{"role":"user","content":"Say OK."}]},8,0,False,None)
    for i,q in enumerate(tasks):
        r=send(q,512,1.,True,seed+i)
        if r["completion_tokens"]<1 or r["stream_decode_s"] is None: raise RuntimeError("empty turn")
        r["index"]=i;out["turns"].append(r);bench.save(output,out)
        print(f"turn {i+1}/{len(tasks)} complete, context {r['prompt_tokens']}, generated {r['completion_tokens']}",flush=True)
    out["valid"]=True;bench.save(output,out)

if __name__=="__main__":
    sys.path.insert(0,"/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/scripts")
    from m6_compare_bench import Power
    sampler=Power(1000)
    try:
        run(sys.argv[1],Path(sys.argv[2]),int(sys.argv[3]),sampler)
    finally:
        sampler.stop()
