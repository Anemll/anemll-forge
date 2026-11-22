"""Causal fixed-anchor screening of match length and feedback. Input/output contain scalar counts only."""
from collections import defaultdict
import json
from pathlib import Path
import sys

BUCKETS=(6,8,12,16,24,32,48)
def bucket(n):
    return max(x for x in BUCKETS if x<=n)

def run(requests, table, policy):
    groups=defaultdict(lambda: [0,0.,0,0])
    for q in requests:
        need=6; ref=None; total_ms=None; draft_ms=None
        group=q["group"]
        # Request-scoped prediction error. Update only after an actual lookup choice.
        errors={"short": [], "long": []}; pending=[]; prev_end=None
        for c in q["rounds"]:
            length,ntrue,lookup,draft,ms=c[:5]
            pos=c[5] if len(c)>5 else 0
            if prev_end!=pos: pending.clear()
            if policy=="shadow_3":
                waiting=[]
                for when,key,error in sorted(pending,key=lambda x:x[0]):
                    if when<=pos:
                        acc=errors[key];acc.append(error);del acc[:-3]
                    else: waiting.append((when,key,error))
                pending=waiting
            prev_end=pos+ntrue
            regime="short" if length<16 else "long"
            state=errors[regime]
            use=False
            if length>=need and policy!="off":
                prior=table[str(bucket(length))]
                if policy.startswith("feedback") or policy=="shadow_3":
                    # Use only the last three lookup rounds in the same length regime.
                    bias=sum(state)/(float(policy.split("_")[1])+len(state))
                    pred=max(1.,min(8.,prior+bias))
                else:
                    pred=prior
                use=length>=16 if ref is None else pred/max(total_ms-draft_ms,1)>1.25*ref
                if policy=="ctx" and pos<32768 and length<48: use=False
                if policy=="recent_floor" and len(state)>=2 and sum(state[-2:]) < -3 and length<32:
                    use=False
            if use:
                tokens=lookup+1; elapsed=max(ms-draft,1)
                state.append(tokens-prior)
                del state[:-3]
                if lookup<2: need=min(48,need*2)
                if lookup==7: need=max(6,need//2)
            else:
                if policy=="shadow_3" and length>=need:
                    # No outcome is exposed to the policy before its first mismatch or 7th match.
                    pending.append((pos+min(7,lookup+1),regime,lookup+1-prior))
                tokens=ntrue; elapsed=ms
                now=ntrue/max(ms,1)
                ref=now if ref is None else .8*ref+.2*now
                total_ms=ms if total_ms is None else .8*total_ms+.2*ms
                draft_ms=draft if draft_ms is None else .8*draft_ms+.2*draft
            for name in (group,"all"):
                v=groups[name]; v[0]+=tokens; v[1]+=elapsed; v[2]+=use; v[3]+=1
    return {k:{"tokens":v[0],"cycle_ms":round(v[1],3),"lookups":v[2],"rounds":v[3],"tps":1000*v[0]/v[1]} for k,v in groups.items()}

def score(requests,table):
    results={}
    for pol in ("off","yield","feedback_1","feedback_3","recent_floor","shadow_3","ctx"):
        v=run(requests,table,pol)
        for k,g in v.items():
            g["vs_off_pct"]=100*(g["tps"]/(results["off"][k]["tps"] if results else g["tps"])-1)
        results[pol]=v
    return results

if __name__=="__main__":
    requests=json.loads(Path(sys.argv[1]).read_text())
    table=json.loads(Path(sys.argv[2]).read_text())["sampled"]
    print(json.dumps(score(requests,table),indent=2))
