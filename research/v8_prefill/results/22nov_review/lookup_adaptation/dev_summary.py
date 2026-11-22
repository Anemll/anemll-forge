"""Audit and score the exact observed round outcomes without using any token IDs."""
import json
from collections import defaultdict
from pathlib import Path
import sys
import replay

def read(run):
    run=Path(run); client=json.loads((run/"1_observe.json").read_text())
    if not client["valid"]: raise ValueError("incomplete observation")
    files=sorted((run/"observe_counts").glob("request_*.json"))
    if len(files)!=1+len(client["turns"]):raise ValueError("unexpected observation request count")
    requests=[]; excluded={}; count=0
    for r,f in zip(client["turns"],files[1:]):
        rows=json.loads(f.read_text())["rounds"]
        if not rows or r["lookup_cycles"]: raise ValueError("observer did not use tree8 as fallback")
        if len(rows)!=r["branched_rounds"]+r["chain_rounds"]: raise ValueError("observer round count differs from service")
        if any(not (0<=x[0]<=64 and 1<=x[1]<=8 and (x[2] is None or 0<=x[2]<=7)) for x in rows):
            raise ValueError("impossible round count")
        if sum(x[3] for x in rows)<=0:raise ValueError("missing drafter timing")
        group=band(r["prompt_tokens"]); store=[]
        # Keep labels only where seven subsequent output positions are visible within
        # an uninterrupted series of rounds. Withhold the final, possibly clipped call.
        end=None; after=None
        for row in rows[::-1]:
            length,tree,matches,draft,ms,pos=row
            if after is None or pos+tree!=after: end=pos
            eligible=end-pos>=7 and matches is not None
            if eligible and ms>draft>=0:
                store.append([length,tree,matches,draft,ms,pos])
            else:excluded[group]=excluded.get(group,0)+1
            after=pos
        count+=len(rows)
        requests.append({"group":group,"index":r["index"],"rounds":store[::-1]})
    return requests,{"recorded_rounds":count,"excluded_by_band":excluded,"retained":sum(len(r["rounds"]) for r in requests)}

def band(pos):
    return "8-16K" if pos<16384 else "16-32K" if pos<32768 else "32-57K"

def feedback_evidence(requests, table):
    # On the observer arm every proposal is declined. Reveal its outcome only when the
    # generated text has resolved the copy or mismatch, then compare the next opportunity.
    groups=defaultdict(list)
    for q in requests:
        history={"short":[],"long":[]}
        pending=[]
        prev_end=None
        for length,tree,matches,draft,wall,pos in q["rounds"]:
            if pos!=prev_end:
                pending=[]
            for when,key,residual in sorted(pending,key=lambda v:v[0]):
                if when<=pos:
                    history[key].append(residual)
                    del history[key][:-3]
            pending=[v for v in pending if v[0]>pos]
            prev_end=pos+tree
            if length<6:continue
            key="short" if length<16 else "long"
            prior=table[str(replay.bucket(length))]
            old=history[key]
            sign="insufficient history" if len(old)<2 else ("below prior" if sum(old)<0 else "at or above prior")
            net=matches+1-tree*(wall-draft)/wall
            for scope in ("all",q["group"]):
                groups[scope,key,sign].append((q["index"],matches+1,net))
            pending.append((pos+min(matches+1,7),key,matches+1-prior))
    return [{"context":g,"match_class":key,"recent_outcomes":sign,"rounds":len(vals),
             "distinct_turns":len({i for i,_,_ in vals}),"mean_copy_yield":sum(y for _,y,_ in vals)/len(vals),
             "mean_net_token_equivalent":sum(n for _,_,n in vals)/len(vals),
             "fraction_copy_beats_tree":sum(n>0 for _,_,n in vals)/len(vals)}
            for (g,key,sign),vals in sorted(groups.items())]

def main(run,root):
    data,meta=read(run); table=json.loads((root/"prior.json").read_text())["sampled"]
    report={"audit":meta,"results":replay.score(data,table),"request_count":len(data),
            "feedback_evidence":feedback_evidence(data,table)}
    for name,pred in (("odd",1),("even",0)):
        report[name]=replay.score([q for q in data if q["index"]%2==pred],table)
    thresh={}
    for g in ("8-16K","16-32K","32-57K","all"):
        for n in (6,8,12,16,24,32,48):
            sub=[r for q in data if g in (q["group"],"all") for r in q["rounds"] if replay.bucket(max(6,r[0]))==n and r[0]>=6]
            if sub:
                thresh[g,f"{n}"]=[len(sub),round(sum(r[2]+1 for r in sub)/len(sub),3),round(sum(r[1] for r in sub)/len(sub),3)]
    report["bucket_n_copy_tv_tree_tv"]={g:{n:v for (k,n),v in thresh.items() if g==k} for g in ("8-16K","16-32K","32-57K","all")}
    print(json.dumps(report,indent=2))
if __name__=="__main__": main(sys.argv[1],Path(__file__).parent)
