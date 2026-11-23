"""Retrospective CPU replay. Run with the M6 research Python. Emits aggregate JSON."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

repo = Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8/research/v8_prefill")
volume = Path("/Volumes/SSD4TB/anemll-forge-research")
pipe = Path("/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe")
sys.path.insert(0, str(pipe))
from tree_verify import build_tree, is_chain

# Map the original BF16 weights read-only. Only convert rows used by the selector.
selector = Path("/Users/anemll/Models/anemll-forge-qwen3.8-27B-192K/drafter/all3/selector.safetensors")
with selector.open("rb") as f:
    length = int.from_bytes(f.read(8), "little")
    header = json.loads(f.read(length))
def codebook(name):
    spec = header["candidate_selector." + name + "_codebook"]
    assert spec["dtype"] == "BF16"
    arr = np.memmap(selector, dtype="<u2", mode="r", offset=8 + length + spec["data_offsets"][0], shape=tuple(spec["shape"]))
    return lambda idx: (arr[idx].astype(np.uint32) << 16).view(np.float32)

pc, sc = codebook("predecessor"), codebook("successor")
def root_scores(z, i):
    return z["unary"][i, 0] + sc(z["cand"][i, 0]) @ (pc(int(z["anchor"][i])) * z["hp"][i, 0].astype(np.float32))

def tree_walk(parent, tok, truth):
    rows = {0}
    depth = np.zeros(len(tok), dtype=int)
    for j in range(1,len(tok)):
        p = parent[j]
        depth[j] = depth[p] + 1
        if p in rows and tok[j] == truth[depth[j]-1]:
            rows.add(j)
    return max(depth[j] for j in rows)

def avg(rows):
    a = np.asarray(rows)
    return dict(cycles=len(rows), **{k:float(x) for k,x in zip(
        ['chain','dynamic','spine6_leaf','margin_policy','dispatched_tree_fraction','chain_replay_match'], a.mean(axis=0))})

def greedy(files, labels, tau):
    groups = defaultdict(list)
    for fn, label in zip(files, labels, strict=True):
        with np.load(fn) as pack:
            z = dict(pack)
        if not len(z["cycles"]): continue
        accepted = {int(p):int(k) for p,k,_ in z["cycles"]}
        start = int(z["cycles"][0,0])
        for i,pos in enumerate(z["p0"]):
            j = int(pos) - start
            truth = z["tokens"][j+1:j+8]
            if j < 0 or len(truth)!=7 or pos not in accepted: continue
            parent,tok,_,_,spine,_ = build_tree(int(z["anchor"][i]),z["cand"][i],z["unary"][i],
                z["hp"][i].astype(np.float32),pc,sc,n=7,topc=4,tau=tau)
            scores = root_scores(z,i)
            order = np.argsort(-scores)
            runner = z["cand"][i,0,order[1]]
            chain = next((k for k,(c,t) in enumerate(zip(spine,truth)) if c!=t),7)
            side = min(6,chain) + int(runner==truth[0])
            dynamic = tree_walk(parent,tok,truth)
            prob=np.exp((scores-scores.max())/tau); prob/=prob.sum()
            # One illustrative margin, specified before the run. No threshold optimization.
            dispatch = (prob[order[0]]-prob[order[1]] < 0.5) and not is_chain(parent,tok,spine)
            values=[1+chain,1+dynamic,1+side,1+(dynamic if dispatch else chain),int(dispatch),int(chain==accepted[pos])]
            groups['all'].append(values)
            groups[label].append(values)
    return {k:avg(v) for k,v in groups.items()}

def sampled(directory):
    totals = []
    nreq=0
    for fn in sorted(directory.glob("*.npz")):
        with np.load(fn) as pack:
            if "has_draft" not in pack: continue
            z=dict(pack)
        measured = z["has_draft"].astype(bool) & (z["dist_p"][:,0].max(axis=1) > 0)
        if measured.any(): nreq+=1
        for i in np.flatnonzero(measured):
            parent,tok=z["parent"][i],z["tok"][i]
            ids,pr=z["dist_ids"][i],z["dist_p"][i]
            candidates=z["cand"][i]; unary=z["unary"][i]; hp=z["hp"][i].astype(np.float32)
            pred=int(z["anchor"][i]); spine=[]
            for d in range(6):
                scores=unary[d]+sc(candidates[d]) @ (pc(pred)*hp[d])
                order=np.argsort(-scores)
                if d==0: runner=int(candidates[d,order[1]])
                pred=int(candidates[d,order[0]]); spine.append(pred)
            assert runner!=spine[0]
            maps=[{int(t):float(q) for t,q in zip(idrow,prow) if q>0} for idrow,prow in zip(ids,pr)]
            def interval(row,t):
                if row is None: return 0.,1.
                if t in maps[row]: return (maps[row][t],)*2
                return 0.,min(1.,max(0.,1.-sum(maps[row].values())))
            children={(int(parent[r]),int(tok[r])):r for r in range(1,8)}
            lo,hi=interval(0,runner); lo+=1;hi+=1
            row=0; plo=phi=1.
            for t in spine:
                l,h=interval(row,t)
                plo*=l;phi*=h;lo+=plo;hi+=phi
                row=children.get((row,t))
            pathp=[1.]
            for r in range(1,8):
                p=int(parent[r]);pathp.append(pathp[p]*maps[p].get(int(tok[r]),0.))
            totals.append([sum(pathp),lo,hi])
    served,lo,hi=np.mean(totals,axis=0)
    return dict(requests=nreq,cycles=len(totals),served=float(served),spine6_leaf_lower=float(lo),spine6_leaf_upper=float(hi))

hist=sorted((repo/"results/19nov/tree/tree_log").glob("req_*.npz"))
bench=json.loads((repo/"results/20nov/bench/chain_v1.json").read_text())
labels=[req['cat'] for rung in bench['rungs'].values() for req in rung]
real=sorted((volume/"tree20/chainlog_v1").glob("req_*.npz"))
assert len(real) in (len(labels),len(labels)+1)
result={'implementation_sha256':hashlib.sha256((pipe/'tree_verify.py').read_bytes()).hexdigest(),
        'greedy_19nov_tau1':greedy(hist,['original']*len(hist),1.)['all'],
        'greedy_realistic_tau1_5':greedy(real[-len(labels):],labels,1.5),
        'sampled_long_pi_like':sampled(volume/'treelog_lgo21')}
assert len(hist)==12 and result['greedy_19nov_tau1']['cycles']==964
assert result['greedy_19nov_tau1']['chain_replay_match']==1
assert abs(result['greedy_19nov_tau1']['dynamic']-3.348)<0.001
assert abs(result['sampled_long_pi_like']['served']-4.0750158)<1e-5
print(json.dumps(result,indent=2))
