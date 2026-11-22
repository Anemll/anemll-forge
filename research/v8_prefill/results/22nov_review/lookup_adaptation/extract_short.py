"""Reduce the existing synthetic lookup-observer NPZs to counts. No weights, model imports or retained token IDs."""
import datetime as dt
import json
from pathlib import Path
import sys
import numpy as np

def get_stamp(path):
    return int(path.stem.split("_")[1])/1000
def main(result):
    result=Path(result)
    trace=Path("/Volumes/SSD4TB/anemll-forge-research/treelog_lgo21")
    files=sorted(trace.glob("tree_*.npz"),key=get_stamp)
    costs=json.loads((result/"lookup_gate/sim.json").read_text())["costs"]
    out=[]
    for label,kind in [("lgo_s_ca","ca_s"),("lgo_s_rm","rm_s")]:
        bench=json.loads((result/"bench"/(label+".json")).read_text())
        for rung,rows in bench["rungs"].items():
            for row in rows:
                # request_end flushes just before the HTTP completion. Require a unique nearby trace.
                near=[p for p in files if row["t0"]<=get_stamp(p)<=row["t_end"]+1 and abs(get_stamp(p)-row["t_end"])<=3]
                if len(near)!=1: raise ValueError((label,rung,row["index"],"ambiguous timestamp",len(near)))
                with np.load(near[0],allow_pickle=False) as z:
                    p=z["p0"]; path=z["path"]; tok=z["tok"]; bonus=z["nxt"]
                    proposals=z["lk_drafts"]; lengths=z["lk_len"]; branch=z["tree"]
                    if not len(p):
                        if row["completion_tokens"] > 8: raise ValueError(("empty trace",label,rung,row["index"],row["completion_tokens"]))
                        continue
                    # Internal positions include template/control tokens. Join by unique wall clock
                    # and the server's total number of tree/chain calls, not client prompt length.
                    if len(p)!=row["server"]["tree_cycles"]+row["server"]["chain_cycles"]:
                        raise ValueError("logged round count differs from server")
                    chunks=[]; sizes=[]
                    for i,pa in enumerate(path):
                        valid=pa[pa>=0]
                        chunks.append([int(tok[i,x]) for x in valid[1:]]+[int(bonus[i])])
                        sizes.append(len(valid))
                    contiguous=(np.diff(p)==np.array(sizes[:-1])) & (bonus[:-1]==tok[1:,0])
                    cc=costs[kind+" "+rung]; rounds=[]; tail=[]; forward={}
                    for i in reversed(range(len(chunks))):
                        if i==len(chunks)-1 or not contiguous[i]: tail=[]
                        # The last call may overrun a cap. Never use its drafts to label an earlier lookup.
                        sequence=(chunks[i]+tail)[:7] if i+1<len(chunks) and contiguous[i] else []
                        if len(sequence)==7: forward[i]=sequence
                        tail=sequence
                    for i in range(len(p)):
                        if i not in forward: continue
                        m=0
                        for d,t in zip(proposals[i],forward[i]):
                            if d<0 or int(d)!=t: break
                            m+=1
                        cost=cc["tree" if branch[i] else "chain"]
                        rounds.append([int(lengths[i]),sizes[i],m,cost-cc["lookup"],cost,int(p[i])])
                    out.append({"group":row["cat"],"rung":rung,"index":row["index"],"rounds":rounds})
    print(json.dumps(out,separators=(",",":")))
if __name__=="__main__":main(sys.argv[1])
