"""Compare the saved 19-turn arms. Read existing benchmark texts only for digest equality."""
import json
import sys
from pathlib import Path

def aggregate(rows):
    g = sum(r["completion_tokens"] - 1 for r in rows)
    sec = sum(r["t_last"] - r["t_first"] for r in rows)
    cycles = sum(r["server"]["gen"] / r["server"]["tok_verify"] for r in rows)
    lookup = sum(r["server"].get("lookup_cycles", 0) for r in rows)
    lookup_y = sum(r["server"].get("lookup_cycles", 0) * r["server"].get("lookup_tok_verify", 0) for r in rows)
    return dict(n=len(rows), tps=g/sec, tokens=g, seconds=sec, lookup_cycles=lookup, lookup_share=lookup/cycles,
                lookup_tv=lookup_y/lookup if lookup else None)
def run(result):
    root=Path(result)/"bench"
    inputs=[json.loads((root/f"lgl_{a}.json").read_text()) for a in ("off1","yield","off2")]
    assert len({(x["prompts_sha"], x["seed"], x["temp"], x["think"]) for x in inputs}) == 1
    a,y,b=[x["rungs"]["long"] for x in inputs]
    assert len(a)==len(y)==len(b)==19
    assert all((l["index"],l["seed"],l["prompt_tokens"],l["sha"])==(r["index"],r["seed"],r["prompt_tokens"],r["sha"]) and
               (r["index"],r["seed"],r["prompt_tokens"])==(t["index"],t["seed"],t["prompt_tokens"]) for l,t,r in zip(a,y,b))
    out={}
    def summary(indices):
        off=aggregate([r[i] for r in [a,b] for i in indices])
        gate=aggregate([y[i] for i in indices])
        return dict(turns=len(indices),gain_pct=100*(gate["tps"]/off["tps"]-1),off=off,gate=gate)
    for name,lo,hi in [("all",0,100000),("8-16K",8192,16384),("16-32K",16384,32768),("32-57K",32768,65536)]:
        inds=[i for i,r in enumerate(a) if lo<=r["prompt_tokens"]<hi]
        out[name]=summary(inds)
        same=[i for i in inds if a[i]["sha"]==y[i]["sha"]]
        out[name]["same_output"]=summary(same) if same else None
    out["mixed_band_projection"]={}
    for name,selector in [("off_below_32k",lambda i:a[i]["prompt_tokens"]>=32768),
                          ("perfect_retrospective_turn_choice",lambda i:aggregate([y[i]])["tps"]>aggregate([a[i],b[i]])["tps"])]:
        # Average the two off-arm totals per request to give each turn one observation.
        ns=ds=0.0
        for i in range(len(a)):
            use=selector(i)
            g=aggregate([y[i]] if use else [a[i],b[i]])
            f=1 if use else 0.5
            ns+=f*g["tokens"]; ds+=f*g["seconds"]
        out["mixed_band_projection"][name]=dict(tps=ns/ds,gain_pct=100*((ns/ds)/out["all"]["off"]["tps"]-1))
    pairs=[(aggregate([a[i],b[i]]),aggregate([y[i]])) for i in range(len(a))]
    rate=out["all"]["off"]["tps"]
    for _ in range(30):
        picks=[max(((off["tokens"]/2,off["seconds"]/2),(gate["tokens"],gate["seconds"])),
                   key=lambda x:x[0]-rate*x[1]) for off,gate in pairs]
        new=sum(n for n,_ in picks)/sum(t for _,t in picks)
        if abs(new-rate)<1e-10: break
        rate=new
    out["mixed_band_projection"]["maximum_over_recorded_turn_choices"]={
        "tps":new,"gain_pct":100*(new/out["all"]["off"]["tps"]-1)}
    return out
if __name__=="__main__":
    print(json.dumps(run(sys.argv[1]),indent=2))
