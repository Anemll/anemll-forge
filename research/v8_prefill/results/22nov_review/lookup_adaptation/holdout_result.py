"""Compare complete, counterbalanced arms on the disjoint-file session, weighting emitted tokens and time."""
import collections
import json
import math
from pathlib import Path
import sys

def band(n):return "<16K" if n<16384 else "16-32K" if n<32768 else "32K+"
def pool(rows):
    tokens=sum(r["completion_tokens"]-1 for r in rows)
    seconds=sum(r["stream_decode_s"] for r in rows)
    if not seconds:return None
    power_rows=[r for r in rows if r.get("decode_power") and r["completion_tokens"]>1]
    power_tokens=sum(r["completion_tokens"]-1 for r in power_rows)
    joules=sum(r["decode_power"]["energy_j"] for r in power_rows)
    return {"requests":len(rows),"tokens":tokens,"seconds":seconds,"tps":tokens/seconds,
            "context_range":[min(r["prompt_tokens"] for r in rows),max(r["prompt_tokens"] for r in rows)],
            "gross_joules_per_streamed_token":joules/power_tokens if power_tokens else None,"gross_joules":joules,
            "power_tokens":power_tokens,
            "lookup_cycles":sum(r["lookup_cycles"] for r in rows),
            "tok_per_verify":sum(r["completion_tokens"] for r in rows)/sum(r["completion_tokens"]/r["tv"] for r in rows),
            "time_to_first_output_seconds":sum(r["ttft_s"] for r in rows),
            "total_request_seconds":sum(r["request_s"] for r in rows),
            "finish":dict(collections.Counter(r["finish"] for r in rows))}
def match(a,b):
    a={r["index"]:r for r in a}; b={r["index"]:r for r in b}
    if a.keys()!=set(range(19)) or a.keys()!=b.keys() or any(a[k]["body_sha256"]!=b[k]["body_sha256"] or a[k]["seed"]!=b[k]["seed"] or a[k]["prompt_tokens"]!=b[k]["prompt_tokens"] for k in a):
        raise ValueError("mismatched seeds, request bodies or indices")
    out={}
    for label in ["all","<16K","16-32K","32K+"]:
        pairs=[(r,b[k]) for k,r in a.items() if label in ("all",band(r["prompt_tokens"]))]
        if not pairs:raise ValueError("unrepresented context band")
        l,r=zip(*pairs); x,y=pool(l),pool(r)
        if not x or not y:raise ValueError("missing timing")
        paired=[((j["completion_tokens"]-1)/j["stream_decode_s"])/((i["completion_tokens"]-1)/i["stream_decode_s"]) for i,j in pairs if min(i["completion_tokens"],j["completion_tokens"])>1 and min(i["stream_decode_s"],j["stream_decode_s"])>0]
        out[label]={"off":x,"candidate":y,"gain_pct":100*(y["tps"]/x["tps"]-1),"same_text":sum(i["sha"]==j["sha"] for i,j in pairs),
                    "wins":sum(t>1 for t in paired),"paired_geomean_pct":100*math.expm1(sum(map(math.log,paired))/len(paired)) if paired else None}
        same=[(i,j) for i,j in pairs if i["sha"]==j["sha"]]
        if same:
            left,right=map(pool,zip(*same))
            out[label]["same_output_subset"]={"off":left,"candidate":right,"gain_pct":100*(right["tps"]/left["tps"]-1)}
    return out
def audit(base):
    base=Path(base)
    if not (base/"restoration.txt").is_file():raise ValueError("no default restoration record")
    if "| lookup off" not in (base/"restored.banners").read_text():raise ValueError("unexpected restored banner")
    arms=collections.defaultdict(dict)
    manifest=None
    for f in sorted(base.glob("[0-9]_*.json")):
        arm=f.stem.split("_",1)[1];d=json.loads(f.read_text())
        banner=f.with_suffix(".banners").read_text()
        watchdog=f.with_suffix(".watchdog.log").read_text()
        required={"off":"| lookup off","ctx":"lookup_gate=ctx:32768:48:1.25","yield":"lookup_gate=yield:1.25"}
        if "tau=1.5 sampled=on" not in banner or required.get(arm,"lookup_gate=yield:1.25") not in banner:
            raise ValueError("unexpected serving arm")
        if "limit +1024 MB" not in watchdog or "watchdog done:" not in watchdog or "WATCHDOG: swap grew" in watchdog:
            raise ValueError("missing or triggered swap watchdog")
        if not d["valid"] or len(d["turns"])!=d["expected_turns"] or len(d["turns"])!=19:raise ValueError(f.name)
        if d["seed"] in arms[arm] or d["seed"] not in (8700,18700):raise ValueError("duplicate or unexpected seed")
        if manifest is None:manifest=d["fixture_sha"]
        if d["fixture_sha"]!=manifest:raise ValueError("holdout manifest changed")
        if arm=="off" and any(r["lookup_cycles"] for r in d["turns"]):raise ValueError("lookup taken in control arm")
        arms[arm][d["seed"]]=d
    if "off" not in arms or len(arms["off"])!=2:raise ValueError("no counterbalanced controls")
    out={}
    for arm,blocks in arms.items():
        if arm=="off":continue
        if blocks.keys()!=arms["off"].keys():raise ValueError("incomplete balanced pair")
        sets={str(seed):match(arms["off"][seed]["turns"],d["turns"]) for seed,d in blocks.items()}
        combined={}
        for name in next(iter(sets.values())):
            sums={}
            for side in ("off","candidate"):
                obs=[v[name][side] for v in sets.values()]
                sums[side]=sum(o["tokens"] for o in obs)/sum(o["seconds"] for o in obs)
            combined[name]=100*(sums["candidate"]/sums["off"]-1)
        summaries={side:{name:pool([r for d in arms[side].values() for r in d["turns"]
                         if name in ("all",band(r["prompt_tokens"]))]) for name in combined} for side in ("off",arm)}
        out[arm]={"paired_blocks":sets,"pooled_gain_pct":combined,"fixture_sha":manifest,"pooled":summaries,
                  "clears_promotion_gate":combined["all"]>=5 and not any(all(b[k]["gain_pct"] < -2 for b in sets.values()) for k in combined if k!="all")}
    if "yield" in arms:
        for arm in list(out):
            if arm=="yield":continue
            out[arm]["against_yield_by_seed"]={str(seed):match(arms["yield"][seed]["turns"],d["turns"]) for seed,d in arms[arm].items()}
    return out
if __name__=="__main__":print(json.dumps(audit(sys.argv[1]),indent=2))
