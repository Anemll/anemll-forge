"""Freeze 19 disjoint, tracked source-file choices before the new calibration; build messages only in memory."""
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

REPO=Path("/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-forge-v8")
BASE=Path("/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/bench/tree/long_agent_v1.json")
ASK=("Add explicit return types to two helpers in {f}.",
     "Add a useful error message for a bad argument in {f} while retaining its exit behavior.",
     "Document the assumptions of the first public function in {f}.",
     "Suggest a small identical-behavior cleanup in {f} and show the change.",
     "Give one identifier in {f} a clearer name and update its uses.",
     "Add an edge-case check or test for one function in {f}.")
def digest(b):return hashlib.sha256(b).hexdigest()
def git(*args):return subprocess.check_output(["git","-C",str(REPO),*args])
def freeze():
    original=json.loads(BASE.read_bytes())["rungs"]["long"]
    excluded={t["file"] for t in original}
    commit=git("rev-parse","HEAD").decode().strip()
    listed=git("ls-tree","-rl",commit,"scripts").decode().splitlines()
    opts=[]
    for l in listed:
        head,path=l.split("\t",1)
        if path.endswith(".py") and path not in excluded and path.count("/")==1:
            size=int(head.split()[-1])
            if 3500<=size<=22500:opts.append((path,size))
    if len(opts)<len(original):raise ValueError("too few held-out source files")
    rng=random.Random(82061);rng.shuffle(opts); picks=[]
    for turn in original:
        target=len(turn["messages"][-1]["content"].encode())
        path,size=min(opts,key=lambda e:abs(target-e[1]));opts.remove((path,size))
        b=git("show",commit+":"+path)
        picks.append({"path":path,"sha256":digest(b)})
    return {"source_files":picks,"commit":commit,"base_fixture_sha":digest(BASE.read_bytes()),"seed":8700,
            "session":"disjoint source files, edited task wording, same system and tools"}
def materialize(plan):
    assert digest(BASE.read_bytes())==plan["base_fixture_sha"]
    base=json.loads(BASE.read_bytes())["rungs"]["long"];tools=base[0]["tools"]
    history=base[0]["messages"][:1]; turns=[]
    for i,entry in enumerate(plan["source_files"]):
        path=entry["path"]; raw=git("show",plan["commit"]+":"+path)
        if digest(raw)!=entry["sha256"]:raise ValueError("changed holdout file")
        call={"id":f"call_{i}","type":"function","function":{"name":"read","arguments":json.dumps({"path":path})}}
        q=history+[{"role":"user","content":ASK[(i+2)%len(ASK)].format(f=path)},
                   {"role":"assistant","content":"","tool_calls":[call]},
                   {"role":"tool","tool_call_id":call["id"],"content":raw.decode(errors="replace")}]
        turns.append({"index":i,"messages":q,"tools":tools})
        history=q+[{"role":"assistant","content":f"I completed the requested change in `{path}`."}]
    return turns
if __name__=="__main__":
    p=Path(sys.argv[1])
    if p.exists():raise SystemExit("manifest already frozen")
    plan=freeze();assert len(materialize(plan))==19
    p.write_text(json.dumps(plan,indent=2)+"\n")
    print("Frozen 19 disjoint tracked files; manifest",digest(p.read_bytes()))
