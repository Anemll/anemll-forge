import sys, re, numpy as np
import xml.etree.ElementTree as ET
def rows(path):
    root = ET.parse(path).getroot()
    ids = {}
    cols = [c.find("mnemonic").text for c in root.iter("col")]
    out = []
    for r in root.iter("row"):
        for d in r.iter():
            if "id" in d.attrib:
                ids[d.attrib["id"]] = d
        vals = [ids[el.attrib["ref"]] if "ref" in el.attrib else el for el in r]
        out.append(dict(zip(cols, vals)))
    return cols, out
T = sys.argv[1]
cols, A = rows(T + "/ane-hw-intervals.xml")
st = np.array([int(r["start"].text) for r in A]) / 1e6; du = np.array([int(r["duration"].text) for r in A]) / 1e6
o = np.argsort(st); st, du = st[o], du[o]
gap = st[1:] - (st[:-1] + du[:-1])
print("ANE predictions:", len(st), "window", round(st[-1] + du[-1] - st[0], 1), "ms")
# active periods: runs separated by gaps > 20 ms
cut = np.where(gap > 20)[0]
segs = np.split(np.arange(len(st)), cut + 1)
for sg in segs:
    if len(sg) < 50: continue
    s0, s1 = st[sg[0]], st[sg[-1]] + du[sg[-1]]
    g = gap[sg[:-1]]
    print(f" run: {len(sg)} preds in {s1-s0:.0f} ms | busy {du[sg].sum()/(s1-s0):.3f} | dur ms median {np.median(du[sg]):.3f} mean {du[sg].mean():.3f} p10 {np.percentile(du[sg],10):.3f} p90 {np.percentile(du[sg],90):.3f} | gap ms median {np.median(g):.3f} mean {g.mean():.3f} p90 {np.percentile(g,90):.3f} | gaps sum {g[g>0].sum():.0f} ms")
cols, G = rows(T + "/metal-gpu-intervals.xml")
print("GPU cols:", cols)
print({k: (v.attrib.get("fmt"), v.text) for k, v in G[0].items()})
