"""Scan a compiled model.mil: for every state, list ops that consume values derived from its read_state and are
not ancestors of its coreml_update_state value; also report their position relative to the write."""
import re
import sys

text = open(sys.argv[1] + "/model.mil").read()
ops = []  # (index, out, op, inputs)
for line in text.splitlines():
    w = re.match(r"\s*write_state\(data = (\w+), input = (\w+)\)", line)
    if w:
        ops.append((len(ops), None, "coreml_update_state", [w.group(2), w.group(1)]))
        continue
    m = re.match(r"\s*(?:tensor<[^>]*>|[\w<>\[\], ]+?)\s+(\w+)\s*=\s*(\w+)\((.*)\)", line)
    if not m:
        continue
    out, op, args = m.groups()
    ins = re.findall(r"=\s*([A-Za-z_]\w*)", args)
    ops.append((len(ops), out, op, ins))
prod = {o[1]: o for o in ops}
for i, out, op, ins in ops:
    if op != "read_state" or "coreml_update_state" in out:
        continue
    state = ins[0]
    derived = {out}
    for j, o2, op2, ins2 in ops[i + 1:]:
        if op2 == "coreml_update_state" and ins2 and ins2[0] == state:
            continue
        if any(x in derived for x in ins2):
            derived.add(o2)
    upd = [o for o in ops if o[2] == "coreml_update_state" and o[3][0] == state]
    if not upd:
        print(f"{state}: no update"); continue
    wi = upd[0][0]
    anc, stack = set(), [upd[0][3][1]]
    while stack:
        v = stack.pop()
        if v in anc or v not in prod:
            continue
        anc.add(v)
        stack.extend(prod[v][3])
    bad = [o for o in ops if o[1] in derived and o[1] not in anc]
    after = [o for o in ops if o[1] in derived and o[0] > wi]
    print(f"{state}: read@{i} write@{wi}  read-derived ops {len(derived)}  not feeding write {len(bad)}  after write {len(after)}")
    for o in bad[:6]:
        print(f"    not-feeding: {o[2]} {o[1]} @{o[0]}")
    for o in after[:6]:
        print(f"    after-write: {o[2]} {o[1]} @{o[0]}")
