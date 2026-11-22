"""Rewrite a compiler auto.alls with a new context assignment of HN layers.

The compiler's script lists post-defuse nodes (conv22_d0..dc, concat_f_from_..., sh_from_...) per context and pins
them to clusters (place) and buffers. Given a map HN layer -> context, every node goes to the context of its parent
layer; auxiliary nodes (concat_/sh_/ws_/fs_/auto_reshape_/..._from_A_to_B) go to the context of the layer they feed
(the last HN name in their name). Contexts whose membership changed are "dirty": their place() lines and every buffers() line
touching them are dropped (buffer sizes depend on placement); ddr()/l4_portal() lines are kept unless their edge
now crosses contexts; plus any line naming a dropped definition, so the
compiler re-derives them; clean contexts keep the reference solution verbatim.

Usage: python mkalls.py REF.alls HN NEW_CTX.json OUT.alls [--res=0.95]
  NEW_CTX.json: {"layer": ctx_index} for HN layer short names (cutsolve.py --emit_json)
"""
import json
import re
import sys

ref, hn_path, ctx_path, out = sys.argv[1:5]
res_cap = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--res=")), None)
alls = open(ref).read()
hn = json.load(open(hn_path))
hn_names = sorted((n.split("/")[-1] for n in hn["layers"]), key=len, reverse=True)
new = json.load(open(ctx_path))

ctx_re = re.compile(r"^(\w+)_context_(\d+) = context\(\[([^\]]*)\]\)\s*$", re.M)
ref_ctx, net = {}, None
for m in ctx_re.finditer(alls):
    net = m.group(1)
    for n in m.group(3).split(","):
        ref_ctx[n.strip()] = int(m.group(2))


def parent(node):
    for h in hn_names:  # direct: conv22, conv22_d3, conv12_sd0, dw1_defuse_1x1
        if node == h or (node.startswith(h + "_") and not node.startswith(("concat_", "sh_from", "ws_from", "fs_from",
                                                                          "auto_reshape", "ddr_portal", "l4_portal"))):
            return h
    # auxiliary: the last HN name mentioned feeds it
    best, pos = None, -1
    for h in hn_names:
        for m in re.finditer(r"(?<![A-Za-z0-9])" + re.escape(h) + r"(?![0-9])", node):
            if m.start() > pos:
                best, pos = h, m.start()
    return best


assign, unknown = {}, []
for node in ref_ctx:
    p = parent(node)
    if p is None or p not in new:
        unknown.append(node)
        assign[node] = ref_ctx[node]
    else:
        assign[node] = new[p]
moved = {n for n in assign if assign[n] != ref_ctx[n]}
used = sorted(set(assign.values()))  # drop empty contexts, keep order
renum = {c: i for i, c in enumerate(used)}
assign = {n: renum[c] for n, c in assign.items()}
moved = {n for n in assign if assign[n] != ref_ctx[n]}
K = len(used)
dirty = {assign[n] for n in moved} | {ref_ctx[n] for n in moved}
print(f"{len(assign)} nodes, {len(moved)} moved, dirty contexts {sorted(dirty)}, {len(unknown)} kept at reference context: {unknown[:8]}")

lines, dropped, dropped_names, places = [], {"place": 0, "pair": 0}, [], []
placed = {}  # node -> (new ctx, cluster) for cluster groups that survive intact
_cand = []
for line in alls.splitlines():
    mp = re.match(r"^\w+_context_\d+\.place\((\d+), \[([^\]]*)\]\)", line)
    if mp:
        nodes = [n.strip() for n in mp.group(2).split(",")]
        tgt = {assign.get(n) for n in nodes}
        if len(tgt) == 1 and None not in tgt:
            _cand.append((tgt.pop(), int(mp.group(1)), nodes))
_taken = set()
for c, cl, nodes in sorted(_cand, key=lambda t: -len(t[2])):
    if (c, cl) not in _taken:
        _taken.add((c, cl))
        for n in nodes:
            placed[n] = (c, cl)
pair_re = re.compile(r"^(?:\w+ = )?(buffers|ddr|l4_portal)\((\w[\w\-]*), ([\w\-]+)")
place_re = re.compile(r"^(\w+)_context_(\d+)\.place\(")
emitted_ctx = False
for line in alls.splitlines():
    if ctx_re.match(line):
        if not emitted_ctx:
            for c in range(K):
                nodes = [n for n in assign if assign[n] == c]
                lines.append(f"{net}_context_{c} = context([{', '.join(nodes)}])")
            emitted_ctx = True
        continue
    m = place_re.match(line)
    if m:  # a cluster group survives if all its nodes land in one context; collisions resolved below
        nodes = re.findall(r"[\w\-]+", line.split("[", 1)[1])
        tgt = {assign.get(n) for n in nodes}
        cl = int(re.match(r"^\w+\.place\((\d+)", line).group(1))
        if len(tgt) == 1 and None not in tgt:
            places.append((tgt.pop(), cl, len(nodes), nodes))
        else:
            dropped["place"] += 1
        continue
    m = pair_re.match(line)
    intact = m and m.group(2) in placed and m.group(3) in placed and placed[m.group(2)][0] == placed[m.group(3)][0]
    if m and (m.group(1) == "buffers" and not intact and (assign.get(m.group(2)) in dirty or assign.get(m.group(3)) in dirty)
              or m.group(1) != "buffers" and assign.get(m.group(2)) != assign.get(m.group(3))):
        # buffer sizes depend on placement; ddr/l4 portals are structural and kept unless the edge now crosses a cut
        dropped["pair"] += 1
        mm = re.match(r"^([\w\-]+) = ", line)
        if mm:
            dropped_names.append(mm.group(1))
        continue
    if res_cap and line.startswith("resources_param("):
        line = re.sub(r"(max_\w+_utilization)=[0-9.]+", r"\g<1>=" + res_cap, line)
    lines.append(line)
seen = {}
for c, cl, sz, nodes in sorted(places, key=lambda t: -t[2]):
    if (c, cl) in seen:
        dropped["place"] += 1
        continue
    seen[(c, cl)] = True
    lines.append(f"{net}_context_{c}.place({cl}, [{', '.join(nodes)}])")
# a dropped `X = ddr(...)` etc. defines X: drop every line that still mentions a removed name (transitively)
removed = set(dropped_names)
while True:
    keep = []
    for line in lines:
        hit = removed and any(re.search(r"(?<![\w\-])" + re.escape(x) + r"(?![\w\-])", line) for x in removed)
        if hit:
            mm = re.match(r"^([\w\-]+) = ", line)
            if mm:
                removed.add(mm.group(1))
            dropped["ref"] = dropped.get("ref", 0) + 1
        else:
            keep.append(line)
    if len(keep) == len(lines):
        break
    lines = keep
open(out, "w").write("\n".join(lines) + "\n")
print(f"wrote {out}: {K} contexts, dropped {dropped}")
