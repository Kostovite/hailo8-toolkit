"""Table of a sweep: per case, contexts, physical units used (sum over clusters) vs node resources, memory blocks,
FPS, and the error for failures. Usage: python hwtab.py RUN_DIR [filter-substring]"""
import collections
import glob
import json
import os
import re
import sys

from hwlayout import layout
from ighw import load

rows = []
for f in sorted(glob.glob(f"{sys.argv[1]}/*/result.json")):
    case = f.split("/")[-2]
    if len(sys.argv) > 2 and sys.argv[2] not in case:
        continue
    j = json.load(open(f))
    h = f.replace("result.json", "m.hef")
    if j["stage"] != "ok" or not os.path.exists(h):
        e = re.sub(r"\s+", " ", j.get("error") or "")
        m = re.search(r"(Memory units capacity exceeded[^)]*\)|could not find a valid partition|microcode[^.]*|"
                      r"vector::_M_range_check|Initial utilization overflow|[A-Z]\w+Error[^.]{0,80})", e)
        rows.append((case, j["stage"], m.group(0) if m else e[:100]))
        continue
    g = load(h)
    L = layout(h)
    multi = len(g.contexts_graph.nodes) > 1
    cl = [c for c in L if (c["ctx"] != "preliminary") == multi]
    u = collections.Counter()
    pcmax = collections.Counter()
    for c in cl:
        for d in c["clusters"].values():
            for k in ("sc", "l2", "apu", "ia", "lcu"):
                u[k] += len(d[k])
                pcmax[k] = max(pcmax[k], len(d[k]))
    rs = [r for cn in g.contexts_graph.nodes for r in cn.network_graph.resources_list]
    wb, ob = sum(r.weights_blocks for r in rs), sum(r.output_blocks for r in rs)
    l2w = sum(r.l2_weights for r in rs)
    l4 = sum(r.l4_cuts for r in rs)
    ncl = max(len(c["clusters"]) for c in cl) if cl else 0
    rows.append((case, "ok", f"fps {g.bottleneck_fps:9.1f} ctx {len(g.contexts_graph.nodes)} cl {ncl} "
                 f"sc {u['sc']:3d} l2 {u['l2']:3d} apu {u['apu']:3d} ia {u['ia']:3d} lcu {u['lcu']:3d} "
                 f"| max/cl sc {pcmax['sc']} l2 {pcmax['l2']} apu {pcmax['apu']} lcu {pcmax['lcu']} "
                 f"| wblk {wb} oblk {ob} l2w {l2w} l4cuts {l4}"))
for r in rows:
    print(f"{r[0]:40s} {r[1]:8s} {r[2]}")
