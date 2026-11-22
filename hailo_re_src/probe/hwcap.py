"""Per-context and per-cluster unit usage over many HEFs -> the chip's capacities and the binding resource.
Usage: python hwcap.py LIST_OF_HEFS.txt"""
import collections
import sys

from hwlayout import layout
from ighw import load

K = ("sc", "l2", "apu", "ia", "lcu")
pc = collections.defaultdict(list)    # per cluster counts
tot = collections.defaultdict(list)   # per context totals
l3i = collections.Counter()
mem = []
for p in (l.strip() for l in open(sys.argv[1])):
    try:
        L = layout(p)
        g = load(p)
    except Exception:  # noqa: BLE001
        continue
    multi = len(g.contexts_graph.nodes) > 1
    for c in L:
        if multi == (c["ctx"] == "preliminary"):
            continue
        cl = c["clusters"]
        if not cl:
            continue
        for d in cl.values():
            for k in K:
                pc[k].append(len(d[k]))
            for i, o in d["l3"]:
                l3i[i] += 1
        for k in K:
            tot[k].append(sum(len(d[k]) for d in cl.values()))
    for cn in g.contexts_graph.nodes:
        rs = cn.network_graph.resources_list
        mem.append((sum(r.weights_blocks for r in rs), sum(r.output_blocks for r in rs),
                    sum(r.l2_weights for r in rs), len({0}), p))
for k in K:
    v = sorted(pc[k])
    t = sorted(tot[k])
    print(f"{k:4s} per-cluster max {v[-1]:3d}  p99 {v[int(.99*len(v))]:3d} | per-context max {t[-1]:4d} p99 {t[int(.99*len(t))]}")
print("initial l3 index max", max(l3i), sorted(l3i.items())[-5:])
m = sorted(mem)
print("max weights_blocks/ctx", max(x[0] for x in mem), "max output_blocks/ctx", max(x[1] for x in mem),
      "max wb+ob", max(x[0] + x[1] for x in mem), "max l2_weights/ctx", max(x[2] for x in mem))
