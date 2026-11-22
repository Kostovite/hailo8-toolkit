"""Join the physical layout (hwlayout: enable_sequencer bitmaps / enable_lcu per cluster) with the compiler's
per-node resources (ighw: subclusters, apus, input aligners, l2 weights...) per context, to name what each bitmap
counts. Usage: python hwjoin.py HEF [HEF ...]"""
import collections
import sys

from hwlayout import layout
from ighw import load


def per_ctx_nodes(path):
    g = load(path)
    out = {}
    for i, cn in enumerate(g.contexts_graph.nodes):
        s = collections.Counter()
        for r in cn.network_graph.resources_list:
            s["sc"] += r.subclusters
            s["apu"] += r.apus
            s["ia"] += r.input_aligners
            s["lcu"] += r.control_type == 1 or 0
            s["wblk"] += r.weights_blocks
            s["oblk"] += r.output_blocks
            s["l2w"] += r.l2_weights
            s["n"] += 1
        out[f"ctx{i}"] = s
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        L = {c["ctx"]: c for c in layout(p)}
        N = per_ctx_nodes(p)
        for ctx, s in N.items():
            c = L.get(ctx) if len(N) > 1 else L.get("preliminary")
            cl = c["clusters"] if c else {}
            hw = {k: sum(len(d[k]) for d in cl.values()) for k in ("sc", "l2", "apu", "ia", "lcu")}
            print(p.split("/")[-2][:40], ctx, "nodes", dict(s), "| bitmaps", hw, "clusters", len(cl))
