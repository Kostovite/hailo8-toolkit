"""Bottleneck relief: read a compiled sub-network's per-node cycles and emit nested-defuse hints for its slowest convs.

Rule (probe/RULES.md 14): a subcluster computes an 8-output-channel x 8-column tile, so a conv can use up to
ceil(o/8) x ceil(W/8) subclusters. For o > 64 the level-1 heuristic packs 16 channels per SC (features_per_mac 2) and
leaves half the tiles idle; splitting the output features first (defuse x2), then each half spatially into the
ceil(W/8) slices, fills the tiles: c72 6653 -> 13107 FPS, c128 3832 -> 7598, c64 12552 -> 14719.
Entry layers (fed across a cut) get only the feature split (R3: never spatial at a context entry).

Usage: python relieve.py SUB_compiled.har [--frac 0.8] [--max 3]  -> hint lines on stdout
"""
import argparse
import json
import math
import os
import re
import sys
import tarfile
import tempfile

sys.path.insert(0, "/mnt/storage/lib_project/models/Hailo_model/hailo_re/probe")
from ighw import load  # noqa: E402

SUFFIX = re.compile(r"(_(s?d\d+|s?dc|fs(_\d+)?|rs|lanes\w*|defuse_\w+))+$")
CUT = re.compile(r"(input|output)_layer5\d\d")


def relieve(har, frac=0.8, max_layers=3):
    with tarfile.open(har) as t:
        names = t.getnames()
        hef = t.extractfile(next(m for m in names if m.endswith(".hef"))).read()
        hn = json.loads(t.extractfile(next(m for m in names if m.endswith(".hn") and not
                                             m.endswith((".native.hn", ".fp.hn")))).read())
    L = {k.split("/")[-1]: v for k, v in hn["layers"].items()}
    with tempfile.NamedTemporaryFile(suffix=".hef") as f:
        f.write(hef)
        f.flush()
        g = load(f.name)
    cyc = {}
    for cn in g.contexts_graph.nodes:
        byidx = {n.index: n.name for n in cn.network_graph.nodes}
        for r in cn.network_graph.resources_list:
            nm = byidx.get(r.layer_index, "")
            base = SUFFIX.sub("", nm)
            if base in L:
                cyc[base] = max(cyc.get(base, 0), r.frame_total_cycles)
    if not cyc:
        return ""
    top = max(cyc.values())
    hint = []
    for base, c in sorted(cyc.items(), key=lambda x: -x[1]):
        v = L[base]
        if c < frac * top or len(hint) >= max_layers:
            break
        if v["type"] != "conv" or (v.get("params", {}).get("groups", 1) or 1) > 1:
            continue
        h, w, o = v["output_shapes"][0][1:4]
        if o < 16 or h < 2 or "feature_splitter" in base or any(  # attention layout / slice families: leave
                L.get(x.split("/")[-1], {}).get("type") in ("matmul", "format_conversion") for x in v["output"]):
            continue
        entry = any(CUT.search(s.split("/")[-1]) for s in v["input"])
        lines = [f"{base}_d0, {base}_d1, {base}_dc = defuse({base}, 2)"]
        s = math.ceil(w / 8)
        if not entry and s >= 2:
            for d in ("d0", "d1"):
                pieces = ", ".join(f"{base}_{d}_sd{j}" for j in range(s))
                lines.append(f"{pieces}, {base}_{d}_sdc = defuse({base}_{d}, {s}, defuse_type=SPATIAL)")
        hint.append("\n".join(lines))
    return "\n".join(hint) + ("\n" if hint else "")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("har")
    ap.add_argument("--frac", type=float, default=0.8)
    ap.add_argument("--max", type=int, default=3)
    a = ap.parse_args()
    sys.stdout.write(relieve(a.har, a.frac, a.max))
