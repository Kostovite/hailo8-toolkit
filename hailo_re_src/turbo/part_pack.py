"""part_pack.py : context partition packed by MEASURED per-layer resources (probe/layer_cost.py), not estimates.

The DFC Slotter and part_gen.py both under-count 16-bit layers (a 112-wide 16-bit conv is ~15 HW nodes / ~113
subclusters: ceil(W/8) spatial pieces), so their initial contexts are overloaded (probe/RULES.md section 29). Here
layers are taken in the HN's topological order and a context is closed before any measured resource would exceed
CAP x the Hailo-8 limit; then part_fix.py makes the cuts legal (R0..R16) and writes the sub-compile hints.

  python part_pack.py Q.har COST_DIR OUT_DIR [--caps 0.5,0.6,0.7]   -> OUT_DIR/p{CAP}/c{i}.layers.json
Limits per context (Hailo-8, 8 clusters): 512 subclusters, 128 layer controllers (HW nodes), 160 APUs, 256 input aligners.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from part_fix import fix, hn_of  # noqa: E402

LIMITS = {"subclusters": 512, "hw_nodes": 128, "apus": 160, "input_aligners": 256}


def load_costs(d):
    c = {}
    for f in glob.glob(os.path.join(d, "*.json")):
        r = json.load(open(f))
        if r.get("ok"):
            c[r["layer"]] = {k: r["res_sum"].get(k, 0.0) for k in LIMITS}
        elif "layer" in r:
            c[r["layer"]] = None  # did not compile alone
    return c


def pack(har, cost_dir, out, caps):
    hn = hn_of(har); L = hn["layers"]
    order = [n for n, v in L.items() if v["type"] not in ("input_layer", "output_layer")]
    pos = {n: i for i, n in enumerate(order)}
    for n in order:
        for s in L[n]["input"]:
            assert s not in pos or pos[s] < pos[n], f"HN order not topological at {n}"
    cost = load_costs(cost_dir)
    short = lambda n: n.split("/")[-1]  # noqa: E731
    missing = [short(n) for n in order if L[n]["type"] != "const_input" and short(n) not in cost]
    alone = [short(n) for n in order if cost.get(short(n), 0) is None]
    if missing:
        print(f"WARNING {len(missing)} layers without a measurement (counted 0): {missing[:8]}")
    if alone:
        print(f"WARNING {len(alone)} layers did not compile alone (counted at the context limit): {alone}")
    tot = {k: sum((cost.get(short(n)) or {}).get(k, 0.0) for n in order) for k in LIMITS}
    print("model totals", {k: round(v) for k, v in tot.items()},
          "lower bound contexts", {k: round(tot[k] / LIMITS[k], 2) for k in LIMITS})
    made = []
    for cap in caps:
        parts, cur, acc = [], [], {k: 0.0 for k in LIMITS}
        for n in order:
            c = cost.get(short(n))
            c = {k: LIMITS[k] * cap for k in LIMITS} if c is None else (c or {k: 0.0 for k in LIMITS})
            if cur and any(acc[k] + c[k] > cap * LIMITS[k] for k in LIMITS):
                parts.append(cur); cur, acc = [], {k: 0.0 for k in LIMITS}
            cur.append(n)
            for k in LIMITS:
                acc[k] += c[k]
        if cur:
            parts.append(cur)
        d = os.path.join(out, f"p{cap}")
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(d):
            if f.startswith("c") and f.endswith((".layers.json", ".fs.alls", ".extra.alls")):
                os.remove(os.path.join(d, f))
        for i, p in enumerate(parts):
            json.dump(p, open(os.path.join(d, f"c{i}.layers.json"), "w"))
        moves, sizes = fix(har, d)
        load = []
        i = 0
        while os.path.exists(os.path.join(d, f"c{i}.layers.json")):
            ls = json.load(open(os.path.join(d, f"c{i}.layers.json")))
            load.append(max(sum((cost.get(short(n)) or {}).get(k, 0.0) for n in ls) / LIMITS[k] for k in LIMITS))
            i += 1
        print(f"{d}: {len(sizes)} contexts sizes {sizes} moves {len(moves)} worst-resource load "
              f"{[round(x, 2) for x in load]}")
        made.append(d)
    return made


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("har"); ap.add_argument("costs"); ap.add_argument("out")
    ap.add_argument("--caps", default="0.5,0.6,0.7"); a = ap.parse_args()
    pack(a.har, a.costs, a.out, [float(x) for x in a.caps.split(",")])
