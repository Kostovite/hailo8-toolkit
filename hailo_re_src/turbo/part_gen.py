"""Candidate context partitions from the HN alone (no compiler search): per-layer MACs and weight bytes, the
layers in topological order cut into K contiguous contexts that balance a cost mix, then part_fix.py rules.

Usage: python part_gen.py FULL.har OUT_DIR [--k 2,3,4] [--mix 0,0.5,1]   -> OUT_DIR/k{K}_m{MIX}/c{i}.layers.json
       python part_gen.py FULL.har --show PART_DIR [PART_DIR ...]          (cost share per context of a partition)
cost(layer) = mix * MAC share + (1 - mix) * weight share; cuts by greedy prefix balance on the cumulative cost.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from part_fix import fix, hn_of  # noqa: E402


def costs(hn):
    out = {}
    for n, v in hn["layers"].items():
        if v["type"] in ("input_layer", "output_layer"):
            continue
        p = v.get("params", {})
        k = p.get("kernel_shape")
        oh, ow, oc = (v["output_shapes"][0][1:4] + [1, 1, 1])[:3] if v.get("output_shapes") else (1, 1, 1)
        w = 0
        if k and len(k) == 4:
            kh, kw, ci, co = k
            w = kh * kw * ci * co
            if v["type"] == "dw":
                w = kh * kw * max(ci, co)
        mac = w * oh * ow
        if v["type"] == "matmul":
            ins = v.get("input_shapes", [])
            if len(ins) == 2:
                a_, b_ = ins
                mac = a_[1] * a_[2] * a_[3] * b_[3]
        out[n] = {"mac": mac, "w": w, "act": oh * ow * oc}
    return out


def show(har, dirs):
    hn = hn_of(har)
    c = costs(hn)
    tm, tw = sum(x["mac"] for x in c.values()), sum(x["w"] for x in c.values())
    full = lambda s: s if "/" in s else f"{hn['name']}/{s}"  # noqa: E731
    for d in dirs:
        i, rows = 0, []
        while os.path.exists(f"{d}/c{i}.layers.json"):
            ls = [full(n) for n in json.load(open(f"{d}/c{i}.layers.json"))]
            m = sum(c[n]["mac"] for n in ls if n in c)
            w = sum(c[n]["w"] for n in ls if n in c)
            rows.append(f"c{i}: {len(ls)} layers MAC {m / tm:.2f} ({m / 1e9:.2f} G) W {w / tw:.2f} ({w / 1e6:.2f} MB)")
            i += 1
        print(d, "|", " | ".join(rows))


def gen(har, out, ks, mixes):
    hn = hn_of(har)
    c = costs(hn)
    order = [n for n in hn["layers"] if n in c]
    pos = {n: i for i, n in enumerate(order)}
    for n in order:  # the HN order must be topological for contiguous cuts to be legal
        for s in hn["layers"][n]["input"]:
            assert s not in pos or pos[s] < pos[n], f"HN order not topological at {n}"
    tm, tw = sum(x["mac"] for x in c.values()) or 1, sum(x["w"] for x in c.values()) or 1
    made = []
    for k in ks:
        for mix in mixes:
            cost = [mix * c[n]["mac"] / tm + (1 - mix) * c[n]["w"] / tw for n in order]
            tot, acc, cuts = sum(cost), 0.0, []
            for i, x in enumerate(cost):
                acc += x
                if len(cuts) < k - 1 and acc >= tot * (len(cuts) + 1) / k:
                    cuts.append(i + 1)
            bounds = [0] + cuts + [len(order)]
            d = f"{out}/k{k}_m{mix}"
            os.makedirs(d, exist_ok=True)
            for f in os.listdir(d):
                if f.startswith("c") and f.endswith((".layers.json", ".fs.alls")):
                    os.remove(f"{d}/{f}")
            for i in range(k):
                json.dump(order[bounds[i]:bounds[i + 1]], open(f"{d}/c{i}.layers.json", "w"))
            moves, sizes = fix(har, d)
            made.append(d)
            print(d, "sizes", sizes, "moves", len(moves))
    return made


if __name__ == "__main__":
    if "--show" in sys.argv:
        show(sys.argv[1], sys.argv[sys.argv.index("--show") + 1:])
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("har")
        ap.add_argument("out")
        ap.add_argument("--k", default="2,3,4")
        ap.add_argument("--mix", default="0,0.5,1")
        a = ap.parse_args()
        gen(a.har, a.out, [int(x) for x in a.k.split(",")], [float(x) for x in a.mix.split(",")])
