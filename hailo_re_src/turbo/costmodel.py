"""Analytic context cost model from the measured Hailo-8 laws (probe/RULES.md 10-14), no compile needed.

Per layer l: work_l (SC-cycles per frame) and cap_l (most SCs it can use):
  conv   work = MAC / 50,  cap = ceil(o/8) * ceil(W/8) tiles (8 out-channels x 8 columns per SC)
  dw     work = MAC / 2,   cap = 16 pieces (INPUT_FEATURES defuse) * ceil(W/8)
  matmul attention: fixed  heads * N^2 / 7.5 cycles (softmax/elementwise over the score maps)
  any    stream floor: out_bytes / 8 cycles per output stream, split up to ceil(W/8) ways
A single context is a balanced pipeline: cycles = max(sum(work) / S_usable, max_l work_l / cap_l, max_l floor_l).
Batch-1 latency = fill: cycles + depth * row term (fitted).

Usage: python costmodel.py COSTDB.jsonl   (fits the 3 free constants, prints the validation)
       from costmodel import predict; predict(hn_layers_subset) -> {"cycles":..., "fps":..., "lat_ms":...}
"""
import json
import math
import sys

K = {"S": 384.0, "conv": 50.0, "dw": 2.0, "attn": 7.5, "stream": 8.0, "row": 1.0, "fill": 1.0}


def layer_terms(v, k=K):
    t = v["type"]
    p = v.get("params", {})
    out = v.get("output_shapes") or [[-1, 1, 1, 1]]
    h, w, o = (list(out[0][1:4]) + [1, 1, 1])[:3]
    kern = p.get("kernel_shape")
    floor = h * w * o / k["stream"] / max(1, math.ceil(w / 8))
    a16 = "a16" in (v.get("quantization_params", {}).get("precision_mode") or "")
    if t == "conv" and kern and len(kern) == 4:
        mac = kern[0] * kern[1] * kern[2] * kern[3] * h * w / max(1, p.get("groups", 1) or 1)
        # 16-bit: 3x3 2.8, 1x1 7.4 MAC/SC/cycle vs ~50 at 8-bit (RULES.md 4)
        work = mac / ((2.8 if kern[0] * kern[1] > 1 else 7.4) if a16 else k["conv"])
        cap = math.ceil(o / 8) * math.ceil(w / 8)
        return work, cap, floor, 0.0
    if t == "dw" and kern:
        mac = kern[0] * kern[1] * max(kern[2], kern[3]) * h * w
        return mac / (k["dw"] * (0.85 if a16 else 1)), 16 * math.ceil(w / 8), floor * (2 if a16 else 1), 0.0
    if t == "matmul":
        ins = v.get("input_shapes", [])
        n = ins[0][2] if ins else w
        heads = ins[0][1] if ins and ins[0][1] > 1 else 1
        return 0.0, 1, floor, heads * n * n / k["attn"]
    return 0.0, 1, floor, 0.0


def predict(layers, k=K):
    """layers: iterable of HN layer dicts of ONE context (cut IO excluded)."""
    work = cap_t = fl = at = 0.0
    depth = 0
    for v in layers:
        if v["type"] in ("input_layer", "output_layer"):
            continue
        wk, cap, f, a = layer_terms(v, k)
        work += wk
        cap_t = max(cap_t, wk / max(cap, 1))
        fl = max(fl, f)
        at = max(at, a)
        depth += 1
    cycles = max(work / k["S"], cap_t, fl, at)
    return {"cycles": cycles, "fps": 400e6 / cycles, "lat_ms": (k["fill"] * cycles + k["row"] * depth * 1000) / 400e3}


if __name__ == "__main__":
    import itertools
    import os
    import tarfile
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from costdb import C, hn_of
    R = [json.loads(x) for x in open(sys.argv[1])]
    data = []
    for r in R:
        b = r["best_fps"]
        hn = hn_of(f"{C}/{b['key']}.har")
        data.append((list(hn["layers"].values()), b["fps"], r["best_lat"]["lat_ms"], b["model"]))

    def err(k):
        e = [abs(math.log(predict(L, k)["fps"] / f)) for L, f, _, _ in data]
        return sorted(e)[len(e) // 2], sorted(e)[int(.9 * len(e))]
    best = None
    for S, cv, st in itertools.product((256, 320, 384, 448), (40, 50, 60), (4, 6, 8, 12)):
        k = dict(K, S=S, conv=cv, stream=st)
        m, p90 = err(k)
        if best is None or m < best[0]:
            best = (m, p90, k)
    m, p90, k = best
    print(f"fps: median |log err| {m:.3f} (x{math.exp(m):.2f}), p90 x{math.exp(p90):.2f}  constants S={k['S']} "
          f"conv={k['conv']} stream={k['stream']}")
    from scipy.stats import spearmanr
    pr = [predict(L, k)["fps"] for L, *_ in data]
    print("spearman fps", round(spearmanr(pr, [f for _, f, _, _ in data]).correlation, 3))
    # which term binds, and the worst misses
    rows = sorted(((abs(math.log(predict(L, k)["fps"] / f)), f, predict(L, k)["fps"], md, len(L))
                   for L, f, _, md in data), reverse=True)[:8]
    for e, f, p_, md, n in rows:
        print(f"  miss x{math.exp(e):.2f}  real {f:9.1f} pred {p_:9.1f}  {md} {n} layers")
    json.dump(k, open(os.path.join(os.path.dirname(sys.argv[1]), "costmodel_k.json"), "w"))
