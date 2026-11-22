"""Parallel local search over context cut points, scored on the chips.

State = K-1 cut positions in the HN topological order. Each round evaluates every neighbour (one cut moved by
+-step) with hcc.py in parallel (sub-compiles shared through hcc's cache), score = hcc's geometric-mean FPS over
--batches; moves to the best neighbour while it improves, then halves the step.

Usage (DFC env): python part_search.py FULL.har OUT_DIR --k 3 --mix 0 [--steps 16,8,4] [--par 8] [--rounds 6]
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from part_fix import fix, hn_of  # noqa: E402
from part_gen import costs  # noqa: E402

R = "/mnt/storage/lib_project/models/Hailo_model/hailo_re"


def seed_cuts(order, c, k, mix):
    tm, tw = sum(x["mac"] for x in c.values()) or 1, sum(x["w"] for x in c.values()) or 1
    cost = [mix * c[n]["mac"] / tm + (1 - mix) * c[n]["w"] / tw for n in order]
    tot, acc, cuts = sum(cost), 0.0, []
    for i, x in enumerate(cost):
        acc += x
        if len(cuts) < k - 1 and acc >= tot * (len(cuts) + 1) / k:
            cuts.append(i + 1)
    return tuple(cuts)


def evaluate(a, order, cuts):
    tag = "c" + "-".join(map(str, cuts))
    d = f"{a.out}/{tag}"
    if os.path.exists(f"{d}/work/hcc.json"):
        j = json.load(open(f"{d}/work/hcc.json"))
    else:
        os.makedirs(f"{d}/part", exist_ok=True)
        b = (0,) + cuts + (len(order),)
        for i in range(len(b) - 1):
            json.dump(order[b[i]:b[i + 1]], open(f"{d}/part/c{i}.layers.json", "w"))
        fix(a.har, f"{d}/part")
        cmd = [sys.executable, "-u", f"{R}/turbo/hcc.py", a.har, f"{d}/part", f"{d}/work", "--cap", str(a.cap),
               "--variants", a.variants, "--batches", a.batches]
        if a.ref:
            cmd += ["--ref", a.ref]
        subprocess.run(cmd, stdout=open(f"{d}/hcc.out", "w"), stderr=subprocess.STDOUT, cwd=R)
        try:
            j = json.load(open(f"{d}/work/hcc.json"))
        except (OSError, ValueError):
            j = None
    if not j:
        return cuts, 0.0, None
    f = j["fps"]
    return cuts, math.exp(sum(math.log(max(v, 1e-9)) for v in f.values()) / len(f)), f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("har")
    ap.add_argument("out")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--mix", type=float, default=0.0)
    ap.add_argument("--start", help="comma-separated cut positions instead of the cost-balanced seed")
    ap.add_argument("--steps", default="16,8,4")
    ap.add_argument("--par", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--cap", type=float, default=300)
    ap.add_argument("--variants", default="default,lat,lat_fs")
    ap.add_argument("--batches", default="1,8")
    ap.add_argument("--ref")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    hn = hn_of(a.har)
    c = costs(hn)
    order = [n for n in hn["layers"] if n in c]
    cur = tuple(int(x) for x in a.start.split(",")) if a.start else seed_cuts(order, c, a.k, a.mix)
    t0 = time.time()
    _, best, bf = evaluate(a, order, cur)
    log = open(f"{a.out}/search.log", "a")
    print(f"seed {cur} score {best:.1f} {bf}", file=log, flush=True)
    for step in [int(s) for s in a.steps.split(",")]:
        for _ in range(a.rounds):
            nb = set()
            for j in range(len(cur)):
                for dlt in (-step, step):
                    x = list(cur)
                    x[j] += dlt
                    if all(0 < x[i] < len(order) for i in range(len(x))) and all(
                            x[i] < x[i + 1] for i in range(len(x) - 1)):
                        nb.add(tuple(x))
            with ThreadPoolExecutor(a.par) as ex:
                res = list(ex.map(lambda cu: evaluate(a, order, cu), sorted(nb)))
            for cu, sc, f in res:
                print(f"  step {step} {cu} score {sc:.1f} {f}", file=log, flush=True)
            cu, sc, f = max(res, key=lambda r: r[1]) if res else (cur, 0, None)
            if sc <= best:
                break
            cur, best, bf = cu, sc, f
            print(f"move -> {cur} score {best:.1f} {bf}  ({time.time() - t0:.0f} s)", file=log, flush=True)
    print(f"BEST {cur} score {best:.1f} {bf} total {time.time() - t0:.0f} s", file=log, flush=True)
    json.dump({"cuts": cur, "score": best, "fps": bf, "dir": f"{a.out}/c" + "-".join(map(str, cur)),
               "total_s": round(time.time() - t0)}, open(f"{a.out}/best.json", "w"), indent=1)
    print(open(f"{a.out}/search.log").read().splitlines()[-1])


if __name__ == "__main__":
    main()
