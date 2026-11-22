"""Measured-objective partition search: move context boundaries, re-sub-compile only the changed contexts (hcc's
cache makes repeats free), score each partition WITHOUT assembling or benchmarking it:

  T(b) = sum_c lat_c + (b - 1) * sum_c 1/fps_c + K * S        (S = context switch, fitted 2.6 ms on yolo26m:
  FPS(b) = b / T(b)                                            stock-15 23.4/40.0 vs measured 23.4/39.2,
  score  = geometric mean of FPS over --batches                r12 24.6/42.9 vs 24.6/39.6)

per context the variant that minimises its share of the objective is taken. A move pushes the last k layers of
context i into i+1 or the first k of i+1 into i (topological order, legal moves only), then part_fix (R1 off,
PART_FIX_NO_R1) makes it legal. Hand-written per-layer recipes (PART_DIR/c{i}.extra.alls: attention pins,
remove_node, defuse) follow their layer into whichever context holds it.

Usage (DFC env): python part_refine.py FULL.har START_PART WORK_DIR [--iters 6] [--ks 2,6,12] [--batches 1,4]
  -> WORK_DIR/best_part (+ WORK_DIR/refine.jsonl: every evaluated partition); then hcc.py on best_part
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hcc  # noqa: E402
from part_fix import hn_of  # noqa: E402

R = hcc.R
S = 2.6e-3


def load_part(d):
    parts, i = [], 0
    while os.path.exists(f"{d}/c{i}.layers.json"):
        parts.append(json.load(open(f"{d}/c{i}.layers.json")))
        i += 1
    extras = {}  # layer short name -> recipe lines
    for i in range(len(parts)):
        p = f"{d}/c{i}.extra.alls"
        if os.path.exists(p):
            for ln in open(p).read().splitlines():
                m = re.match(r"^\s*(?:[\w, -]+=\s*)?\w+\(\s*([\w-]+)", ln)
                if m:  # pieces of a defuse (matmul3_d0, _dc, _rs, _fs) belong to their base layer
                    extras.setdefault(re.sub(r"_(d\d+|dc|rs|fs)$", "", m.group(1)), []).append(ln)
    return parts, extras


def write_part(d, parts, extras, har):
    os.makedirs(d, exist_ok=True)
    for f in os.listdir(d):
        os.remove(f"{d}/{f}")
    for i, p in enumerate(parts):
        json.dump(p, open(f"{d}/c{i}.layers.json", "w"))
    out = subprocess.run([sys.executable, f"{R}/turbo/part_fix.py", har, d], capture_output=True, text=True,
                         env=dict(os.environ, PART_FIX_NO_R1="1"))
    parts, _ = load_part(d)
    for i, p in enumerate(parts):  # recipes follow their layers (part_fix rewrote extra.alls with its own pins)
        mine = [ln for n in p for ln in extras.get(n.split("/")[-1], [])]
        if mine:
            xp = f"{d}/c{i}.extra.alls"
            have = open(xp).read().splitlines() if os.path.exists(xp) else []
            keep = [h for h in have if not any(h.startswith(f"compilation_param({n.split('/')[-1]},") for n in p
                                                if n.split("/")[-1] in extras)]
            body = [ln for ln in mine if ln.startswith("remove_node")] + \
                   [ln for ln in mine if not ln.startswith("remove_node")] + keep
            open(xp, "w").write("\n".join(dict.fromkeys(body)) + "\n")
    return parts, out.stdout.strip()


def score(work, n, variants, batches):
    """best predicted geometric-mean FPS over the variants that compiled; None if a context has none."""
    res = []
    for i in range(n):
        got = [j for v in variants for j in [hcc.ok(f"{work}/c{i}_{v}")] if j]
        if not got:
            return None, None
        res.append([(j["fps"], j["latency_ms"] / 1e3) for j in got])
    # per context choose the variant minimising sum over batches of its T(b) share (independent across contexts)
    pick = [min(r, key=lambda x: sum(x[1] + (b - 1) / x[0] for b in batches)) for r in res]
    fps = {b: b / (sum(l for _, l in pick) + (b - 1) * sum(1 / f for f, _ in pick) + n * S) for b in batches}
    return math.exp(sum(math.log(v) for v in fps.values()) / len(fps)), fps


def moves(parts, L, ks):
    ctx = {n: i for i, p in enumerate(parts) for n in p}
    out = []
    for i in range(len(parts) - 1):
        for k in ks:
            for direction in ("fwd", "back"):
                new = [list(p) for p in parts]
                src, dst = (i, i + 1) if direction == "fwd" else (i + 1, i)
                moved = 0
                order = new[src][::-1] if direction == "fwd" else list(new[src])
                for n in order:
                    if moved >= k:
                        break
                    if direction == "fwd":  # every consumer already in a later context
                        legal = all(ctx.get(o, 99) > i for o in L[n]["output"] if o in ctx) and \
                            all(ctx.get(o, 99) != i or o not in new[i] for o in L[n]["output"] if o in ctx)
                    else:  # every producer already in an earlier-or-same context
                        legal = all(ctx.get(s, -1) <= i for s in L[n]["input"] if s in ctx) and \
                            all(s not in new[i + 1] for s in L[n]["input"])
                    if legal:
                        new[src].remove(n)
                        (new[dst].insert(0, n) if direction == "fwd" else new[dst].append(n))
                        ctx[n] = dst
                        moved += 1
                for n in parts[src]:
                    ctx[n] = next(j for j, p in enumerate(parts) if n in p)
                if moved and all(new):
                    out.append((f"b{i}{'+' if direction == 'fwd' else '-'}{moved}", new))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("har")
    ap.add_argument("start")
    ap.add_argument("work")
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--ks", default="2,6,12")
    ap.add_argument("--batches", default="1,4")
    ap.add_argument("--variants", default="default,lat,default_sr,lat_sr,lat_fs,lat_fs_sr")
    ap.add_argument("--cap", type=float, default=1500)
    ap.add_argument("--width", type=int, default=12, help="candidates evaluated per iteration")
    a = ap.parse_args()
    har = os.path.abspath(a.har)
    L = hn_of(har)["layers"]
    ks, batches, variants = [int(x) for x in a.ks.split(",")], [int(x) for x in a.batches.split(",")], \
        a.variants.split(",")
    os.makedirs(a.work, exist_ok=True)
    log = open(f"{a.work}/refine.jsonl", "a")
    parts, extras = load_part(a.start)
    cur_dir = f"{a.work}/p0"
    parts, _ = write_part(cur_dir, parts, extras, har)

    def evaluate(d, n):
        w = d + "_w"
        os.makedirs(w, exist_ok=True)
        hcc.sub_compiles(SimpleNamespace(har=har, part=d, work=w, variants=a.variants, cap=a.cap), n)
        return score(w, n, variants, batches)
    best_s, best_f = evaluate(cur_dir, len(parts))
    print(f"start {best_s and round(best_s, 2)} {best_f}", flush=True)
    log.write(json.dumps({"it": 0, "dir": cur_dir, "score": best_s, "fps": best_f}) + "\n")
    log.flush()
    for it in range(1, a.iters + 1):
        w = cur_dir + "_w"
        lat = []
        for i in range(len(parts)):  # rank boundaries by the latency of the contexts on either side
            got = [hcc.ok(f"{w}/c{i}_{v}") for v in variants]
            lat.append(min((j["latency_ms"] for j in got if j), default=99))
        cands = moves(parts, L, ks)
        cands.sort(key=lambda c: -max(lat[int(re.match(r"b(\d+)", c[0]).group(1))],
                                      lat[int(re.match(r"b(\d+)", c[0]).group(1)) + 1]))
        cands = cands[:a.width]
        dirs = []
        for tag, new in cands:
            d = f"{a.work}/it{it}_{tag}"
            p2, fx = write_part(d, new, extras, har)
            dirs.append((tag, d, p2, fx))
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(len(dirs)) as ex:
            got = list(ex.map(lambda x: evaluate(x[1], len(x[2])), dirs))
        improved = False
        for (tag, d, p2, fx), (s_, f_) in zip(dirs, got):
            log.write(json.dumps({"it": it, "dir": d, "tag": tag, "score": s_, "fps": f_, "fix": fx[-150:],
                                  "sizes": [len(p) for p in p2]}) + "\n")
            log.flush()
            print(f"it{it} {tag:8s} {s_ and round(s_, 2)} {f_ and {b: round(v, 2) for b, v in f_.items()}}",
                  flush=True)
            if s_ and (best_s is None or s_ > best_s * 1.002):
                best_s, best_f, cur_dir, parts, improved = s_, f_, d, p2, True
        print(f"iter {it}: best {round(best_s, 2)} {best_f} at {cur_dir}", flush=True)
        if not improved:
            break
    if os.path.exists(f"{a.work}/best_part"):
        shutil.rmtree(f"{a.work}/best_part")
    shutil.copytree(cur_dir, f"{a.work}/best_part")
    print("BEST", cur_dir, best_s, best_f)


if __name__ == "__main__":
    main()
