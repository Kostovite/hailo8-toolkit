"""ctx_grow.py : context partition found by COMPILING, not estimating.

The DFC Slotter starts from a per-node resource estimate that is far off for 16-bit nets (probe/RULES.md section 29):
on PP-DocLayoutV3 mix16 every initial context was overloaded and the search ran > 4 h without one valid partition.
Here a context is grown along the HN's topological order by real single-context compiles of candidate layer spans
(subhar.py cut, context_switch_param(mode=disabled) so a span that does not fit one context fails instead of being
split), several candidate ends in parallel per round:
  round 1: ends s+step, s+2*step, ... (P spans at once); keep the longest that compiles
  round 2: the ends between it and the first failure, one by one in parallel
then the next context starts after it. Every context of the result compiled alone, so the partition is feasible
context by context; turbo/hcc.py (or assemble.py + compile) turns it into a full multi-context HEF.

  python ctx_grow.py Q.har OUT_DIR [--step 6] [--par 8] [--timeout 1200]
  -> OUT_DIR/part/c{i}.layers.json, OUT_DIR/spans/* (every test: sub HAR, log, json), OUT_DIR/grow.log
"""
import argparse
import json
import os
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
SKIP = ("input_layer", "output_layer")
NO_START = ("format_conversion", "const_input", "feature_multiplier")  # RULES R10/R13: a context may not start here


def hn_of(har):
    with tarfile.open(har) as t:
        n = next(m for m in t.getnames() if m.endswith(".hn") and not m.endswith((".native.hn", ".fp.hn")))
        return json.loads(t.extractfile(n).read())


def span_layers(L, order, a, b):
    """short names of order[a:b] + the const_input producers they consume (R0)"""
    mem = set(order[a:b])
    for n in order[a:b]:
        for p in L[n]["input"]:
            if L.get(p, {}).get("type") == "const_input":
                mem.add(p)
    return [n.split("/")[-1] for n in sorted(mem, key=lambda x: list(L).index(x))]


WORKER = r'''
import json, os, sys, time, traceback
sys.path.insert(0, sys.argv[4])
import subhar
from hailo_sdk_client import ClientRunner
har, layers, out = sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3]
os.makedirs(out + ".d", exist_ok=True); os.chdir(out + ".d")
res = {"n": len(layers)}; t0 = time.time()
try:
    res.update(subhar.build(har, layers, out + ".har"))
except Exception as e:
    res.update(ok=False, error=f"{type(e).__name__}: {str(e)[:240]}")
# RULES 29F: map::at in "Trying to apply higher utilization solution" (the span FITS; the compiler crashes while
# spreading it over the idle chip) -> retry with a lower throughput target, highest first
LADDER = [int(x) if x != "default" else None for x in os.environ.get("CTXG_FPS_LADDER", "20").split(",")]
for fps in LADDER if "error" not in res else []:  # default "20": the minimum-footprint fit test only (fast, no map::at)
    try:
        r = ClientRunner(har=out + ".har")
        r.load_model_script("context_switch_param(mode=disabled)\n" + (f"performance_param(fps={fps})\n" if fps else ""))
        r.compile()
        res.update(ok=True, fps_target=fps); res.pop("error", None)
        break
    except Exception as e:
        res.update(ok=False, error=f"{type(e).__name__}: {str(e)[:240]}", fps_target=fps)
        # any failure at the default level: retry at the minimum footprint (the real "fits" test)
res["s"] = round(time.time() - t0, 1)
json.dump(res, open(out + ".json", "w"))
'''


def test(args, L, order, a, b):
    """True if order[a:b] compiles in one context (cached on disk)."""
    tag = f"s{a:03d}_{b:03d}"
    d = os.path.join(args.out, "spans")
    j = os.path.join(d, tag + ".json")
    if not os.path.exists(j):
        lj = os.path.join(d, tag + ".layers.json")
        json.dump(span_layers(L, order, a, b), open(lj, "w"))
        sdk = os.path.join(d, tag + ".sdk", "etc", "hailo", "dfc"); os.makedirs(sdk, exist_ok=True)
        env = dict(os.environ, VIRTUAL_ENV=os.path.join(d, tag + ".sdk"), CONDA_PREFIX=os.path.join(d, tag + ".sdk"),
                   HAILO_CPSERVER="", CUDA_VISIBLE_DEVICES="99", OMP_NUM_THREADS="4")
        with open(os.path.join(d, tag + ".log"), "w") as lg:
            try:
                unit = f"ctxgrow-{os.getpid()}-{tag}"
                pr = subprocess.Popen(["systemd-run", "--user", "--scope", "-q", f"--unit={unit}", "-p",
                                       f"MemoryMax={args.mem}", "nice", "-n", "5",
                                       sys.executable, "-u", "-c", WORKER, args.har, lj, os.path.join(d, tag), HERE],
                                      stdout=lg, stderr=subprocess.STDOUT, env=env)
                pr.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                # kill the WHOLE scope: killing systemd-run alone leaves the DFC compiler running for hours
                subprocess.run(["systemctl", "--user", "kill", "--signal=KILL", f"{unit}.scope"], capture_output=True)
                pr.wait()
                json.dump({"ok": False, "error": "timeout"}, open(j, "w"))
        if not os.path.exists(j):
            json.dump({"ok": False, "error": "worker died"}, open(j, "w"))
    r = json.load(open(j))
    if "unexpected crash" in str(r.get("error", "")) and not getattr(test, "_retry", False):
        os.remove(j)  # a DFC crash is not "does not fit": retry once, never cache it
        test._retry = True
        try:
            return test(args, L, order, a, b)
        finally:
            test._retry = False
    return bool(r.get("ok")), r


def legal_end(L, order, b):
    return b >= len(order) or L[order[b]]["type"] not in NO_START and not order[b].split("/")[-1].startswith("precision_change")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("har"); ap.add_argument("out")
    ap.add_argument("--step", type=int, default=6); ap.add_argument("--par", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=1200); ap.add_argument("--mem", default="20G")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "spans"), exist_ok=True); os.makedirs(os.path.join(args.out, "part"), exist_ok=True)
    hn = hn_of(args.har); L = hn["layers"]
    order = [n for n, v in L.items() if v["type"] not in SKIP and v["type"] != "const_input"]
    log = open(os.path.join(args.out, "grow.log"), "a")
    say = lambda m: (print(m, flush=True), log.write(m + "\n"), log.flush())  # noqa: E731
    say(f"{time.strftime('%H:%M:%S')} {len(order)} layers, step {args.step}, par {args.par}")
    pool = ThreadPoolExecutor(args.par)
    s, ctxs = 0, []
    while s < len(order):
        t0 = time.time()
        ends = [e for e in range(s + args.step, s + args.step * args.par + 1, args.step)] + [len(order)]
        ends = sorted({min(e, len(order)) for e in ends})
        ends = [e for e in ends if legal_end(L, order, e)] or [min(s + 1, len(order))]
        res = dict(zip(ends, pool.map(lambda e: test(args, L, order, s, e)[0], ends)))
        good = [e for e in ends if res[e]]
        if not good:  # even the shortest step failed or timed out: every shorter end, in parallel
            short = [e for e in range(s + 1, min(s + args.step, len(order) + 1)) if legal_end(L, order, e)]
            r3 = dict(zip(short, pool.map(lambda e: test(args, L, order, s, e), short)))
            okk = [e for e in short if r3[e][0]]
            if not okk:
                say(f"context {len(ctxs)}: layers {s}..{s + 1} do not compile alone: {r3[short[0]][1].get('error')}")
                sys.exit(2)
            best = max(okk)
        else:
            best = max(good)
            fail = min([e for e in ends if e > best and not res[e]], default=None)
            if fail and fail - best > 1:  # refine between the longest good and the first bad end
                mids = [m for m in range(best + 1, fail) if legal_end(L, order, m)]
                mids = mids[-args.par:] if len(mids) > args.par else mids
                r2 = dict(zip(mids, pool.map(lambda e: test(args, L, order, s, e)[0], mids)))
                best = max([best] + [m for m in mids if r2[m]])
        ctxs.append((s, best))
        json.dump(span_layers(L, order, s, best), open(os.path.join(args.out, "part", f"c{len(ctxs) - 1}.layers.json"), "w"))
        say(f"{time.strftime('%H:%M:%S')} context {len(ctxs) - 1}: layers {s}..{best - 1} ({best - s}) "
            f"{order[s].split('/')[-1]} .. {order[best - 1].split('/')[-1]}  [{time.time() - t0:.0f} s]")
        s = best
    say(f"{time.strftime('%H:%M:%S')} DONE {len(ctxs)} contexts: {[b - a for a, b in ctxs]}")


if __name__ == "__main__":
    main()
