"""hcc: parallel multi-context compile from a partition (per-context HN layer lists).

  1. every context x every objective variant = one compile_sub.py process, all at once; at --cap seconds the
     stragglers are stopped once every context has a result
  2. per context rank the finished variants by single-context latency (a context's batch-1 time is its pipeline
     fill), then fps; keep the top --topk
  3. every combination of those (<= --combos) is assembled and fully compiled in parallel, each in its own dir:
     assemble.py -> drop buffers() -> R6 weights out of L2 (mixed_mem=disabled) -> compile_alls.py, script errors
     repaired from the compiler's own message. Feasibility depends on the combination, not on one context alone.
  4. every HEF that compiled is benchmarked on the two chips (one HEF per chip at a time) at --batches; the
     winner maximises the geometric mean FPS; it is byte-compared with the other candidates (or --ref)

Usage (DFC env): python hcc.py FULL.har PART_DIR WORK_DIR [--cap 300] [--topk 2] [--batches 1,8] [--ref X.hef]
  PART_DIR/c{i}.layers.json (part_from_alls.py / part_gen.py, then part_fix.py), optional PART_DIR/c{i}.extra.alls
  (appended to every variant of ctx i) and c{i}.fs.alls (R5 entry-layer feature defuse, used by *_fs variants)
"""
import argparse
import glob
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

R = "/mnt/storage/lib_project/models/Hailo_model/hailo_re"
PAR = int(os.environ.get("HCC_PAR", "64"))  # concurrent sub-compiles
BASE = "allocator_param(automatic_reshapes=disabled, enable_auto_spatial_reshapes=False)\n"
# R9: with auto spatial reshapes off the compiler cannot defuse attention matmuls (memory units overflow); with
# them on, a sub-network input that needs a spatial reshape breaks instead (R2) -> both, chosen per context
BASE_SR = "allocator_param(automatic_reshapes=disabled)\n"
VARIANTS = {
    "default": "",
    "lat": "performance_param(optimize_for_batch=1)\n",
    "fps2k": "performance_param(fps=2000)\n",
    "fps4k": "performance_param(fps=4000)\n",
    # RULES 29F: 16-bit contexts that crash (map::at) or time out spreading over the idle chip compile at a capped target
    "f5": "performance_param(fps=5)\n",  # 2026-11-23 layout c11: conv78 (16-bit) tops out at 9.9 fps in one context
    "f20": "performance_param(fps=20)\n",
    "f50": "performance_param(fps=50)\n",
    "f100": "performance_param(fps=100)\n",
    "f200": "performance_param(fps=200)\n",
}
DEVS = tuple(os.environ.get("HCC_DEVS", "0000:02:00.0,0000:03:00.0").split(","))  # HCC_DEVS: keep a chip free
CACHE = f"{R}/turbo/t/subcache"


def ok(pre):
    try:
        j = json.load(open(pre + ".json"))
    except (OSError, ValueError):
        return None
    return j if j.get("ok") and j.get("fps") else None


def sub_compiles(a, n):
    procs, pending = {}, []
    t0 = time.time()
    for i in range(n):
        extra = open(f"{a.part}/c{i}.extra.alls").read() if os.path.exists(f"{a.part}/c{i}.extra.alls") else ""
        fs = f"{a.part}/c{i}.fs.alls"
        for v in a.variants.split(","):
            pre = f"{a.work}/c{i}_{v}"
            flags = set(v.split("_")[1:])  # _fs: entry-layer defuse hints; _sr: auto spatial reshapes ON
            if "fs" in flags and not os.path.exists(fs):
                continue
            nd = f"{a.work}/c{i}.nd.alls"  # _nd: round-2 bottleneck relief hints (relieve.py) on top of variant v
            if "nd" in flags and not os.path.exists(nd):
                continue
            perf = VARIANTS[v.split("_")[0]]
            if "max" in flags:  # level max fixes level-1 under-allocation (dw5/dw7 25x, C2f n=1 8x; RULES.md 12)
                perf = (perf.replace(")\n", ", compiler_optimization_level=max)\n") if perf
                        else "performance_param(compiler_optimization_level=max)\n")
            md = ""
            if "md" in flags:  # _md: mixed_mem=disabled on every model layer BEFORE allocating (stock's R6 pattern,
                # RULES.md 20); forcing it after the sub-compile chose a mixed-memory allocation fails (ctx0 Splitter)
                pre_txt = extra + (open(fs).read() if "fs" in flags else "")
                gone = set(re.findall(r"remove_node\(([\w-]+)\)", pre_txt))
                pieces = {m_.group(2): [x.strip() for x in m_.group(1).split(",")]
                          for m_ in re.finditer(r"^([\w, -]+?)\s*=\s*defuse\(([\w-]+)", pre_txt, re.M)}
                names = []
                for ln_ in json.load(open(f"{a.part}/c{i}.layers.json")):
                    s_ = ln_.split("/")[-1]
                    if s_ not in gone:
                        names += pieces.get(s_, [s_])
                pinned = set(re.findall(r"compilation_param\(([\w-]+),[^)]*mixed_mem", pre_txt))  # a 2nd line resets the pin
                md = "".join(f"compilation_param({s_}, mixed_mem=disabled)\n" for s_ in names if s_ not in pinned)
            body = ((BASE_SR if "sr" in flags else BASE) + extra + (open(fs).read() if "fs" in flags else "") + md
                    + (open(nd).read() if "nd" in flags else "") + perf)
            layers = sorted(n.split("/")[-1] for n in json.load(open(f"{a.part}/c{i}.layers.json")))
            key = hashlib.sha1(json.dumps([os.path.abspath(a.har), layers, body]).encode()).hexdigest()[:16]
            cpre = f"{CACHE}/{key}"
            for ext in (".json", "_compiled.har", ".log"):  # the work dir points into the cache
                if os.path.lexists(pre + ext):
                    os.remove(pre + ext)
                os.symlink(cpre + ext, pre + ext)
            if os.path.exists(cpre + ".log"):  # compiled (or compiling / given up) by an earlier or parallel run
                continue
            open(cpre + ".alls", "w").write(body)
            open(cpre + ".log", "w").close()  # claims the key (an empty log = queued / compiling)
            pending.append(((i, v), [sys.executable, "-u", f"{R}/turbo/compile_sub.py", a.har,
                                     f"{a.part}/c{i}.layers.json", cpre, cpre + ".alls"], cpre + ".log"))

    def launch():  # at most HCC_PAR sub-compiles at once: each loads the full HAR (SmolLM2: 2.9 GB x 60 = OOM)
        while pending and sum(p.poll() is None for p in procs.values()) < PAR:
            k_, cmd, log = pending.pop(0)
            procs[k_] = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, cwd=R)
    launch()
    def have():
        return {i for i in range(n) if any(ok(f"{a.work}/c{i}_{v}") for v in a.variants.split(","))}
    while pending or (procs and any(p.poll() is None for p in procs.values())) or (
            have() != set(range(n)) and time.time() - t0 < 2 * a.cap):
        launch()
        if time.time() - t0 > a.cap and have() >= set(range(n)):
            for _, _, log in pending:  # never started: drop the claim so a later run compiles them
                os.remove(log)
            pending.clear()
            for p in procs.values():
                if p.poll() is None:
                    p.kill()
            break
        if procs and all(p.poll() is not None for p in procs.values()) and time.time() - t0 > a.cap:
            break
        time.sleep(2)
    return time.time() - t0


def repair(text, log):
    """Script-level errors of an assembled script: a helper node only a sub-compile made ('could not be found in
    scope', 'Cannot find layer') -> drop its own statements and remove it from every node list; a buffer format
    the assembled graph cannot use ("Can't set BufferFormat") -> automatic."""
    gone = set(re.findall(r"layers (?:\w+/)?([\w-]+) could not be found in scope", log))
    gone |= {x.split("/")[-1] for x in re.findall(r"Cannot find layer \['([^']+)'\]", log)}
    # "Can't set BufferFormat: DIVIDE_BY_ROWS of layer conv2 to matmul1": the sub-compile's buffer format of A was
    # chosen with its format_conversion still there (assemble removes it) -> A's format back to automatic (= stock)
    fmt = set(re.findall(r"Can't set BufferFormat: \w+ of layer ([\w-]+) to ", log))
    if fmt:
        text = "\n".join(re.sub(r"layer_output_buffers_format=\w+", "layer_output_buffers_format=automatic", ln)
                         if re.match(rf"^compilation_param\(({'|'.join(map(re.escape, fmt))}),", ln) else ln
                         for ln in text.splitlines()) + "\n"
    if not gone:
        return text
    lines = []
    for ln in text.splitlines():
        m = re.match(r"^compilation_param\(([^,]+),", ln)
        if m and m.group(1).strip() in gone:
            continue
        lm = re.match(r"^(.*?\[)([^\]]*)(\].*)$", ln)
        if lm and (" = context(" in ln or ".place(" in ln):
            items = [x.strip() for x in lm.group(2).split(",") if x.strip() and x.strip() not in gone]
            if not items:
                continue
            ln = lm.group(1) + ", ".join(items) + lm.group(3)
        lines.append(ln)
    return "\n".join(lines) + "\n"


HELPER = re.compile(r"^(sh_from_|ws_from_|l4_portal_from_|ddr_portal_from_|auto_reshape_from_|concat_w_from_)")


def r6_stock(text):
    """R6 as stock writes it (yolo26m auto.alls: 319 disabled / 30 allowed): mixed_mem=disabled on every model
    layer and its defuse pieces, `allowed` on compiler helper nodes (shortcuts, weight shortcuts, L4/DDR portals,
    auto reshapes, concat_w stitching). Blanket `disabled` on the helpers too broke yolo26m (Splitter failed,
    "l4_portal_... not found in toposort recipe"); keeping the sub-compiles' `enabled` costs ~1.8 ms of sequencer
    time per context on the chip (fw actions: 0.3-1.0 ms with stock's pattern) = b1 23.4 vs stock 26.3."""
    out = []
    for ln in text.splitlines():
        m = re.match(r"^compilation_param\(([\w-]+),", ln)
        if m and "mixed_mem=" in ln:
            ln = re.sub(r"mixed_mem=\w+", "mixed_mem=" + ("allowed" if HELPER.match(m.group(1)) else "disabled"), ln)
        out.append(ln)
    return "\n".join(out) + "\n"


def full_compile(har, subs, d):
    """assemble -> drop buffers -> R6 -> compile in dir d; script-level errors repaired in place."""
    os.makedirs(d, exist_ok=True)
    out = subprocess.run([sys.executable, f"{R}/turbo/assemble.py", f"{d}/full_b.alls", *subs], cwd=R,
                         capture_output=True, text=True)
    if out.returncode:
        return False, out.stderr[-3000:]
    text = open(f"{d}/full_b.alls").read()
    text = "\n".join(ln for ln in text.splitlines() if not ln.startswith("buffers(")) + "\n"
    r6 = r6_stock(text)
    text = r6 if r6 != text else text
    log = ""
    for attempt in range(7):
        open(f"{d}/full.alls", "w").write(text)
        out = subprocess.run([sys.executable, "-u", f"{R}/turbo/t/compile_alls.py", har, f"{d}/full.alls",
                              f"{d}/full.hef"], cwd=R, capture_output=True, text=True)
        log = out.stdout + out.stderr
        open(f"{d}/full.{attempt}.log", "w").write(log)
        if os.path.exists(f"{d}/full.hef") and "COMPILE_S" in log:
            return True, log
        text2 = repair(text, log)
        if text2 == text and r6 is not None and "mixed_mem=disabled" in text and re.search(
                r"Splitter failed|Parallel activation required", log):
            # R6 is not always feasible (yolo26m ctx0): fall back to each sub-compile's own mixed_mem choice
            text2 = "\n".join(ln for ln in open(f"{d}/full_b.alls").read().splitlines()
                               if not ln.startswith("buffers(")) + "\n"
            text2 = repair(text2, log)
            r6 = None
            open(f"{d}/r6_off", "w").write("")
        if text2 == text:
            break
        text = text2
    return False, log


def chip_lock(dev):
    """lock file of one Hailo-8 (every chip user takes the lock of the device it touches)."""
    return f"{R}/turbo/.chip_{dev.replace(':', '_')}.lock"


class chips:
    """Exclusive use of the Hailo-8s this run touches (DEVS, or DEVICE for the exact check), one lock per chip so
    jobs on different chips run in parallel; never share a measurement device. Locks taken in sorted order."""
    def __init__(self, devs=None):
        self.devs = sorted(set(devs or DEVS))

    def __enter__(self):
        import fcntl
        self.f = [open(chip_lock(d), "w") for d in self.devs]
        for f in self.f:
            fcntl.flock(f, fcntl.LOCK_EX)

    def __exit__(self, *exc):
        import fcntl
        for f in self.f:
            fcntl.flock(f, fcntl.LOCK_UN)
            f.close()


def bench(hefs, batches, one_chip=False, t=5):
    """{hef: {batch: fps}}; two HEFs at a time, one per chip (the chips do not disturb each other: bench2.sh)."""
    def one(h, dev, b):
        out = subprocess.run(["hailortcli", "benchmark", h, "-t", str(t), "--batch-size", str(b), "--device-id", dev],
                             capture_output=True, text=True).stdout
        m = re.search(r"FPS\s+\(hw_only\)\s+=\s+([\d.]+)", out)
        return float(m.group(1)) if m else 0.0
    res = {h: {} for h in hefs}
    jobs = [(h, b) for h in hefs for b in batches]
    step = 1 if one_chip or len(DEVS) == 1 else 2  # PCIe-heavy nets (yolo26m) disturb each other on the shared uplink
    for k in range(0, len(jobs), step):
        pair = jobs[k:k + step]
        with ThreadPoolExecutor(2) as ex:
            fs = [ex.submit(one, h, DEVS[j], b) for j, (h, b) in enumerate(pair)]
            for (h, b), f in zip(pair, fs):
                res[h][b] = f.result()
    return res


def strict_retry(a, msg):
    """A context or the full model failed: re-partition with part_fix's next retry tier (1: R17/R18/R20, 2: also
    R19/R21/R22/R16b) and rerun into WORK_t<tier>; a tier that moves nothing escalates straight to the next."""
    lvl = int(os.environ.get("HCC_STRICT") or 0) + 1
    if lvl > 2:
        sys.exit(msg)
    print(msg, f"-> part_fix retry tier {lvl}", flush=True)
    sp, sw = a.part.rstrip("/") + f"_t{lvl}", a.work.rstrip("/") + f"_t{lvl}"
    env = dict(os.environ, PART_FIX_STRICT=str(lvl), HCC_STRICT=str(lvl))
    fx = subprocess.run([sys.executable, f"{R}/turbo/part_fix.py", a.har, a.part, sp], env=env, capture_output=True,
                        text=True)
    print("strict", fx.stdout.strip()[-300:], flush=True)
    if fx.returncode:
        sys.exit(msg)
    if not re.search(r"'(R1[6-9]b?|R2\d|R10)<'", fx.stdout):  # no TIER move (base rules such as R1 may fire when
        # the caller had disabled them, e.g. yolo26m's PART_FIX_NO_R1 partition): try the next tier, same partition
        os.environ["HCC_STRICT"] = str(lvl)
        return strict_retry(a, msg)
    for f in glob.glob(f"{a.part}/c*.extra.alls"):  # hand recipes part_fix would not regenerate
        if not os.path.exists(f"{sp}/{os.path.basename(f)}"):
            shutil.copy(f, sp)
    argv = [x if x not in (a.part, a.work) else (sp if x == a.part else sw) for x in sys.argv]
    os.execve(sys.executable, [sys.executable, "-u"] + argv, env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("har")
    ap.add_argument("part")
    ap.add_argument("work")
    ap.add_argument("--cap", type=float, default=300)
    ap.add_argument("--variants", default="default,lat,lat_fs,default_sr,lat_sr,lat_fs_sr")
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--combos", type=int, default=16)
    ap.add_argument("--batches", default="1,8")
    ap.add_argument("--ref", help="known-good HEF of the same HAR for the bit-exact check")
    ap.add_argument("--relieve", action="store_true", help="round 2: relieve.py nested defuse of bottleneck convs")
    ap.add_argument("--pin", default="", help="I=VARIANT,... fix context I's variant (sub-compile fps ties hide "
                    "full-model differences, e.g. yolo26m ctx1 lat_fs_sr_md is 7 ms/frame slower at b4)")
    ap.add_argument("--one-chip", action="store_true", help="benchmark one HEF at a time (PCIe-heavy nets)")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    os.makedirs(CACHE, exist_ok=True)
    t0 = time.time()
    n = len(glob.glob(f"{a.part}/c*.layers.json"))
    t_sub = sub_compiles(a, n)
    allv = a.variants.split(",")
    if a.relieve:  # round 2: nested-defuse the slowest convs of each context's best variant, keep it if better
        from relieve import relieve
        nd = []
        for i in range(n):
            best1 = min(((j["latency_ms"], v) for v in allv if (j := ok(f"{a.work}/c{i}_{v}"))), default=None)
            if best1 and "nd" not in best1[1]:
                hint = relieve(f"{a.work}/c{i}_{best1[1]}_compiled.har")
                if hint:
                    open(f"{a.work}/c{i}.nd.alls", "w").write(hint)
                    nd.append(best1[1] + "_nd")
        if nd:
            a2 = argparse.Namespace(**vars(a))
            a2.variants = ",".join(dict.fromkeys(nd))
            t_sub += sub_compiles(a2, n)
            allv += [v for v in dict.fromkeys(nd) if v not in allv]
    ranked, table = [], []
    for i in range(n):
        cands = []
        for v in allv:
            j = ok(f"{a.work}/c{i}_{v}")
            table.append((i, v, j and round(j["fps"]), j and j["latency_ms"], j and j.get("compile_s")))
            if j:
                cands.append((round(j["latency_ms"], 4), -round(j["fps"]), v))
        if not cands:
            strict_retry(a, f"context {i}: no variant compiled")
        ranked.append([v for *_, v in sorted(cands)])  # all finished variants, best first
    for p_ in filter(None, a.pin.split(",")):
        i_, v_ = p_.split("=")
        ranked[int(i_)] = [v_]
    for row in table:
        print("sub", *row)
    print(f"sub-compiles {t_sub:.0f} s; ranked {ranked}", flush=True)
    t1 = time.time()

    def build(combo):
        d = f"{a.work}/full_" + "_".join(combo)
        good, log = full_compile(a.har, [f"{a.work}/c{i}_{v}_compiled.har" for i, v in enumerate(combo)], d)
        return combo, d, good, log
    good, tried, k = [], set(), a.topk
    while not good:  # escalate: wider per-context choice until some combination is feasible in the full model
        combos = [c for c in itertools.product(*[r[:k] for r in ranked]) if c not in tried][:a.combos]
        if not combos:
            break
        tried |= set(combos)
        with ThreadPoolExecutor(len(combos)) as ex:
            built = list(ex.map(build, combos))
        good = [(c, d) for c, d, g, _ in built if g]
        for c, d, g, log in built:
            err = "" if g else (re.findall(r".*(?:Exception|failed).*", log) or ["?"])[-1][:150]
            print("full", "+".join(c), "OK" if g else "FAILED", err, flush=True)
        if k >= max(len(r) for r in ranked):
            break
        k += 1
    print(f"full compiles {time.time() - t1:.0f} s: {len(good)}/{len(tried)} OK", flush=True)
    if not good:
        strict_retry(a, "no combination compiled")
    batches = [int(b) for b in a.batches.split(",")]
    t2 = time.time()
    hefs = [f"{d}/full.hef" for _, d in good]
    with chips(DEVS[:1] if a.one_chip else DEVS):  # --one-chip benches on DEVS[0] only
        if len(hefs) > 4:  # screen everything at 2 s, measure only the top 4 at 5 s (16 combos of m: 514 -> ~250 s)
            f0 = bench(hefs, batches, a.one_chip, t=2)
            sc0 = {h: math.exp(sum(math.log(max(v, 1e-9)) for v in r.values()) / len(r)) for h, r in f0.items()}
            hefs = sorted(hefs, key=lambda h: -sc0[h])[:4]
        fps = bench(hefs, batches, a.one_chip)
    score = {h: math.exp(sum(math.log(max(v, 1e-9)) for v in r.values()) / len(r)) for h, r in fps.items()}

    def guard(h):  # every known Hailo-8 silent-wrong pattern (RULES 22-26) on a candidate HEF and its own script
        ch = os.path.dirname(h) + "/full_compiled.har"
        g = subprocess.run([sys.executable, f"{R}/turbo/hailo8_guard.py", h, ch if os.path.exists(ch) else a.har,
                            os.path.dirname(h) + "/full.alls", a.har], cwd=R, capture_output=True, text=True)
        return [ln for ln in g.stdout.splitlines() if ln.startswith(("S1", "WARN", "hailo8_guard"))], g.returncode == 0
    with ThreadPoolExecutor(len(fps)) as ex:
        gd = dict(zip(fps, ex.map(guard, fps)))
    for h, r in sorted(fps.items(), key=lambda x: -score[x[0]]):
        print("bench", h.split("/")[-2], " ".join(f"b{b}={v:.1f}" for b, v in r.items()), f"score={score[h]:.1f}",
              "guard", "ok" if gd[h][1] else "S1")
    clean = [h for h in score if gd[h][1]]  # never ship a known silent-wrong lowering when a clean one compiled
    best = max(clean or score, key=score.get)
    others = [a.ref] if a.ref else [h for h in fps if h != best][:3]
    exact = None
    if others:
        with chips([os.environ.get("DEVICE", "0000:02:00.0")]):  # exact_check runs on DEVICE
            out = subprocess.run([sys.executable, f"{R}/cprof/port/exact_check.py", others[0], best, *others[1:]],
                                 cwd=R, capture_output=True, text=True)
        out = out.stdout + out.stderr[-500:]
        exact = out.count("BIT-EXACT") >= 1 + len(others[1:]) and "DIFF" not in out.upper()
        print("exact check:", "BIT-EXACT" if exact else out[-400:])
    os.makedirs(f"{a.work}/best", exist_ok=True)
    subprocess.run(["cp", best, f"{a.work}/best/full.hef"])
    subprocess.run(["cp", os.path.dirname(best) + "/full.alls", f"{a.work}/best/full.alls"])
    total = time.time() - t0
    print(f"BEST {best} {fps[best]} bench {time.time() - t2:.0f} s, total {total:.0f} s")
    # every known Hailo-8 silent-wrong pattern (RULES 22-26) checked on the winner and its own script
    guard, guard_ok = gd[best]
    print("\n".join(guard))
    if not guard_ok:
        print("WARNING: no candidate passed hailo8_guard: the winner has a known silent-wrong pattern (fix above)")
    json.dump({"best": best, "fps": fps[best], "exact": exact, "sub_s": round(t_sub), "total_s": round(total),
               "ranked": ranked, "all_fps": fps, "table": table, "guard": guard, "guard_ok": guard_ok,
               "guard_all": {h: g_[1] for h, g_ in gd.items()}},
              open(f"{a.work}/hcc.json", "w"), indent=1)


if __name__ == "__main__":
    main()
