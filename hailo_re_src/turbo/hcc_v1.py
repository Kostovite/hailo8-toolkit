"""hcc: parallel multi-context compile from a partition (per-context HN layer lists).

  1. every context x every objective variant = one compile_sub.py process, all at once, each capped at --cap s
  2. per context keep the finished variant with the lowest single-context latency (batch-1 time of a context is its
     pipeline fill), ties -> higher fps
  3. assemble.py -> drop buffers() (sub-compile sizes do not hold once format conversions collapse) -> R6: weights
     out of L2 (mixed_mem=disabled; single-context compiles keep them in L2, multi-context reloads them through the
     sequencer at every switch) -> full compile (compile_alls.py)

Usage (DFC env): python hcc.py FULL.har PART_DIR WORK_DIR [--cap 600] [--variants default,fps2x,lat]
  PART_DIR/c{i}.layers.json (part_from_alls.py), optional PART_DIR/c{i}.extra.alls appended to every variant of ctx i
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

R = "/mnt/storage/lib_project/models/Hailo_model/hailo_re"
BASE = "allocator_param(automatic_reshapes=disabled, enable_auto_spatial_reshapes=False)\n"
VARIANTS = {
    "default": "",
    "lat": "performance_param(optimize_for_batch=1)\n",
    "fps2k": "performance_param(fps=2000)\n",
    "fps4k": "performance_param(fps=4000)\n",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("har")
    ap.add_argument("part")
    ap.add_argument("work")
    ap.add_argument("--cap", type=float, default=600)
    ap.add_argument("--variants", default="default,lat,fps2k,lat_fs,fps2k_fs")
    ap.add_argument("--reuse", action="store_true", help="keep sub-compiles already in WORK_DIR")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    t0 = time.time()
    n = len(glob.glob(f"{a.part}/c*.layers.json"))
    procs = {}
    for i in range(n):
        extra = open(f"{a.part}/c{i}.extra.alls").read() if os.path.exists(f"{a.part}/c{i}.extra.alls") else ""
        for v in a.variants.split(","):
            pre = f"{a.work}/c{i}_{v}"
            if a.reuse and os.path.exists(pre + ".log"):  # ran before (finished, failed or killed at the cap)
                continue
            fs = f"{a.part}/c{i}.fs.alls"  # R5 hint from part_fix.py: feature-defuse the entry dw layers
            if v.endswith("_fs") and not os.path.exists(fs):
                continue
            body = BASE + extra + (open(fs).read() if v.endswith("_fs") else "") + VARIANTS[v.removesuffix("_fs")]
            open(pre + ".alls", "w").write(body)
            log = open(pre + ".log", "w")
            procs[(i, v)] = subprocess.Popen([sys.executable, "-u", f"{R}/turbo/compile_sub.py", a.har,
                                              f"{a.part}/c{i}.layers.json", pre, pre + ".alls"],
                                             stdout=log, stderr=subprocess.STDOUT, cwd=R)
    while procs and any(p.poll() is None for p in procs.values()):
        if time.time() - t0 > a.cap:
            done_ctx = {i for (i, v), p in procs.items() if p.poll() == 0 and ok(f"{a.work}/c{i}_{v}")}
            if done_ctx >= set(range(n)):  # every context has a result: stop the stragglers
                for p in procs.values():
                    if p.poll() is None:
                        p.kill()
                break
        time.sleep(2)
    t_sub = time.time() - t0
    ranked, table = [], []
    for i in range(n):
        cands = []
        for v in a.variants.split(","):
            j = ok(f"{a.work}/c{i}_{v}")
            table.append((i, v, j and round(j["fps"]), j and j["latency_ms"], j and j.get("compile_s")))
            if j:
                cands.append((j["latency_ms"], -j["fps"], v))
        if not cands:
            sys.exit(f"context {i}: no variant compiled")
        ranked.append([v for *_, v in sorted(cands)])
    for row in table:
        print("sub", *row)
    pick = [0] * n  # index into ranked[i]
    t1 = time.time()
    good, tries = False, 0
    while tries < 12:
        names = [ranked[i][k] for i, k in enumerate(pick)]
        print("assemble", names, flush=True)
        good, log, text = full_compile(a, [f"{a.work}/c{i}_{v}_compiled.har" for i, v in enumerate(names)], tries)
        tries += 1
        if good:
            break
        bad = set(re.findall(r"Validator failed on node: ([\w-]+)", log))
        ctx = {i for b in bad for i in node_contexts(text, b)}
        nxt = [i for i in sorted(ctx) if pick[i] + 1 < len(ranked[i])]
        if not nxt:
            break
        for i in nxt:  # that context's choice is infeasible inside the full model: its next-best variant
            pick[i] += 1
    names = [ranked[i][k] for i, k in enumerate(pick)]
    print(f"full compile {'OK' if good else 'FAILED'} ({tries} assemblies) {time.time() - t1:.0f} s, "
          f"total {time.time() - t0:.0f} s, picked {names}")
    if not good:
        print("\n".join(re.findall(r".*(?:Exception|Error|failed).*", log)[-5:]))
    json.dump({"pick": names, "sub_s": round(t_sub), "total_s": round(time.time() - t0), "ok": good,
               "table": table}, open(f"{a.work}/hcc.json", "w"), indent=1)


def node_contexts(text, node):
    return [int(m.group(1)) for m in re.finditer(r"^\w+_context_(\d+) = context\(\[([^\]]*)\]\)", text, re.M)
            if node in [x.strip() for x in m.group(2).split(",")]]


def full_compile(a, subs, tag):
    """assemble -> drop buffers -> R6 (weights out of L2) -> compile; script-level errors repaired in place."""
    out = subprocess.run([sys.executable, f"{R}/turbo/assemble.py", f"{a.work}/full_b.alls", *subs], cwd=R,
                         capture_output=True, text=True)
    if out.returncode:
        return False, out.stderr[-3000:], ""
    orig = open(f"{a.work}/full_b.alls").read()
    text = "\n".join(ln for ln in orig.splitlines() if not ln.startswith("buffers(")) + "\n"
    text = text.replace("mixed_mem=enabled", "mixed_mem=disabled")
    for attempt in range(6):
        open(f"{a.work}/full.alls", "w").write(text)
        if os.path.exists(f"{a.work}/full.hef"):
            os.remove(f"{a.work}/full.hef")
        out = subprocess.run([sys.executable, "-u", f"{R}/turbo/t/compile_alls.py", a.har, f"{a.work}/full.alls",
                              f"{a.work}/full.hef"], cwd=R, capture_output=True, text=True)
        log = out.stdout + out.stderr
        open(f"{a.work}/full.{tag}.{attempt}.log", "w").write(log)
        if os.path.exists(f"{a.work}/full.hef") and "COMPILE_S" in log:
            return True, log, text
        text2 = repair(text, log)
        if text2 == text:
            return False, log, text
        text = text2
    return False, log, text


def repair(text, log):
    """Script-level errors of an assembled script: a helper node only a sub-compile made ('could not be found in
    scope', 'Cannot find layer') -> drop its own statements and remove it from every node list."""
    gone = set(re.findall(r"layers (?:\w+/)?([\w-]+) could not be found in scope", log))
    gone |= {x.split("/")[-1] for x in re.findall(r"Cannot find layer \['([^']+)'\]", log)}
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


def ok(pre):
    try:
        j = json.load(open(pre + ".json"))
    except (OSError, ValueError):
        return None
    return j if j.get("ok") and j.get("fps") else None


if __name__ == "__main__":
    main()
