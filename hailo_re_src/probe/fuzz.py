"""Crash hunt over fuzz_gen.py models: stock single compile (reference), then the own pipeline (part_fix + hcc) at
random cuts: K single cuts and one 3-context cut, seeded per model. Every failure / non-bit-exact result is logged
with the cut's neighbour layer types, so failures cluster into rules.

Usage (DFC env): python fuzz.py CASES.txt OUT_DIR [PAR] [K]   -> OUT_DIR/fuzz.jsonl
"""
import json
import os
import random
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from bnd import hn_of

R = "/mnt/storage/lib_project/models/Hailo_model/hailo_re"
P = f"{R}/probe"


def run_cut(sd, d, name, ctxs):
    pd, wd = f"{d}/{name}/part", f"{d}/{name}/work"
    os.makedirs(pd, exist_ok=True)
    for i, c in enumerate(ctxs):
        json.dump(c, open(f"{pd}/c{i}.layers.json", "w"))
    fx = subprocess.run([sys.executable, f"{R}/turbo/part_fix.py", f"{sd}/q.har", pd], capture_output=True,
                        text=True, cwd=R)
    h = subprocess.run([sys.executable, "-u", f"{R}/turbo/hcc.py", f"{sd}/q.har", pd, wd, "--variants",
                        "default,lat_fs,default_sr", "--topk", "2", "--batches", "1", "--cap", "900", "--ref",
                        f"{sd}/m.hef"], capture_output=True, text=True, cwd=R)
    txt = h.stdout + h.stderr
    open(f"{d}/{name}/hcc.out", "w").write(txt)
    rec = {"moves": fx.stdout.strip()[-200:], "fix_err": fx.stderr.strip()[-200:] if fx.returncode else ""}
    try:
        j = json.load(open(f"{wd}/hcc.json"))
        rec.update(ok=True, exact=j["exact"], fps=j["fps"])
    except (OSError, ValueError):
        errs = re.findall(r"^full \S+ FAILED (.*)$", txt, re.M)
        subs = []
        for x in sorted(os.listdir(wd)) if os.path.isdir(wd) else []:
            if x.endswith(".json"):
                try:
                    jj = json.load(open(f"{wd}/{x}"))
                    if not jj.get("ok"):
                        subs.append(x + ": " + (jj.get("error") or "")[:160])
                except (OSError, ValueError):
                    pass
        tail = re.findall(r"(?:Error|Exception|failed|no variant|no combination)[^\n]*", txt)
        rec.update(ok=False, full_err=sorted(set(e[:160] for e in errs))[:3], sub_err=subs[:3],
                   err=(tail[-1] if tail else "?")[:200])
    return rec


def one(case, out, k):
    name = os.path.basename(case)
    d = f"{out}/{name}"
    os.makedirs(d, exist_ok=True)
    subprocess.run([sys.executable, "-u", f"{P}/sweep.py", "run", case, d], capture_output=True, cwd=P)
    sd = f"{d}/{name}"
    rec = {"case": name, "spec": open(case[1:] + "/spec.txt").read().strip()}
    try:
        r = json.load(open(f"{sd}/result.json"))
    except (OSError, ValueError):
        rec["stock"] = "missing"
        return rec
    rec["stock"] = r["stage"]
    if r["stage"] != "ok":
        rec["stock_err"] = (r.get("error") or "")[:200]
        return rec
    hn = hn_of(f"{sd}/q.har")
    L = hn["layers"]
    order = [n for n, v in L.items() if v["type"] not in ("input_layer", "output_layer")]
    rng = random.Random(name)
    pos = list(range(1, len(order)))
    cuts = [[c] for c in rng.sample(pos, min(k, len(pos)))]
    if len(pos) >= 4:
        cuts.append(sorted(rng.sample(pos, 2)))
    jobs = []
    for cs in cuts:
        b = [0] + cs + [len(order)]
        ctxs = [order[b[i]:b[i + 1]] for i in range(len(b) - 1)]
        tag = "cut_" + "_".join(map(str, cs))
        nb = [f"{L[order[c - 1]]['type']}|{L[order[c]]['type']}" for c in cs]
        jobs.append((tag, ctxs, nb))
    with ThreadPoolExecutor(len(jobs)) as ex:
        res = list(ex.map(lambda j: run_cut(sd, d, j[0], j[1]), jobs))
    rec["cuts"] = [dict(tag=t, at=nb, **x) for (t, _, nb), x in zip(jobs, res)]
    return rec


if __name__ == "__main__":
    cases = [c.strip() for c in open(sys.argv[1]) if c.strip()]
    out = os.path.abspath(sys.argv[2])
    par = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    k = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    os.makedirs(out, exist_ok=True)
    with ThreadPoolExecutor(par) as ex, open(f"{out}/fuzz.jsonl", "a") as f:
        for rec in ex.map(lambda c: one(c, out, k), cases):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            bad = [c for c in rec.get("cuts", []) if not (c.get("ok") and c.get("exact"))]
            print(rec["case"], rec["spec"], "stock", rec["stock"], f"{len(rec.get('cuts', [])) - len(bad)}/"
                  f"{len(rec.get('cuts', []))} exact", [(c["tag"], c["at"], c.get("err", "NOTEXACT")[:80])
                                                        for c in bad], flush=True)
