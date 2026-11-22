"""Multi-context attention probe (RULES.md 15: yolo26m full assembly crashes with map::at).

Per pseudo-model (attn_ctx_gen.py: pre convs -> PSA -> post convs): stock single compile (reference + q.har), then
our pipeline on partitions that put the PSA in its own context:
  p3   [pre] [PSA] [post]          p2a  [pre+PSA] [post]          p2b  [pre] [PSA+post]
hcc runs the given variants; every full-compile outcome (OK / map::at / other) is recorded per combination.

Usage (DFC env): python mapat.py OUT_DIR PAR VARIANTS DIR [DIR ...]  -> OUT_DIR/mapat.jsonl
"""
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from bnd import hn_of

R = "/mnt/storage/lib_project/models/Hailo_model/hailo_re"
P = f"{R}/probe"


def parts(hn, e, p_=2, np_=1):
    L = hn["layers"]
    order = [n for n, v in L.items() if v["type"] not in ("input_layer", "output_layer")]
    convs = [i for i, n in enumerate(order) if L[n]["type"] == "conv"]
    a = convs[e - 1] + 1 if e else 0                       # PSA starts after the e pre convs
    mm = [i for i, n in enumerate(order) if L[n]["type"] == "matmul"]
    b = convs[-p_] if p_ else len(order)  # residual adds fold into convs: the PSA ends at the 1st of p post convs
    if np_ == 2:  # two PSA blocks with 2 convs between: [pre] [PSA1] [mid] [PSA2] [post] -> p3 = PSA1 | PSA2
        mid = [i for i in convs if mm[1] < i < mm[2]]
        c = mid[-2] if len(mid) >= 2 else mid[-1]  # cut before the 2 convs leading into PSA2's qkv
        return order, {"p3": [order[:a], order[a:c], order[c:b], order[b:]], "p2a": [order[:c], order[c:]]}
    return order, {"p3": [order[:a], order[a:b], order[b:]], "p2a": [order[:b], order[b:]],
                   "p2b": [order[:a], order[a:]]}


def one(d, out, variants):
    name = os.path.basename(d)
    e = int(re.search(r"_e(\d+)", name).group(1))
    od = f"{out}/{name}"
    subprocess.run([sys.executable, "-u", f"{P}/sweep.py", "run", "@" + os.path.abspath(d), od], cwd=P,
                   capture_output=True)
    sd = f"{od}/{name}"
    r = json.load(open(f"{sd}/result.json"))
    rec = {"model": name, "stock": r["stage"], "stock_fps": r.get("bottleneck_fps"), "err": r.get("error")}
    if r["stage"] != "ok":
        return rec
    p_ = int(re.search(r"_p(\d+)", name).group(1))
    np_ = int(re.search(r"_n(\d+)", name).group(1)) if "_n" in name else 1
    order, P_ = parts(hn_of(f"{sd}/q.har"), e, p_, np_)

    def run_part(item):
        pn, ctxs = item
        if any(not c for c in ctxs):
            return
        pd, wd = f"{od}/{pn}/part", f"{od}/{pn}/work"
        os.makedirs(pd, exist_ok=True)
        for i, c in enumerate(ctxs):
            json.dump(c, open(f"{pd}/c{i}.layers.json", "w"))
        fx = subprocess.run([sys.executable, f"{R}/turbo/part_fix.py", f"{sd}/q.har", pd], capture_output=True,
                            text=True, cwd=R)
        h = subprocess.run([sys.executable, "-u", f"{R}/turbo/hcc.py", f"{sd}/q.har", pd, wd, "--variants", variants,
                            "--topk", "9", "--combos", "32", "--batches", "1", "--cap", "900", "--ref", f"{sd}/m.hef"],
                           capture_output=True, text=True, cwd=R)
        txt = h.stdout + h.stderr
        open(f"{od}/{pn}/hcc.out", "w").write(txt)
        full = re.findall(r"^full (\S+) (OK|FAILED)(.*)$", txt, re.M)
        rec[pn] = {"moves": fx.stdout.strip()[:200],
                   "subs": re.findall(r"^sub (\d+) (\S+) (\S+)", txt, re.M),
                   "full": [(c, s, ("map::at" if "map::at" in m else m.strip()[:90])) for c, s, m in full],
                   "best": re.findall(r"^BEST .*", txt, re.M)[-1:]}
    with ThreadPoolExecutor(len(P_)) as ex:  # partitions of one model in parallel too
        list(ex.map(run_part, P_.items()))
    return rec


if __name__ == "__main__":
    out, par, variants = os.path.abspath(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    os.makedirs(out, exist_ok=True)
    with ThreadPoolExecutor(par) as ex, open(f"{out}/mapat.jsonl", "a") as f:
        for rec in ex.map(lambda d: one(d, out, variants), sys.argv[4:]):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(rec["model"], rec.get("stock"), {k: [x[1] + ("" if x[1] == "OK" else ":" + x[2][:40])
                                                      for x in v["full"]] for k, v in rec.items()
                                                  if isinstance(v, dict)}, flush=True)
