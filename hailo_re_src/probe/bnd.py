"""Boundary probes of OUR multi-context pipeline: pre-convs -> OP -> post-convs, compiled by stock in one context
(the reference HEF) and by turbo/hcc.py as two contexts cut right BEFORE the op (op = context entry) and right
AFTER it (op = last of context 0). Every failure or non-bit-exact result is a rule the pipeline must handle.

Usage (DFC env): python bnd.py CASES.txt OUT_DIR [PAR]     cases as sweep.py, e.g. maxpool:c=64,h=20,w=20,k=5
  -> OUT_DIR/<case>/{stock: sweep.py dir, cut_before/, cut_after/}, OUT_DIR/bnd.jsonl
"""
import json
import os
import re
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor

R = "/mnt/storage/lib_project/models/Hailo_model/hailo_re"
P = f"{R}/probe"


def hn_of(har):
    with tarfile.open(har) as t:
        nm = next(m for m in t.getnames() if m.endswith(".hn") and not m.endswith((".native.hn", ".fp.hn")))
        return json.loads(t.extractfile(nm).read())


def one(case, out, pre=2, post=2):
    tag = os.path.basename(case) if case.startswith("@") else re.sub(r"[^\w=,.-]", "_", case)
    d = f"{out}/{tag}"
    os.makedirs(d, exist_ok=True)
    if case.startswith("@"):  # prebuilt pseudo-model (op_ctx_gen.py): already has `pre` convs, OP, `post` convs
        full = case
    else:
        full = f"{case},pre={pre},post={post}" if ":" in case and not case.endswith(":") else f"{case}pre={pre},post={post}"
    rec = {"case": case}
    sub = subprocess.run([sys.executable, "-u", f"{P}/sweep.py", "run", full, d], capture_output=True, text=True,
                         cwd=P)
    sd = next((f"{d}/{x}" for x in os.listdir(d) if os.path.isdir(f"{d}/{x}") and x.startswith(tag.split(":")[0])
               and os.path.exists(f"{d}/{x}/result.json")), None)
    if not sd:
        rec["stage"] = "stock-missing"
        rec["err"] = sub.stdout[-300:] + sub.stderr[-300:]
        return rec
    r = json.load(open(f"{sd}/result.json"))
    rec["stock"] = r["stage"]
    if r["stage"] != "ok":
        rec["err"] = r.get("error")
        return rec
    hn = hn_of(f"{sd}/q.har")
    order = [n for n, v in hn["layers"].items() if v["type"] not in ("input_layer", "output_layer")]
    convs = [i for i, n in enumerate(order) if hn["layers"][n]["type"] in ("conv", "dw")]  # 1-channel convs become dw
    cb, ca = convs[pre - 1] + 1, convs[-post]  # first op layer, first post conv
    rec["op_layers"] = [n.split("/")[-1] + ":" + hn["layers"][n]["type"] for n in order[cb:ca]]
    for name, cut in (("cut_before", cb), ("cut_after", ca)):
        pd, wd = f"{d}/{name}/part", f"{d}/{name}/work"
        os.makedirs(pd, exist_ok=True)
        json.dump(order[:cut], open(f"{pd}/c0.layers.json", "w"))
        json.dump(order[cut:], open(f"{pd}/c1.layers.json", "w"))
        subprocess.run([sys.executable, f"{R}/turbo/part_fix.py", f"{sd}/q.har", pd], capture_output=True, cwd=R)
        h = subprocess.run([sys.executable, "-u", f"{R}/turbo/hcc.py", f"{sd}/q.har", pd, wd, "--variants",
                            "default,lat_fs,default_sr", "--topk", "1", "--batches", "1", "--cap", "600", "--ref",
                            f"{sd}/m.hef"], capture_output=True, text=True, cwd=R)
        open(f"{d}/{name}/hcc.out", "w").write(h.stdout + h.stderr)
        try:
            j = json.load(open(f"{wd}/hcc.json"))
            rec[name] = {"ok": True, "exact": j["exact"], "fps": j["fps"], "stock_fps": r["bottleneck_fps"]}
        except (OSError, ValueError):
            err = re.findall(r"(?:Exception|Error|failed|no variant|no combination)[^\n]*", h.stdout + h.stderr)
            subs = [f"{wd}/{x}" for x in os.listdir(wd) if x.endswith(".json")] if os.path.isdir(wd) else []
            sub_err = []
            for x in subs:
                try:
                    jj = json.load(open(x))
                    if not jj.get("ok"):
                        sub_err.append(os.path.basename(x) + ": " + (jj.get("error") or "")[:200])
                except (OSError, ValueError):
                    pass
            rec[name] = {"ok": False, "err": (err[-1] if err else "?")[:300], "sub_err": sub_err[:4]}
    return rec


if __name__ == "__main__":
    cases = [c.strip() for c in open(sys.argv[1]) if c.strip() and not c.startswith("#")]
    out = os.path.abspath(sys.argv[2])
    os.makedirs(out, exist_ok=True)
    with ThreadPoolExecutor(int(sys.argv[3]) if len(sys.argv) > 3 else 8) as ex, open(f"{out}/bnd.jsonl", "a") as f:
        for rec in ex.map(lambda c: one(c, out), cases):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(json.dumps({k: rec.get(k) for k in ("case", "stock", "cut_before", "cut_after")})[:400], flush=True)
