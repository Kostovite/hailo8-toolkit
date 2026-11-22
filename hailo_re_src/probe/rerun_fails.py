"""Re-run the fuzz cuts that did not come out bit-exact, with the CURRENT part_fix rules (R13-R16 were added
mid-run). Input partitions are the original cut's part dir. Usage: python rerun_fails.py OUT.jsonl [PAR]"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fuzz import run_cut  # noqa: E402

P = os.path.dirname(os.path.abspath(__file__))
re_ok = {(r["case"], r["tag"]) for r in map(json.loads, open(f"{P}/runs/recheck_infra.jsonl")) if r.get("res") == "EXACT"}
todo = []
if os.environ.get("SRC"):  # re-run the still-failing cuts of an earlier rerun (env SRC=its jsonl, PFX=new prefix)
    import glob
    for r in map(json.loads, open(os.environ["SRC"])):
        if not r.get("exact"):
            d = glob.glob(f"{P}/runs/fz*/{r['case']}")[0]
            todo.append((d, r["case"], r["tag"], r["at"]))
for f in () if todo else ("fz1", "fz2"):
    for l in open(f"{P}/runs/{f}/fuzz.jsonl"):
        r = json.loads(l)
        if r.get("stock") != "ok":
            continue
        for c in r["cuts"]:
            if not (c.get("exact") or (r["case"], c["tag"]) in re_ok):
                todo.append((f"{P}/runs/{f}/{r['case']}", r["case"], c["tag"], c["at"]))
out = open(sys.argv[1], "a")


def go(t):
    d, case, tag, at = t
    pd = f"{d}/{tag}/part"
    n = len([x for x in os.listdir(pd) if x.endswith(".layers.json")])
    ctxs = [json.load(open(f"{pd}/c{i}.layers.json")) for i in range(n)]
    rec = {"case": case, "tag": tag, "at": at, **run_cut(f"{d}/{case}", d, os.environ.get("PFX", "R2_") + tag, ctxs)}
    out.write(json.dumps(rec) + "\n"); out.flush()
    print(case, tag, at, "EXACT" if rec.get("exact") else rec.get("err", "not exact"), flush=True)


with ThreadPoolExecutor(int(sys.argv[2]) if len(sys.argv) > 2 else 4) as ex:
    list(ex.map(go, todo))
