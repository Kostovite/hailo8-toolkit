"""Tabulate a sweep: per case the compiler's decisions and costs. python analyze.py RUN_DIR [--rules] [--csv OUT]

Columns: fps, bottleneck node + cycles, total subclusters / apus / input aligners / clusters used, the defuse
decision, MACs of the probed op and MAC per (subcluster x cycle) of its compute nodes (the chip's compute rate).
"""
import csv
import glob
import json
import re
import sys
from collections import Counter

IO = ("INPUT", "OUTPUT", "EXTERNAL_INPUT", "EXTERNAL_OUTPUT")


def macs(fam, p):
    c, o, h, w, k, s = p["c"], p["o"], p["h"], p["w"], p["k"], p["s"]
    s = s or 1
    ho, wo = -(-h // s), -(-w // s)
    if fam == "conv":
        return ho * wo * k * k * (c * o + (p["n"] - 1) * o * o)
    if fam == "dw":
        return p["n"] * ho * wo * k * k * c
    if fam == "gconv":
        return ho * wo * k * k * c * o // p["g"]
    return 0


def row(r):
    p, fam = r["params"], r["family"]
    nodes = r.get("nodes", [])
    typ = lambda n: n["type"].replace("PROTO_NETWORK_", "")  # noqa: E731
    cyc = lambda n: int(n["res"].get("frame_total_cycles", 0) or 0)  # noqa: E731
    comp = [n for n in nodes if typ(n) not in IO]
    tot = lambda k: sum(int(n["res"].get(k, 0) or 0) for n in comp)  # noqa: E731
    bn = max(nodes, key=cyc) if nodes else None
    clusters = {m.group(1) for n in nodes for m in [re.match(r"(?:input_to|output_from)_cluster_(\d+)", n["name"])] if m}
    work = [n for n in comp if typ(n) in ("CONV", "DW", "DEPTHWISE", "DENSE", "CONV_AND_ADD", "GROUP_CONV")]
    sc_cyc = sum(int(n["res"].get("subclusters", 0)) * cyc(n) for n in work)
    m = macs(fam, p)
    dl = r.get("defuse_lines", [])
    return {"case": r["case"], "stage": r["stage"], "fps": round(r.get("bottleneck_fps") or 0, 1),
            "bn_node": bn["name"] if bn else "", "bn_type": typ(bn) if bn else "", "bn_cycles": cyc(bn) if bn else 0,
            "subclusters": tot("subclusters"), "apus": tot("apus"), "aligners": tot("input_aligners"),
            "out_blocks": tot("output_blocks"), "l2_weights": tot("l2_weights"), "clusters": len(clusters),
            "nodes": len(comp), "work_nodes": len(work),
            "defuse": ";".join(re.sub(r"^.* = defuse\((\w+), (\d+), defuse_type=(\w+)\)$", r"\1x\2:\3", x) for x in dl),
            "macs": m, "mac_per_sc_cycle": round(m / sc_cyc, 2) if sc_cyc else "",
            "compile_s": r.get("compile_s"), "error": (r.get("error") or "")[:120]}


if __name__ == "__main__":
    rs = [json.load(open(f)) for f in sorted(glob.glob(f"{sys.argv[1]}/*/result.json"))]
    rows = [row(r) for r in rs]
    cols = ["case", "fps", "bn_type", "bn_cycles", "subclusters", "apus", "aligners", "clusters", "work_nodes",
            "defuse", "mac_per_sc_cycle", "error"]
    for x in rows:
        print("  ".join(str(x[c]) for c in cols))
    if "--csv" in sys.argv:
        with open(sys.argv[sys.argv.index("--csv") + 1], "w") as f:
            wr = csv.DictWriter(f, rows[0].keys())
            wr.writeheader()
            wr.writerows(rows)
    if "--rules" in sys.argv:
        rc = Counter(t for r in rs for t in r.get("rules", []))
        print(f"\n{len(rc)} rule templates")
        for t, n in sorted(rc.items()):
            print(f"{n:4d}  {t}")
