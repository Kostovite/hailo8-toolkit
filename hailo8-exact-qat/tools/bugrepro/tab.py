"""Tabulate shape sweep (chip vs emu per case per shape) and the GroupNorm1 CxH grid."""
import glob, json, os, re, collections
S = os.path.dirname(os.path.abspath(__file__))
def load(pat):
    out = {}
    for f in glob.glob(pat):
        for l in open(f):
            try: r = json.loads(l)
            except Exception: continue
            out[os.path.basename(r["dir"])] = r
    return out
tags = sorted({re.sub(r"\.res[01]\.jsonl$", "", os.path.basename(f)) for f in glob.glob(f"{S}/C*_H*_W*.res*.jsonl")})
cases = collections.defaultdict(dict)
print("## Shape sweep: chip vs SDK_BIT_EXACT (frac mismatching / max LSB); only cases with any mismatch\n")
for t in tags:
    st = open(f"{S}/{t}.stages.txt").read().split("\n") if os.path.exists(f"{S}/{t}.stages.txt") else []
    res = load(f"{S}/{t}.res*.jsonl")
    nbad = 0
    for c, r in res.items():
        fr = max((o["frac"] for o in r.get("outs", [])), default=None); mx = max((o["max"] for o in r.get("outs", [])), default=None)
        if r.get("error"): cases[c][t] = "ERR"
        elif fr: cases[c][t] = f"{fr:.3f}/{mx}"; nbad += 1
    print(f"{t}: {len(res)} chip-run, {nbad} with mismatch; stages: " + ", ".join(s.strip() for s in st if s.strip()))
print("\n| case | " + " | ".join(tags) + " |\n|---|" + "---|" * len(tags))
for c in sorted(cases):
    print(f"| {c} | " + " | ".join(cases[c].get(t, "0") for t in tags) + " |")
g = load(f"{S}/gn/res.jsonl")
if g:
    Cs = sorted({int(re.search(r"C(\d+)", k).group(1)) for k in g}); Hs = sorted({int(re.search(r"H(\d+)", k).group(1)) for k in g})
    print("\n## GroupNorm(1) grid: frac mismatching (max LSB), rows C, cols H=W\n\n| C \\ H | " + " | ".join(map(str, Hs)) + " |\n|---|" + "---|" * len(Hs))
    for c in Cs:
        row = []
        for h in Hs:
            r = g.get(f"gn1_C{c}_H{h}")
            if r is None: row.append("-"); continue
            o = r.get("outs") or [{}]; row.append(f"{o[0].get('frac', 'ERR')}({o[0].get('max', '')})")
        print(f"| {c} | " + " | ".join(row) + " |")
