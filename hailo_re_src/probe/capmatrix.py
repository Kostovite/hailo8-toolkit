"""rules_v*.json -> capability table: node type x feature -> supported bit (from '<feature> AND !(<bit>)' clauses),
limit clauses kept as their distinct rendered forms. python capmatrix.py db/rules_v1.json [--md OUT.md]"""
import json
import re
import sys

SHORT = {"16 bit to 8 bit transition": "16->8", "8 bit to 16 bit transition": "8->16", "16bit bias decompision": "b16dec",
         "16bit bias is": "b16", "8bit bias": "b8", "16x4 is not supported with": "16x4@A16", "16x4 is": "16x4",
         "4bit": "w4", "Buffer in l4": "bufL4", "Dilation": "dil", "No contexts": "noctx", "Padding scheme": "pad",
         "double scale": "dblscale@A16", "precision_mode A16_W16": "A16W16"}
J = json.load(open(sys.argv[1]))
M = J["matrix"]


def short(c):
    return next((v for k, v in SHORT.items() if c.startswith(k)), None)


def bit(form):
    m = re.search(r"!\((\d)\)\)*\]?$", form) or re.search(r"!\((\d)\)", form)
    return m.group(1) if m else None


cols = list(dict.fromkeys(SHORT.values()))
rows = []
for t in sorted(M):
    cells, extra = {}, {}
    for c, forms in M[t].items():
        s = short(c)
        if s:
            bits = sorted({bit(f) or "?" for f in forms})
            cells[s] = "/".join(bits)
        else:
            extra[c.replace(" <L>", "").replace("for layer", "")] = sorted(forms)[:3]
    n = sum(next(iter(M[t].values())).values())
    rows.append((t, n, cells, extra))
w = max(len(t) for t, *_ in rows)
print(f"{'node type':{w}s} {'n':>6s} " + " ".join(f"{c:>8s}" for c in cols))
for t, n, cells, extra in rows:
    print(f"{t:{w}s} {n:6d} " + " ".join(f"{cells.get(c, '-'):>8s}" for c in cols))
print("\nlimits / data-dependent clauses:")
for t, n, cells, extra in rows:
    for c, forms in extra.items():
        if any(re.search(r"BIGGER|MOD", f) for f in forms):
            print(f"  {t:{w}s} {c[:60]:60s} {forms}")
