"""The compiler's hardware rule formulas (HEF se_formula) -> a capability matrix per HW node type.

Every per-node formula is an AND of named clauses "<feature used> AND !(<supported>)" whose constants are the node
type's capability bits / limits. rules.py renders each clause with the node name removed, then aggregates over many
HEFs: node type x clause -> the distinct rendered forms seen (one form = a fixed rule; several = data-dependent).

Usage (DFC env): python rules.py OUT.json HEF [HEF ...]     or   python rules.py OUT.json --glob 'PATTERN' ...
"""
import glob
import json
import re
import sys
from collections import defaultdict
from multiprocessing import Pool

sys.path.insert(0, __file__.rsplit("/", 1)[0])


def node_clauses(path):
    from ighw import load
    g = load(path)
    types = {}
    for cn in g.contexts_graph.nodes:
        for n in cn.network_graph.nodes:
            types[n.name] = n.DESCRIPTOR.fields_by_name["type"].enum_type.values_by_number[n.type].name.replace(
                "PROTO_NETWORK_", "")
    if not g.se_formula:
        return []
    F = g.se_formula[0].DESCRIPTOR
    tn = lambda f: F.fields_by_name["type"].enum_type.values_by_number[f.type].name.replace("PROTO_SE_F_", "")  # noqa
    vn = lambda f: F.fields_by_name["var"].enum_type.values_by_number[f.var].name.replace("PROTO_SE_", "")  # noqa

    def render(f):
        t, a = tn(f), [render(s) for s in f.formulas]
        if t == "VAR":
            return vn(f)
        if t == "CONST":
            return str(f.value)
        if t in ("TRUE", "FALSE"):
            return t
        if t == "NOT":
            return f"!({a[0]})"
        if t == "FORMULA":
            return a[0] if len(a) == 1 else "F(" + ",".join(a) + ")"
        return "(" + f" {t} ".join(a) + ")"

    def flat(x):
        if tn(x) == "AND" and not x.explanation:
            for s in x.formulas:
                yield from flat(s)
        else:
            yield x
    out = []
    for f in g.se_formula:
        L = f.layer_name
        for c in flat(f):
            name = c.explanation.replace(L, "<L>") if c.explanation else "?"
            m = re.match(r"<L>: the input buffers should be divisible by (\d+) and it is currently: (\d+)", name)
            if m:  # data: the divisor is the rule, "currently" is this instance
                name = f"input buffers divisible by {m.group(1)}"
            out.append((types.get(L, "?"), name, render(c)))
    return out


def safe(p):
    try:
        return p, node_clauses(p)
    except Exception as e:  # noqa: BLE001
        return p, f"{type(e).__name__}: {e}"


if __name__ == "__main__":
    out = sys.argv[1]
    hefs = []
    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == "--glob":
            hefs += glob.glob(args[i + 1], recursive=True)
            i += 2
        else:
            hefs.append(args[i])
            i += 1
    agg = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # type -> clause -> form -> count
    bad = {}
    with Pool(32) as pool:
        for p, cl in pool.imap_unordered(safe, sorted(set(hefs)), chunksize=4):
            if isinstance(cl, str):
                bad[p] = cl
                continue
            for t, name, form in cl:
                agg[t][name][form] += 1
    json.dump({"hefs": len(hefs), "errors": bad, "matrix": agg}, open(out, "w"), indent=1)
    print(f"hefs {len(hefs)} errors {len(bad)} node types {len(agg)}")
