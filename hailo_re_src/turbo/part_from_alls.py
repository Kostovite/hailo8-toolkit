"""A compiled auto.alls's context partition -> per-context HN layer lists (the input of compile_sub.py / run_asm.sh).

Post-defuse nodes map back to their HN layer through the defuse() lines; compiler-made helpers (auto_reshape_*,
*_reshape, portals) are dropped; an HN layer whose pieces span contexts goes to the context of its first piece.
format_conversion layers (removed in the full script) go with their consumer's context.

Usage (DFC env): python part_from_alls.py FULL.har AUTO.alls OUT_DIR   -> OUT_DIR/c{i}.layers.json
"""
import json
import os
import re
import sys
import tarfile


def hn_of(har):
    with tarfile.open(har) as t:
        nm = next(m for m in t.getnames() if m.endswith(".hn") and not m.endswith((".native.hn", ".fp.hn")))
        return json.loads(t.extractfile(nm).read())


def partition(har, text):
    hn = hn_of(har)
    net = hn["name"]
    L = {k.split("/")[-1]: v for k, v in hn["layers"].items()}
    src = {}
    for m in re.finditer(r"^(.+?) = defuse\((\w+),", text, re.M):
        for p in m.group(1).split(","):
            src[p.strip()] = m.group(2)

    def root(n):
        for _ in range(20):
            if n not in src:
                break
            n = src[n]
        return n
    ctx = {}
    for m in re.finditer(r"^\w+_context_(\d+) = context\(\[([^\]]*)\]\)", text, re.M):
        for n in m.group(2).split(","):
            r = root(n.strip())
            if r in L and L[r]["type"] not in ("input_layer", "output_layer"):
                ctx.setdefault(r, int(m.group(1)))
    for n, v in L.items():  # format conversions: with the (first) consumer
        if n not in ctx and v["type"] == "format_conversion":
            cons = [c.split("/")[-1] for c in v["output"]]
            for _ in range(10):
                hit = [ctx[c] for c in cons if c in ctx]
                if hit:
                    ctx[n] = min(hit)
                    break
                cons = [c2.split("/")[-1] for c in cons if c in L for c2 in L[c]["output"]]
    missing = [n for n, v in L.items() if v["type"] not in ("input_layer", "output_layer") and n not in ctx]
    k = max(ctx.values()) + 1
    parts = [[f"{net}/{n}" for n in L if ctx.get(n) == i] for i in range(k)]
    return parts, missing


if __name__ == "__main__":
    har, alls, out = sys.argv[1:4]
    text = open(alls).read() if not alls.endswith(".har") else next(
        tarfile.open(alls).extractfile(m).read().decode() for m in tarfile.open(alls).getnames() if m.endswith(".auto.alls"))
    parts, missing = partition(har, text)
    os.makedirs(out, exist_ok=True)
    for i, p in enumerate(parts):
        json.dump(p, open(f"{out}/c{i}.layers.json", "w"))
    print("contexts", len(parts), "sizes", [len(p) for p in parts], "unassigned", missing)
