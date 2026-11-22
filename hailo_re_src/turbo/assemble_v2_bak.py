"""Assemble a multi-context model script from per-context single-context compiles (subhar.py sub-networks).

Each sub-compile's best.auto.alls holds the allocator's solution for that context: per-layer compilation_param (defuse
decisions), the context's post-defuse node list, cluster placement, buffers and portals. Context i of the full model
gets sub i's lines with the sub-network's artificial cut IO (input_layer5xx / output_layer5xx and their
auto_reshape nodes) removed and place() scoped to best_context_i; buffers that touch a cut are dropped (the compiler
derives the inter-context transfers itself, they never appear in an auto.alls). Global lines come from sub 0.

Usage: python assemble.py OUT.alls SUB0_compiled.har SUB1_compiled.har ...
"""
import re
import sys
import tarfile

CUT = re.compile(r"(input|output)_layer5\d\d(?!\d)")  # names continue with "_to_..." (no \b there)


def auto_alls(har):
    with tarfile.open(har) as t:
        name = next(m for m in t.getnames() if m.endswith(".auto.alls"))
        return t.extractfile(name).read().decode()


def hn_layers(har):
    """The sub-network's own HN layers (pre-defuse names: compilation_param targets), cut IO excluded."""
    import json
    with tarfile.open(har) as t:
        name = next(m for m in t.getnames() if m.endswith(".hn") and not m.endswith((".native.hn", ".fp.hn")))
        hn = json.loads(t.extractfile(name).read())
    return {n.split("/")[-1] for n, v in hn["layers"].items() if not CUT.search(n.split("/")[-1])}


def cut_sources(har):
    """input_layer5xx of a sub-network -> the full model's producer it stands for (subhar.py original_names)."""
    import json
    with tarfile.open(har) as t:
        name = next(m for m in t.getnames() if m.endswith(".hn") and not m.endswith((".native.hn", ".fp.hn")))
        hn = json.loads(t.extractfile(name).read())
    out = {}
    for n, v in hn["layers"].items():
        o = (v.get("original_names") or [""])[0]
        m = re.match(r"cut_(.+)_\d+$", o)
        if v["type"] == "input_layer" and m:
            out[n.split("/")[-1]] = m.group(1)
    return out


def split_args(s):
    return [a.strip() for a in s.split(",") if a.strip()]


ORDER = ("remove_node", "decl", "portal", "compilation_param", "buffers", "context", "place", "other")


def bucket(stmt):
    """Statement kind, in the order the compiler's own auto.alls uses (nodes must be declared before use)."""
    if stmt.startswith("remove_node("):
        return "remove_node"
    if re.search(r" = (ddr|l4_portal)\(", stmt):
        return "portal"
    if re.search(r" = context\(", stmt):
        return "context"
    if ".place(" in stmt:
        return "place"
    if stmt.startswith("compilation_param("):
        return "compilation_param"
    if stmt.startswith("buffers("):
        return "buffers"
    if re.search(r" = \w+\(", stmt):  # defuse / shortcut / concat / shape_splitter / format_conversion
        return "decl"
    return "other"


def dropped(har):
    """format_conversion layers subhar.py cut away (sub-network input after them): compile_sub.py's PREFIX.json."""
    import json
    import os
    js = har[:-len("_compiled.har")] + ".json"
    return json.load(open(js)).get("dropped", []) if os.path.exists(js) else []


def assemble(subs, net=None):
    head, body = [], []
    if net is None:  # the network name the compiler scopes everything with (the HN name: "best", "probe", ...)
        net = re.search(r"^(\w+)_context_0 = context\(", auto_alls(subs[0]), re.M).group(1)
    for i, har in enumerate(subs):
        for n in dropped(har):  # the full compile collapses them, as its own script does
            body.append(f"remove_node({n})")
        text = auto_alls(har)
        layers = hn_layers(har)
        nodes = set()
        m = re.search(r"^\w+_context_0 = context\(\[([^\]]*)\]\)", text, re.M)
        assert m, f"{har}: no context list"
        for n in split_args(m.group(1)):
            if not CUT.search(n):
                nodes.add(n)
        # A sub-compile with reshapes off splits format conversions into explicit SPATIAL_RESHAPE nodes (whole-tensor
        # buffering: the assembled pipeline stalls, "Latency simulation reached timeout"). The full compile removes
        # them (collapse into neighbours), so do as its own script does: remove_node, and forget the pieces.
        pieces = set()
        for fm in re.finditer(r"^(.+?) = defuse\((format_conversion\d+), \d+, defuse_type=SPATIAL_RESHAPE\)$", text, re.M):
            pieces |= set(split_args(fm.group(1)))
            body.append(f"remove_node({fm.group(2)})")
        nodes -= pieces
        mentions_piece = (lambda st: bool(pieces) and any(re.search(rf"(?<![\w-]){re.escape(p)}(?![\w-])", st)
                                                           for p in pieces))
        body.append(f"{net}_context_{i} = context([{', '.join(sorted(nodes))}])")
        real = cut_sources(har)
        for line in text.splitlines():
            s = line.strip()
            fc = re.match(r"^(\w+) = format_conversion\((input_layer5\d\d), ", s)
            if fc and fc.group(2) in real:  # e.g. a transpose_matmul conversion of an operand fed across the cut:
                s = s.replace(fc.group(2), real[fc.group(2)], 1)  # the full model has it on the real producer
                nodes.add(fc.group(1))
                body.append(s)
                continue
            if not s or s.startswith(f"{net}_context_0 = context("):
                continue
            if pieces and not s.startswith("place(") and (" = defuse(format_conversion" in s or mentions_piece(s)):
                continue
            if s.startswith(("allocator_param(", "resources_param(", "platform_param(", "performance_param(",
                             "quantization_param(", f"{net} = network_group(")) or re.match(r"^\w+ = normalization\(", s):
                if i == 0:
                    head.append(s)
                continue
            if CUT.search(s):
                continue
            pm = re.match(r"^place\((\d+), \[([^\]]*)\]\)$", s)
            if pm:
                keep = [n for n in split_args(pm.group(2)) if n in nodes]
                if keep:
                    body.append(f"{net}_context_{i}.place({pm.group(1)}, [{', '.join(keep)}])")
                continue
            cm = re.match(r"^compilation_param\(([^,]+),", s)
            if cm:  # pre-defuse layer names: the defuse decision itself (creates the _sd0/_d1 nodes listed)
                if cm.group(1).strip() in nodes or cm.group(1).strip() in layers:
                    body.append(s)
                continue
            bm = re.match(r"^buffers\(([^,]+), ([^,]+),", s)
            if bm:
                if bm.group(1).strip() in nodes and bm.group(2).strip() in nodes:
                    body.append(s)
                continue
            dm = re.match(r"^(\w+) = (ddr|l4_portal|format_conversion)\(([^,]+), ([^,)]+)", s)
            if dm:
                if dm.group(3).strip() in nodes or dm.group(4).strip() in nodes:
                    body.append(s)
                continue
            body.append(s)  # remove_node(...) and anything else: keep
    body = list(dict.fromkeys(body))  # the same remove_node can come from two subs
    body.sort(key=lambda st: ORDER.index(bucket(st)))  # stable: keeps each sub's own order inside a kind
    return "\n".join(head + [""] + body) + "\n"


if __name__ == "__main__":
    open(sys.argv[1], "w").write(assemble(sys.argv[2:]))
    print("wrote", sys.argv[1])
