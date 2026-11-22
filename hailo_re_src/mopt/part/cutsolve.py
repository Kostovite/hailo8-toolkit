"""Minimum-PCIe context partition for a Hailo-8 multi-context network.

Model: contexts are contiguous segments of a topological order of the HN layers. A tensor produced in context i and
consumed in a later context crosses PCIe twice per frame (DMA out once, DMA in once per consuming context). Each
context's resource load (subclusters, APUs, input aligners, layer controllers, L2 bias space) is the sum over its
layers, taken from the compiler's own per-node resource table of a reference compile, and must not exceed a cap
(by default the largest load the compiler itself put in one context, i.e. known feasible).

Exact solution by DP over the order: best[k][j] = min over i<j of best[k-1][i] + cut_bytes(i, j) subject to
load(i, j) <= cap, where a cut's cost is the bytes of tensors live across that boundary (counted per consumer
context).

Usage: python cutsolve.py HN HW_JSON REF_ALLS K [--slack 1.0] [--emit OUT.alls] [--emit_json OUT.json]
"""
import argparse
import collections
import json
import re

RES = ("subclusters", "apus", "input_aligners", "layer_controller_utilization", "l2_16bit_bias_space")


def hn_graph(hn):
    L = hn["layers"]
    names = list(L)
    short = {n: n.split("/")[-1] for n in names}
    out_bytes = {}
    for n, v in L.items():
        shp = v.get("output_shapes", [[0]])[0]
        el = 1
        for s in shp[1:]:
            el *= max(int(s), 1)
        pm = v.get("quantization_params", {}).get("precision_mode", "") or ""
        out_bytes[short[n]] = el * (2 if str(pm).endswith("a16") or "a16_w16" in str(pm) else 1)
    succ = {short[n]: [short[s] for s in v.get("output", []) if s in L] for n, v in L.items()}
    return [short[n] for n in names], out_bytes, succ, {short[n]: L[n]["type"] for n in names}


def layer_resources(hw, hn_names):
    """Sum each HW node's resources into the HN layer it came from (longest-prefix name match)."""
    by_len = sorted(hn_names, key=len, reverse=True)
    res = collections.defaultdict(collections.Counter)
    unmatched = collections.Counter()
    for c in hw["contexts_graph"]["nodes"]:
        ng = c["network_graph"]
        for node, r in zip(ng["nodes"], ng.get("resources_list", [])):
            nm = node["name"].split("/")[-1]
            hit = next((h for h in by_len if nm == h or nm.startswith(h + "_")), None)
            if hit is None:
                unmatched[node["type"]] += 1
                continue
            for k in RES:
                v = r.get(k, 0)
                res[hit][k] += float(v) if not isinstance(v, list) else 0.0
    return res, unmatched


def ref_contexts(alls):
    ctx = {}
    for m in re.finditer(r"(\w+)_context_(\d+) = context\(\[([^\]]*)\]\)", alls):
        for n in m.group(3).split(","):
            ctx[n.strip()] = int(m.group(2))
    return ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hn")
    ap.add_argument("hw")
    ap.add_argument("ref_alls")
    ap.add_argument("K", type=int)
    ap.add_argument("--slack", type=float, default=1.0, help="cap = slack * max per-context load of the reference")
    ap.add_argument("--emit", default=None)
    ap.add_argument("--emit_json", default=None, help="layer -> context map for mkalls.py")
    args = ap.parse_args()
    hn = json.load(open(args.hn))
    order, nbytes, succ, types = hn_graph(hn)
    hw = json.load(open(args.hw))
    res, unmatched = layer_resources(hw, order)
    print("HW nodes not matched to an HN layer:", dict(unmatched))
    refc = ref_contexts(open(args.ref_alls).read())
    pos = {n: i for i, n in enumerate(order)}
    # reference per-context load -> cap
    load = collections.defaultdict(collections.Counter)
    ref_of = {}
    for n in order:
        c = refc.get(n)
        if c is None:  # HN layer that the compiler split: take the context of its first derived node
            c = next((refc[m] for m in refc if m.startswith(n + "_")), None)
        ref_of[n] = c
        if c is not None:
            load[c].update(res[n])
    cap = {k: args.slack * max(load[c][k] for c in load) for k in RES}
    print("reference loads:", {c: {k: round(load[c][k], 1) for k in RES} for c in sorted(load)})
    print("cap:", {k: round(v, 1) for k, v in cap.items()})

    n = len(order)
    # cut_cost[j]: bytes crossing boundary between position j-1 and j for a segment structure is not separable in
    # general (a tensor skipping several contexts is read once per consumer context). Use the per-boundary live-set
    # cost: sum of bytes of tensors produced before j and consumed at >= j. This counts a tensor once per boundary it
    # spans, an upper bound that equals the true cost when every skip tensor has consumers in one later context.
    live = [0] * (n + 1)
    for p in order:
        last = max([pos[s] for s in succ[p]] or [pos[p]])
        for j in range(pos[p] + 1, last + 1):
            live[j] += nbytes[p]
    pref = {k: [0.0] * (n + 1) for k in RES}
    for i, p in enumerate(order):
        for k in RES:
            pref[k][i + 1] = pref[k][i] + res[p][k]

    def fits(i, j):
        return all(pref[k][j] - pref[k][i] <= cap[k] + 1e-9 for k in RES)

    INF = float("inf")
    best = [[INF] * (n + 1) for _ in range(args.K + 1)]
    arg = [[-1] * (n + 1) for _ in range(args.K + 1)]
    best[0][0] = 0.0
    for k in range(1, args.K + 1):
        for j in range(1, n + 1):
            for i in range(j - 1, -1, -1):
                if not fits(i, j):
                    break
                if best[k - 1][i] == INF:
                    continue
                c = best[k - 1][i] + (live[i] if i > 0 else 0)
                if c < best[k][j]:
                    best[k][j], arg[k][j] = c, i
    if best[args.K][n] == INF:
        print(f"K={args.K}: infeasible under cap")
        return
    cuts, j = [], n
    for k in range(args.K, 0, -1):
        i = arg[k][j]
        cuts.append((i, j))
        j = i
    cuts.reverse()
    ref_live = sum(live[i] for i in range(1, n) if ref_of.get(order[i]) != ref_of.get(order[i - 1])
                   and ref_of.get(order[i]) is not None and ref_of.get(order[i - 1]) is not None)
    print(f"K={args.K}: boundary live bytes {best[args.K][n] / 2**20:.2f} MiB/frame (x2 for out+in); "
          f"reference partition in this order: {ref_live / 2**20:.2f} MiB")
    for c, (i, j) in enumerate(cuts):
        print(f"  ctx{c}: {order[i]} .. {order[j - 1]} ({j - i} layers) "
              + " ".join(f"{k[:8]}={pref[k][j] - pref[k][i]:.0f}/{cap[k]:.0f}" for k in RES)
              + (f"  | cut after: {live[j] / 2**20:.2f} MiB" if j < n else ""))
    if args.emit_json:
        m = {order[p]: c for c, (i, j) in enumerate(cuts) for p in range(i, j)}
        for p in order:  # output layers live with their producer
            if types[p] == "output_layer":
                prod = [q for q in order if p in succ[q]]
                if prod:
                    m[p] = m[prod[0]]
        json.dump(m, open(args.emit_json, "w"))
        print("wrote", args.emit_json)
    if args.emit:
        with open(args.emit, "w") as f:
            f.write("context_switch_param(mode=manual)\n")
            for c, (i, j) in enumerate(cuts):
                f.write(f"ctx{c} = context([{', '.join(order[i:j])}])\n")
        print("wrote", args.emit)


if __name__ == "__main__":
    main()
