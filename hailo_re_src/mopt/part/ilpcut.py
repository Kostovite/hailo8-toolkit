"""Minimum-PCIe context assignment for Hailo-8 as a MILP (scipy/HiGHS).

Variables: x[v,c] = HN layer v runs in context c; z[e,c] = tensor of edge e (u -> v) is DMA'd into context c.
  sum_c x[v,c] = 1
  ctx(u) <= ctx(v) for every edge (precedence: sum_c c*x[u,c] <= sum_c c*x[v,c])
  z[u,c] >= x[v,c] - x[u,c] for every consumer v of producer u's output port   (port tensor needed in c, made earlier)
  sum_v res_k(v) x[v,c] <= cap_k                                               (per-context resources)
  x[u,c] = x[v,c] for layers the reference compile packed into one cluster group (keeps them mappable)
minimize  sum_{port,c} 2 * bytes(port) * z[port,c]      (out + in per consuming context)

Usage: python ilpcut.py HN HW_JSON REF_ALLS K [--slack 1.0] [--time 600] [--attn_groups|--no_groups] [--emit_json OUT.json] [--eval REF_JSON]
"""
import argparse
import collections
import json
import re

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

import cutsolve as CS


def ports(hn):
    """(producer, port bytes, [consumers]) for every output port of every HN layer."""
    L = hn["layers"]
    out = []
    for n, v in L.items():
        u = n.split("/")[-1]
        groups = collections.defaultdict(list)
        shapes = v.get("output_shapes", [])
        for i, s in enumerate(v.get("output", [])):
            if s not in L:
                continue
            shp = shapes[i] if i < len(shapes) else shapes[0]
            el = int(np.prod([max(int(d), 1) for d in shp[1:]]))
            groups[el].append(s.split("/")[-1])
        for el, cons in groups.items():
            out.append((u, el, cons))
    return out


def ref_groups(alls, names):
    by_len = sorted(names, key=len, reverse=True)
    groups = []
    for m in re.finditer(r"^\w+_context_\d+\.place\(\d+, \[([^\]]*)\]\)", alls, re.M):
        g = set()
        for node in m.group(1).split(","):
            node = node.strip()
            h = next((h for h in by_len if node == h or node.startswith(h + "_")), None)
            if h is not None and not node.startswith(("concat_", "sh_from", "ws_from", "fs_from", "auto_reshape")):
                g.add(h)
        if len(g) > 1:
            groups.append(sorted(g))
    return groups


def cost_of(assign, P):
    tot = 0
    for u, b, cons in P:
        tot += 2 * b * len({assign[v] for v in cons if assign[v] > assign[u]})
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hn")
    ap.add_argument("hw")
    ap.add_argument("ref_alls")
    ap.add_argument("K", type=int)
    ap.add_argument("--slack", type=float, default=1.0)
    ap.add_argument("--time", type=float, default=600)
    ap.add_argument("--emit_json", default=None)
    ap.add_argument("--eval", default=None, help="only print the model cost of this layer->ctx json")
    ap.add_argument("--no_groups", action="store_true")
    ap.add_argument("--attn_groups", action="store_true", help="keep only cluster groups that contain matmul/softmax layers")
    args = ap.parse_args()
    hn = json.load(open(args.hn))
    order, nbytes, succ, types = CS.hn_graph(hn)
    P = ports(hn)
    alls = open(args.ref_alls).read()
    ref = json.load(open(args.eval)) if args.eval else None
    if ref is not None:
        missing = [v for v in order if v not in ref]
        for v in missing:  # output layers etc.: with their producer
            prod = [u for u, _, cons in P if v in cons]
            ref[v] = ref[prod[0]] if prod and prod[0] in ref else 0
        print(f"model cost {cost_of(ref, P) / 2**20:.2f} MiB/frame (inter-context out+in); {len(missing)} filled")
        return
    hw = json.load(open(args.hw))
    res, _ = CS.layer_resources(hw, order)
    refc = CS.ref_contexts(alls)
    load = collections.defaultdict(collections.Counter)
    for v in order:
        c = refc.get(v)
        if c is None:
            c = next((refc[m] for m in refc if m.startswith(v + "_")), None)
        if c is not None:
            load[c].update(res[v])
    cap = {k: args.slack * max(load[c][k] for c in load) for k in CS.RES}
    K, V = args.K, len(order)
    idx = {v: i for i, v in enumerate(order)}
    nx = V * K
    nz = len(P) * K
    N = nx + nz
    X = lambda v, c: idx[v] * K + c  # noqa: E731
    Z = lambda p, c: nx + p * K + c  # noqa: E731
    cost = np.zeros(N)
    for p, (u, b, cons) in enumerate(P):
        for c in range(K):
            cost[Z(p, c)] = 2.0 * b
    rows, lo, hi = [], [], []
    A = lil_matrix((V + 4 * len(P) * K + V * 4 + K * len(CS.RES) + 4000, N))
    r = 0

    def add(coefs, l, h):
        nonlocal r
        for j, a in coefs:
            A[r, j] = a
        lo.append(l)
        hi.append(h)
        r += 1

    for v in order:
        add([(X(v, c), 1) for c in range(K)], 1, 1)
    edges = [(u, v) for u, _, cons in P for v in cons]
    for u, v in set(edges):
        add([(X(u, c), c) for c in range(K)] + [(X(v, c), -c) for c in range(K)], -np.inf, 0)
    for p, (u, b, cons) in enumerate(P):
        for v in cons:
            for c in range(K):  # z >= x[v,c] - x[u,c]
                add([(Z(p, c), 1), (X(v, c), -1), (X(u, c), 1)], 0, np.inf)
    for k in CS.RES:
        for c in range(K):
            add([(X(v, c), res[v][k]) for v in order if res[v][k]], -np.inf, cap[k])
    if not args.no_groups:
        for g in ref_groups(alls, order):
            if args.attn_groups and not any(t in v for v in g for t in ("matmul", "softmax")):
                continue
            for a, b2 in zip(g, g[1:]):
                for c in range(K):
                    add([(X(a, c), 1), (X(b2, c), -1)], 0, 0)
    # input layer in context 0
    inp = [v for v in order if types[v] == "input_layer"][0]
    add([(X(inp, 0), 1)], 1, 1)
    A = A[:r].tocsr()
    integrality = np.zeros(N)
    integrality[:nx] = 1
    sol = milp(cost, constraints=LinearConstraint(A, lo, hi), integrality=integrality, bounds=Bounds(0, 1),
               options={"time_limit": args.time, "disp": False, "mip_rel_gap": 1e-3})
    print("status", sol.status, sol.message)
    if sol.x is None:
        return
    x = sol.x[:nx].reshape(V, K)
    assign = {v: int(np.argmax(x[idx[v]])) for v in order}
    print(f"K={K}: model cost {cost_of(assign, P) / 2**20:.2f} MiB/frame (out+in), bound {sol.mip_dual_bound / 2**20:.2f}")
    for c in range(K):
        mem = [v for v in order if assign[v] == c]
        ld = {k: sum(res[v][k] for v in mem) for k in CS.RES}
        print(f"  ctx{c}: {len(mem)} layers " + " ".join(f"{k[:8]}={ld[k]:.0f}/{cap[k]:.0f}" for k in CS.RES))
    if args.emit_json:
        json.dump(assign, open(args.emit_json, "w"))
        print("wrote", args.emit_json)


if __name__ == "__main__":
    main()
