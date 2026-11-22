"""Any unary function on Hailo-8 as native APU activations, at any precision.

The Hailo-8 APU evaluates a per-channel piecewise-linear function of <= 9 pieces whose slopes must be NON-NEGATIVE
(SAGE mantissas are unsigned). A K-piece fit of f, written in hinge form

    f(x) ~ a + c0 * x + sum_k c_k * relu(x - b_k)

is split by the sign of each hinge coefficient into non-decreasing PWL layers of <= 8 hinges each:
    f = sum_j G_j(x) - sum_j H_j(x)       (G_j: c_k > 0 hinges, H_j: |c_k| for c_k < 0; c0 goes to G or H by sign)
so f costs ceil(#pos/8) + ceil(#neg/8) activation layers plus adds. Breakpoints are fitted (least squares on the
calibrated input distribution + 25% uniform mass) by coordinate descent.

  1. fit   (any env with onnxruntime):  python pwl_inject.py fit IN.onnx CALIB.npy FIT.json --ops Sin Erf [--pieces 17]
  2. mark  (any env):                   python pwl_inject.py mark IN.onnx FIT.json OUT.onnx
             each fitted node -> Relu placeholders named pwl<k>__G0.. / __H0.. + Add/Sub tree
  3. apply (DFC env, after translate_onnx_model):  from pwl_inject import apply_pwl; apply_pwl(runner, FIT.json)
"""
import argparse
import json
from math import erf as _erf

import numpy as np

FUNCS = {
    "Sin": np.sin, "Cos": np.cos, "Tan": np.tan, "Atan": np.arctan, "Asin": np.arcsin, "Acos": np.arccos,
    "Sinh": np.sinh, "Cosh": np.cosh, "Asinh": np.arcsinh, "Acosh": np.arccosh, "Atanh": np.arctanh,
    "Erf": np.vectorize(_erf), "Sign": np.sign, "Round": np.round, "Floor": np.floor, "Ceil": np.ceil,
    "Selu": lambda x: 1.0507009873554805 * np.where(x > 0, x, 1.6732632423543772 * (np.exp(x) - 1)),
    "Celu": lambda x: np.maximum(0, x) + np.minimum(0, np.exp(x) - 1),
    "SiLU": lambda x: x / (1 + np.exp(-x)),  # the Mul(x, Sigmoid(x)) pattern, see _silu_nodes
    "Exp": np.exp, "Log": np.log, "Reciprocal": lambda x: 1 / x, "Sqrt": np.sqrt,
}
MAX_HINGES = 8  # 9 pieces per APU
UNIFORM = 0.25  # uniform mass added over the observed range (fraction of the samples): tames the tails


def fit_free(f, x, n_pieces=9, iters=60, restarts=4):
    """Best ONE-layer APU approximation: pieces are independent lines with slope >= 0 and FREE offsets (the APU does
    not require continuity; Hailo's own SiLU/GELU use flat pieces at descending offsets for their dips). Per piece:
    least squares line, slope clamped at 0 (then the mean). Breakpoints: coordinate descent from several starts.
    Returns (thresholds, slopes, offsets, snr, range)."""
    x = np.sort(np.asarray(x, np.float64).ravel())
    if x.size > 20000:
        x = x[np.linspace(0, x.size - 1, 20000).astype(int)]
    lo, hi = x[0], x[-1]
    x_all = np.sort(np.concatenate([x, np.linspace(lo, hi, int(x.size * UNIFORM))]))
    y_all = f(x_all)
    sub = np.linspace(0, x_all.size - 1, min(6000, x_all.size)).astype(int)
    xs, ys = x_all[sub], y_all[sub]
    # prefix sums for O(1) per-piece least squares
    def prefix(xx, yy):
        return [np.concatenate([[0.0], np.cumsum(v)]) for v in (np.ones_like(xx), xx, yy, xx * xx, xx * yy, yy * yy)]
    S = prefix(xs, ys)

    def piece(i0, i1, S=S):
        n, sx, sy, sxx, sxy, syy = (s[i1] - s[i0] for s in S)
        if n < 1:
            return 0.0, 0.0, 0.0
        vx = sxx - sx * sx / n
        slope = (sxy - sx * sy / n) / vx if vx > 1e-12 else 0.0
        slope = max(slope, 0.0)
        off = (sy - slope * sx) / n
        sse = syy - 2 * slope * sxy - 2 * off * sy + slope * slope * sxx + 2 * slope * off * sx + n * off * off
        return slope, off, max(sse, 0.0)

    def cost(b, xx=xs, S=S):
        idx = np.concatenate([[0], np.searchsorted(xx, b), [len(xx)]])
        return sum(piece(idx[i], idx[i + 1], S)[2] for i in range(len(idx) - 1))

    best_b, best_e = None, np.inf
    starts = [np.quantile(xs, np.linspace(0, 1, n_pieces + 1)[1:-1]), np.linspace(lo, hi, n_pieces + 1)[1:-1]]
    rng = np.random.default_rng(0)
    starts += [np.sort(rng.uniform(lo, hi, n_pieces - 1)) for _ in range(max(restarts - 2, 0))]
    span = hi - lo
    for b in starts:
        err = cost(b)
        for it in range(iters):
            step = span / (n_pieces * (1.5 + it))
            improved = False
            for k in range(len(b)):
                for d in (-step, step, -step / 4, step / 4):
                    nb = np.sort(np.clip(np.where(np.arange(len(b)) == k, b + d, b), lo, hi))
                    e2 = cost(nb)
                    if e2 < err:
                        b, err, improved = nb, e2, True
            if not improved and step < span * 1e-4:
                break
        if err < best_e:
            best_b, best_e = b, err
    b = best_b
    SA = prefix(x_all, y_all)
    idx = np.concatenate([[0], np.searchsorted(x_all, b), [len(x_all)]])
    fits = [piece(idx[i], idx[i + 1], SA) for i in range(len(idx) - 1)]
    sl = np.array([p[0] for p in fits])
    of = np.array([p[1] for p in fits])
    err = sum(p[2] for p in fits) / len(x_all)
    snr = 10 * np.log10((y_all ** 2).mean() / max(err, 1e-30))
    return b, sl, of, float(snr), (float(lo), float(hi))


def fit_monotone(f, x, n_pieces=9, iters=80):
    """Best NON-DECREASING K-piece continuous PWL (one APU layer, fusable into the producing conv): slopes s_i >= 0
    by NNLS on ramp features, breakpoints by coordinate descent. Same (b, a, c0, c, snr, range) as fit_hinges."""
    from scipy.optimize import nnls
    x = np.sort(np.asarray(x, np.float64).ravel())
    if x.size > 20000:
        x = x[np.linspace(0, x.size - 1, 20000).astype(int)]
    lo, hi = x[0], x[-1]
    x_all = np.sort(np.concatenate([x, np.linspace(lo, hi, int(x.size * UNIFORM))]))
    y_all = f(x_all)
    sub = np.linspace(0, x_all.size - 1, min(4000, x_all.size)).astype(int)
    xs, ys = x_all[sub], y_all[sub]

    def feats(xx, b):
        e = np.concatenate([[lo], b, [hi]])
        # piece i contributes s_i * (length of [e_i, e_i+1] covered by x), extended linearly beyond the range ends
        R = [np.minimum(xx, e[1]) - lo]
        R += [np.clip(xx - e[i], 0, e[i + 1] - e[i]) for i in range(1, len(e) - 2)]
        R += [np.maximum(xx - e[-2], 0)]
        return np.stack(R, 1)

    def solve(b, xx=None, yy=None):
        xx, yy = (xs, ys) if xx is None else (xx, yy)
        A = feats(xx, b)
        A2 = np.concatenate([A, np.ones((len(xx), 1)), -np.ones((len(xx), 1))], 1)  # free constant = c+ - c-
        coef, _ = nnls(A2, yy, maxiter=2000)
        pred = A2 @ coef
        return coef, float(((pred - yy) ** 2).mean())

    b = np.quantile(xs, np.linspace(0, 1, n_pieces + 1)[1:-1])
    coef, err = solve(b)
    span = hi - lo
    for it in range(iters):
        step = span / (n_pieces * (2 + it))
        improved = False
        for k in range(len(b)):
            for d in (-step, step):
                nb = np.sort(np.clip(np.where(np.arange(len(b)) == k, b + d, b), lo + 1e-9, hi - 1e-9))
                c2, e2 = solve(nb)
                if e2 < err:
                    b, coef, err, improved = nb, c2, e2, True
        if not improved and step < span * 1e-4:
            break
    coef, err = solve(b, x_all, y_all)
    s, const = coef[:n_pieces], coef[n_pieces] - coef[n_pieces + 1]
    # back to hinge form: f = a + c0*x + sum c_k relu(x - b_k), value at lo = const
    c0 = s[0]
    c = np.diff(s)
    a = const - c0 * lo
    snr = 10 * np.log10((y_all ** 2).mean() / max(err, 1e-30))
    return b, a, c0, c, float(snr), (float(lo), float(hi))


def fit_hinges(f, x, n_pieces, iters=80):
    """K-piece continuous least-squares PWL. Returns (b, a, c0, c, snr, range): f ~ a + c0*x + sum c_k relu(x-b_k)."""
    x = np.sort(np.asarray(x, np.float64).ravel())
    if x.size > 20000:
        x = x[np.linspace(0, x.size - 1, 20000).astype(int)]
    lo, hi = x[0], x[-1]
    # 25% uniform mass: the empirical density alone leaves the tails unconstrained (wild slopes there inflate the
    # output range the quantizer has to cover)
    x_all = np.sort(np.concatenate([x, np.linspace(lo, hi, int(x.size * UNIFORM))]))
    y_all = f(x_all)
    # breakpoint search on a 4k subsample (coordinate descent solves one least squares per move), final solve on all
    sub = np.linspace(0, x_all.size - 1, min(4000, x_all.size)).astype(int)
    x, y = x_all[sub], y_all[sub]

    def solve(b, xx=None, yy=None):
        xx, yy = (x, y) if xx is None else (xx, yy)
        A = np.stack([np.ones_like(xx), xx] + [np.maximum(xx - bk, 0) for bk in b], 1)
        coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
        return coef, float(((A @ coef - yy) ** 2).mean())

    b = np.quantile(x, np.linspace(0, 1, n_pieces + 1)[1:-1])
    coef, err = solve(b)
    span = hi - lo
    for it in range(iters):
        step = span / (n_pieces * (2 + it))
        improved = False
        for k in range(len(b)):
            for d in (-step, step):
                nb = np.sort(np.clip(np.where(np.arange(len(b)) == k, b + d, b), lo, hi))
                c2, e2 = solve(nb)
                if e2 < err:
                    b, coef, err, improved = nb, c2, e2, True
        if not improved and step < span * 1e-4:
            break
    coef, err = solve(b, x_all, y_all)
    snr = 10 * np.log10((y_all ** 2).mean() / max(err, 1e-30))
    return b, coef[0], coef[1], coef[2:], float(snr), (float(lo), float(hi))


WINDOW = MAX_HINGES - 1  # pieces of f per APU layer: a flat piece on each side keeps the layer bounded


def to_layers(b, a, c0, c, lo):
    """Hinge form -> bounded non-decreasing PWL layers {"G0": (thresholds, slopes, offsets), ..., "H0": ...}.

    f's K pieces have slopes s_i; G takes max(s_i, 0), H takes max(-s_i, 0) (Jordan decomposition, so G - H = f).
    Each layer covers a window of <= 7 consecutive pieces and is FLAT outside it (non-negative slopes may return to
    0): a layer's output range is the variation inside its window, not the cumulative rise to the end of the range
    (which is what wrecks 8-bit outputs if a layer keeps its last slope)."""
    K = len(b) + 1
    s = c0 + np.concatenate([[0.0], np.cumsum(c)])  # slope of piece i
    f_lo = a + c0 * lo  # all b_k >= lo
    layers = {}
    # windows of consecutive pieces: 9 per APU minus a flat piece on each side that is not the end of the range
    windows, p0 = [], 0
    while p0 < K:
        cap = MAX_HINGES + 1 - (p0 > 0)
        p1 = K - 1 if K - p0 <= cap else p0 + cap - 2
        windows.append((p0, p1))
        p0 = p1 + 1
    for sign, sl in (("G", np.maximum(s, 0)), ("H", np.maximum(-s, 0))):
        j = 0
        for p0, p1 in windows:
            if not sl[p0:p1 + 1].any() and not (sign == "G" and p0 == 0):
                continue
            th, slopes = [], []
            if p0 > 0:  # flat before the window
                th.append(b[p0 - 1])
                slopes.append(0.0)
            slopes.append(sl[p0])
            for i in range(p0 + 1, p1 + 1):
                th.append(b[i - 1])
                slopes.append(sl[i])
            if p1 < K - 1:  # flat after the window
                th.append(b[p1])
                slopes.append(0.0)
            # offsets by continuity; anchor: value 0 at the window's left edge (x = lo for window 0), plus f(lo)
            x0 = lo if p0 == 0 else b[p0 - 1]
            first = 1 if p0 > 0 else 0  # index of the first in-window piece
            offs = [0.0] * len(slopes)
            anchor = (f_lo if (sign == "G" and p0 == 0) else 0.0)
            offs[first] = anchor - slopes[first] * x0
            for k in range(first + 1, len(slopes)):
                offs[k] = offs[k - 1] + (slopes[k - 1] - slopes[k]) * th[k - 1]
            for k in range(first - 1, -1, -1):
                offs[k] = offs[k + 1] + (slopes[k + 1] - slopes[k]) * th[k]
            layers[f"{sign}{j}"] = (list(map(float, th)), list(map(float, slopes)), list(map(float, offs)))
            j += 1
    return layers


STEPS = {  # staircases: jump positions in [lo, hi] and the value left of the first jump (APU offsets are free per
    # piece, so jumps are exact; a threshold is the first x of the new piece, i.e. f is right-continuous)
    "Sign": lambda lo, hi: ([0.0, 1e-6], lambda x: np.sign(x)),
    "Floor": lambda lo, hi: (list(np.arange(np.floor(lo) + 1, np.floor(hi) + 1)), np.floor),
    "Ceil": lambda lo, hi: (list(np.arange(np.ceil(lo), np.ceil(hi)) + 1e-6), np.ceil),
    "Round": lambda lo, hi: (list(np.arange(np.floor(lo) + 0.5, np.ceil(hi) - 0.5 + 1e-9)), np.round),
    # truncation toward 0 (C fmod): jumps at the nonzero integers, up at +k from the right, at -k from the left
    "Trunc": lambda lo, hi: ([float(k) for k in np.arange(np.floor(lo) + 1, np.floor(hi) + 1) if k > 0] +
                             [float(k) + 1e-6 for k in np.arange(np.ceil(lo), np.ceil(hi)) if k < 0], np.trunc),
}


EXACT = {  # piecewise-linear functions with upward jumps: one exact non-decreasing APU layer (attr -> pieces)
    "ThresholdedRelu": lambda n, a: ([a.get("alpha", 1.0)], [0.0, 1.0], [0.0, 0.0]),
    "Hardshrink": lambda n, a: ([-a.get("lambd", 0.5), a.get("lambd", 0.5) + 1e-6], [1.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
}


def step_layers(func, lo, hi):
    """Exact staircase as non-decreasing zero-slope PWL layers of <= 8 jumps each (sum of the layers)."""
    jumps, f = STEPS[func](lo, hi)
    jumps = sorted(j for j in jumps if lo < j <= hi)
    # value of every global piece, evaluated at its midpoint (robust to where exactly the jump sits)
    edges = [lo] + jumps + [hi]
    v = [float(f(np.array([(a + b) / 2]))[0]) for a, b in zip(edges[:-1], edges[1:])]
    layers = {}
    for j, k in enumerate(range(0, max(len(jumps), 1), MAX_HINGES)):
        th = jumps[k:k + MAX_HINGES] or [hi]
        vals = [v[0] if j == 0 else 0.0]
        for i in range(k, k + len(th)):  # jump i (between global pieces i and i+1)
            vals.append(vals[-1] + (v[i + 1] - v[i] if i + 1 < len(v) else 0.0))
        layers[f"G{j}"] = (list(map(float, th)), [0.0] * (len(th) + 1), vals)
    return layers


def _silu_nodes(m):
    """Mul(x, Sigmoid(x)) nodes -> {mul name: x}: SiLU has no ONNX op, its pattern is replaced at the Mul."""
    sig = {n.output[0]: n.input[0] for n in m.graph.node if n.op_type == "Sigmoid"}
    return {n.name: [i for i in n.input if i not in sig][0] for n in m.graph.node
            if n.op_type == "Mul" and len(n.input) == 2 and any(sig.get(i) in n.input for i in n.input)}


def _trunc_nodes(m):
    """Mul(Sign(t), Floor(Abs(t))) -> {mul name: t} (compat2 mod, fmod=1): one Trunc staircase, no ew_mult."""
    prod = {o: n for n in m.graph.node for o in n.output}
    out = {}
    for n in m.graph.node:
        if n.op_type != "Mul" or len(n.input) != 2:
            continue
        a, b = (prod.get(i) for i in n.input)
        for sg, fl in ((a, b), (b, a)):
            if sg is None or fl is None or sg.op_type != "Sign" or fl.op_type != "Floor":
                continue
            ab = prod.get(fl.input[0])
            if ab is not None and ab.op_type == "Abs" and ab.input[0] == sg.input[0]:
                out[n.name] = (sg.input[0], {sg.name, fl.name})
    return out


def _x(n, silu):
    return silu.get(n.name, n.input[0])


def fit(src, calib_npy, out_json, ops, n_pieces):
    import onnx
    import onnxruntime as ort
    m = onnx.load(src)
    silu = _silu_nodes(m) if "SiLU" in ops else {}
    trunc = _trunc_nodes(m) if "Trunc" in ops else {}
    inner = set().union(*[v[1] for v in trunc.values()]) if trunc else set()
    silu.update({k: v[0] for k, v in trunc.items()})
    targets = [n for n in m.graph.node if (n.op_type in ops and n.name not in inner) or n.name in silu]
    have = {o.name for o in m.graph.output}
    for n in targets:
        if _x(n, silu) not in have:
            m.graph.output.append(onnx.helper.make_empty_tensor_value_info(_x(n, silu)))
            have.add(_x(n, silu))
    sess = ort.InferenceSession(m.SerializeToString())
    calib = np.load(calib_npy).astype(np.float32)
    names = [o.name for o in sess.get_outputs()]
    taps = {_x(n, silu): [] for n in targets}
    for i in range(len(calib)):
        outs = dict(zip(names, sess.run(None, {sess.get_inputs()[0].name: calib[i:i + 1]})))
        for t in taps:
            taps[t].append(outs[t].ravel())
    res = {}
    for k, n in enumerate(targets):
        x = np.concatenate(taps[_x(n, silu)])
        func = "Trunc" if n.name in trunc else "SiLU" if n.name in silu else n.op_type
        if n.op_type in EXACT:
            import onnx.helper as oh
            attrs = {a.name: oh.get_attribute_value(a) for a in n.attribute}
            rng = (float(x.min()), float(x.max()))
            layers, snr = {"G0": EXACT[n.op_type](n, attrs)}, float("inf")
        elif func in STEPS:
            rng = (float(x.min()), float(x.max()))
            layers, snr = step_layers(func, *rng), float("inf")
        else:
            b, a, c0, c, snr, rng = fit_hinges(FUNCS[func], x, n_pieces)
            layers = to_layers(b, a, c0, c, rng[0])
        key = f"pwl{k}"
        res[key] = {"node": n.name, "x": _x(n, silu), "func": func, "range": rng, "pieces": n_pieces, "fit_sqnr_db": snr,
                    "layers": layers}
        print(f"{n.name}: {n.op_type} on [{rng[0]:.3f}, {rng[1]:.3f}] {n_pieces} pieces -> fit {snr:.1f} dB, "
              f"{len(layers)} APU layers ({', '.join(layers)})")
    json.dump(res, open(out_json, "w"), indent=1)


def mark(src, fit_json, dst):
    import onnx
    from onnx import helper
    fits = json.load(open(fit_json))
    by_node = {v["node"]: (k, v) for k, v in fits.items()}
    m = onnx.load(src)
    nodes = []
    for n in m.graph.node:
        if n.name not in by_node:
            nodes.append(n)
            continue
        key, v = by_node[n.name]
        x, y = v.get("x", n.input[0]), n.output[0]
        parts = {"G": [], "H": []}
        for lname in v["layers"]:
            t = f"{key}__{lname}"
            nodes.append(helper.make_node("Relu", [x], [t], name=t))
            parts[lname[0]].append(t)

        def tree(ts, tag):
            while len(ts) > 1:
                nxt = []
                for i in range(0, len(ts) - 1, 2):
                    o = f"{key}__{tag}{len(nodes)}"
                    nodes.append(helper.make_node("Add", [ts[i], ts[i + 1]], [o], name=o))
                    nxt.append(o)
                ts = nxt + (ts[-1:] if len(ts) % 2 else [])
            return ts[0] if ts else None

        g, h = tree(parts["G"], "g"), tree(parts["H"], "h")
        if h is None:
            nodes.append(helper.make_node("Identity", [g], [y], name=f"{key}__out"))
        else:
            nodes.append(helper.make_node("Sub", [g, h], [y], name=f"{key}__out"))
    used = {i for n in nodes for i in n.input} | {o.name for o in m.graph.output}
    nodes = [n for n in nodes if any(o in used for o in n.output)]  # the orphaned Sigmoid of a SiLU pattern
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    onnx.save(m, dst)
    print(f"marked {len(fits)} nodes -> {dst}")


def apply_pwl(runner, fit_json):
    """Marked layers -> activation=pwl with their fitted (thresholds, slopes, offsets). A placeholder Relu right
    after a conv is fused into the conv by the parser: the conv's own APU then evaluates the PWL (no extra layer)."""
    fits = json.load(open(fit_json))
    want = {f"{k}__{ln}": tuple(p) for k, v in fits.items() for ln, p in v["layers"].items()}
    hn = runner.get_hn_dict()
    params = dict(runner.get_params())
    done = []
    for name, L in hn["layers"].items():
        hit = [o for o in L.get("original_names", []) if o in want]
        if not hit or "activation" not in L.get("params", {}):
            continue
        assert len(hit) == 1, (name, hit)
        th, sl, of = want[hit[0]]
        L["params"]["activation"] = "pwl"
        params[f"{name}/thresholds"] = np.asarray(th, np.float32)
        params[f"{name}/slopes"] = np.asarray(sl, np.float32)
        params[f"{name}/offsets"] = np.asarray(of, np.float32)
        done.append(name)
    assert len(done) == len(want), f"placeholders found {len(done)} of {len(want)}: {sorted(want)}"
    runner.set_hn(hn)
    runner.load_params(params)
    return done


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "mark"])
    ap.add_argument("args", nargs="+")
    ap.add_argument("--ops", nargs="+", default=[])
    ap.add_argument("--pieces", type=int, default=9)
    a = ap.parse_args()
    if a.cmd == "fit":
        fit(a.args[0], a.args[1], a.args[2], set(a.ops), a.pieces)
    else:
        mark(*a.args[:3])
