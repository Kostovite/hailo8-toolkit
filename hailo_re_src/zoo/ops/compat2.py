"""More ONNX -> Hailo-8 rewrites (registered into onnx_hailo_compat.ALL). Exact in float unless noted.

  ln_axis      LayerNormalization over W of an NCHW tensor -> W<->C transposes around the channel LayerNorm pattern
  logsumexp    ReduceLogSumExp -> max + log(sum(exp(x - max)))
  reduce_prod  ReduceProd -> exp(sum(log x))                                  (x > 0)
  lp_norm      LpNormalization p=2 -> x / ReduceL2, p=1 -> x / ReduceSum(|x|)
  lp_pool      LpPool p -> (k * AvgPool(|x|^p))^(1/p)  (p=2: Sqrt)
  mvn          MeanVarianceNormalization -> InstanceNormalization(scale 1, bias 0)
  where        Where(c, a, b) -> b + m * (a - b) with c's boolean ops as 0/1 arithmetic (And = m1 m2,
               Or = m1 + m2 - m1 m2, Not = 1 - m, comparisons -> Cast(...) float)
  mod          Mod(x, m) -> x - m * Floor(x / m)              (Floor then via pwl_inject steps)
  pow_tensor   Pow(x, y) tensors -> Exp(y * Log(x))           (x > 0)
  flip         Slice step -1 on H/W -> Transpose to channels + Gather(reversed) + Transpose back
  cumsum       CumSum over H/W -> log-step shift-and-add (Pad + Slice + Add)
  hardmax      Hardmax -> Cast(Greater(x - ReduceMax(x), -eps))   (approx: ties give several ones)
  matmul_const MatMul(x NCHW, K x M const) over W -> W<->C Transpose + 1x1 Conv + Transpose back
  lrn          LRN -> x / (bias + alpha/size * band_conv(x^2))^beta
  equal        Cast(Equal(x, c)) -> Cast(Greater(x, c - .5)) * Cast(Less(x, c + .5))  (exact for integer-valued x)
  gelu_any     GELU written in any Add/Mul order around Erf(x / sqrt2) -> torch's order (verified numerically)
"""
import math

import numpy as np
import onnx
from onnx import TensorProto, helper

RULES = ["equal", "group_norm", "ln_axis", "logsumexp", "reduce_prod", "lp_norm", "lp_pool", "mvn", "where", "mod", "pow_tensor", "flip",
         "cumsum", "hardmax", "matmul_const", "lrn", "gelu_any"]


def _rw(core):
    return core.rewrite


def rule_ln_axis(r, core):
    def f(n):
        if n.op_type != "LayerNormalization" or r.rank(n.input[0]) != 4:
            return None
        ax = r.attr(n, "axis", -1) % 4
        if ax != 3:
            return None
        x, y = n.input[0], n.output[0]
        a, t1 = r.node("Transpose", [x], perm=[0, 3, 2, 1])      # W <-> C: the normalized axis is now channels
        b, t2 = r.node("Transpose", [t1], perm=[0, 2, 3, 1])     # channel LayerNorm pattern (NCHW -> NHWC)
        ln = helper.make_node("LayerNormalization", [t2] + list(n.input[1:]), [t2 + "_ln"], name=n.name + "/ln",
                              axis=-1, epsilon=r.attr(n, "epsilon", 1e-5))
        c, t3 = r.node("Transpose", [t2 + "_ln"], perm=[0, 3, 1, 2])
        d = helper.make_node("Transpose", [t3], [y], name=n.name + "/back", perm=[0, 3, 2, 1])
        r.hit("ln_axis")
        return [a, b, ln, c, d]
    core.rewrite(r, f)


def _axes(r, n):
    return r.axes_of(n)


def rule_logsumexp(r, core):
    def f(n):
        if n.op_type != "ReduceLogSumExp" or not r.attr(n, "keepdims", 1):
            return None
        ax = _axes(r, n)
        x = n.input[0]
        m, tm = r.node("ReduceMax", [x], axes=ax, keepdims=1)
        s, ts = r.node("Sub", [x, tm])
        e, te = r.node("Exp", [ts])
        axc = r.c(np.array(ax, np.int64), np.int64)
        su, tsu = r.node("ReduceSum", [te, axc], keepdims=1)
        lg, tl = r.node("Log", [tsu])
        r.hit("logsumexp")
        return [m, s, e, su, lg, helper.make_node("Add", [tl, tm], list(n.output), name=n.name + "/lse")]
    core.rewrite(r, f)


def rule_reduce_prod(r, core):
    def f(n):
        if n.op_type != "ReduceProd":
            return None
        ax = _axes(r, n)
        lg, tl = r.node("Log", [n.input[0]])
        su, ts = r.node("ReduceSum", [tl, r.c(np.array(ax, np.int64), np.int64)], keepdims=r.attr(n, "keepdims", 1))
        r.hit("reduce_prod")
        return [lg, su, helper.make_node("Exp", [ts], list(n.output), name=n.name + "/prod")]
    core.rewrite(r, f)


def rule_lp_norm(r, core):
    def f(n):
        if n.op_type != "LpNormalization":
            return None
        p, ax = r.attr(n, "p", 2), r.attr(n, "axis", -1)
        x = n.input[0]
        if p == 2:
            d, td = r.node("ReduceL2", [x], axes=[ax], keepdims=1)
            nodes = [d]
        elif p == 1:
            a, ta = r.node("Abs", [x])
            d, td = r.node("ReduceSum", [ta, r.c(np.array([ax], np.int64), np.int64)], keepdims=1)
            nodes = [a, d]
        else:
            return None
        r.hit("lp_norm")
        return nodes + [helper.make_node("Div", [x, td], list(n.output), name=n.name + "/lpn")]
    core.rewrite(r, f)


def rule_lp_pool(r, core):
    def f(n):
        if n.op_type != "LpPool":
            return None
        p = r.attr(n, "p", 2)
        k = r.attr(n, "kernel_shape")
        attrs = {a: r.attr(n, a) for a in ("kernel_shape", "strides", "pads") if r.attr(n, a) is not None}
        x = n.input[0]
        if p == 2:
            sq, t = r.node("Mul", [x, x])
            nodes = [sq]
        else:
            a, ta = r.node("Abs", [x])
            sq, t = r.node("Pow", [ta, r.c(np.array(float(p), np.float32))])
            nodes = [a, sq]
        ap, tp = r.node("AveragePool", [t], **attrs)
        mu, tm = r.node("Mul", [tp, r.c(np.array(float(np.prod(k)), np.float32))])
        nodes += [ap, mu]
        if p == 2:
            nodes.append(helper.make_node("Sqrt", [tm], list(n.output), name=n.name + "/lp"))
        else:
            nodes.append(helper.make_node("Pow", [tm, r.c(np.array(1.0 / p, np.float32))], list(n.output),
                                          name=n.name + "/lp"))
        r.hit("lp_pool")
        return nodes
    core.rewrite(r, f)


def rule_mvn(r, core):
    def f(n):
        if n.op_type != "MeanVarianceNormalization":
            return None
        ax = r.attr(n, "axes", [0, 2, 3])
        s = r.shapes.get(n.input[0])
        if sorted(a % 4 for a in ax) != [0, 2, 3] or not s or s[1] is None:
            return None
        c = s[1]
        r.hit("mvn")
        return [helper.make_node("InstanceNormalization", [n.input[0], r.c(np.ones(c)), r.c(np.zeros(c))],
                                 list(n.output), name=n.name + "/in", epsilon=1e-9)]
    core.rewrite(r, f)


_CMP = ("Greater", "Less", "GreaterOrEqual", "LessOrEqual", "Equal")


def rule_where(r, core):
    """Boolean subgraphs feeding Where become 0/1 float arithmetic."""
    prod = {o: n for n in r.g.node for o in n.output}
    memo = {}
    new_nodes = []

    def mask(t):
        if t in memo:
            return memo[t]
        n = prod.get(t)
        if n is None:
            raise ValueError("boolean graph input")
        if n.op_type in _CMP:
            a, tf_ = r.node("Cast", [t], to=TensorProto.FLOAT)
            new_nodes.append(a)
            out = tf_
        elif n.op_type == "Not":
            m = mask(n.input[0])
            a, out = r.node("Sub", [r.c(np.array(1.0, np.float32)), m])
            new_nodes.append(a)
        elif n.op_type in ("And", "Or", "Xor"):  # linear forms for 0/1 masks (a mask x mask product: 13 dB on chip)
            import compat3
            out = compat3.bool_linear(r, n.op_type, mask(n.input[0]), mask(n.input[1]), new_nodes)
        elif n.op_type == "Cast":
            out = mask(n.input[0])
        else:
            raise ValueError(n.op_type)
        memo[t] = out
        return out

    def f(n):
        if n.op_type != "Where":
            return None
        try:
            m = mask(n.input[0])
        except ValueError:
            return None
        cond_nodes = list(new_nodes)
        new_nodes.clear()
        d, td = r.node("Sub", [n.input[1], n.input[2]])
        mu, tm = r.node("Mul", [m, td])
        r.hit("where")
        return cond_nodes + [d, mu, helper.make_node("Add", [n.input[2], tm], list(n.output), name=n.name + "/where")]
    core.rewrite(r, f)


def rule_mod(r, core):
    def f(n):
        if n.op_type != "Mod":
            return None
        m = r.val(n.input[1])
        if m is None or m.size != 1:
            return None
        m = float(m.ravel()[0])
        x = n.input[0]
        dv, td = r.node("Div", [x, r.c(np.array(m, np.float32))])
        if r.attr(n, "fmod", 0):  # C fmod: truncation, trunc(t) = Sign(t) * Floor(|t|)
            ab, ta = r.node("Abs", [td])
            fl, tf0 = r.node("Floor", [ta])
            sg, tsg = r.node("Sign", [td])
            mt, tf_ = r.node("Mul", [tsg, tf0])
            pre = [dv, ab, fl, sg, mt]
        else:  # Python mod: floor
            fl, tf_ = r.node("Floor", [td])
            pre = [dv, fl]
        mu, tm = r.node("Mul", [tf_, r.c(np.array(m, np.float32))])
        r.hit("mod")
        return pre + [mu, helper.make_node("Sub", [x, tm], list(n.output), name=n.name + "/mod")]
    core.rewrite(r, f)


def rule_pow_tensor(r, core):
    def f(n):
        if n.op_type != "Pow" or r.val(n.input[1]) is not None:
            return None
        lg, tl = r.node("Log", [n.input[0]])
        mu, tm = r.node("Mul", [n.input[1], tl])
        r.hit("pow_tensor")
        return [lg, mu, helper.make_node("Exp", [tm], list(n.output), name=n.name + "/powt")]
    core.rewrite(r, f)


def rule_flip(r, core):
    def f(n):
        if n.op_type != "Slice" or len(n.input) < 5:
            return None
        st, ax = r.val(n.input[4]), r.val(n.input[3])
        s = r.shapes.get(n.input[0])
        if st is None or ax is None or st.size != 1 or int(st.ravel()[0]) != -1 or not s or len(s) != 4:
            return None
        axis = int(ax.ravel()[0]) % 4
        if axis not in (2, 3) or s[axis] is None:
            return None
        L = s[axis]
        perm = [0, axis, 2, 1] if axis == 2 else [0, 3, 2, 1]  # bring H or W to channels
        nodes, parts = [], []
        for i in range(L - 1, -1, -1):  # reversed single-row/column slices + one concat (exact, slice/concat only)
            sl, t = r.node("Slice", [n.input[0], r.c([i], np.int64), r.c([i + 1], np.int64), r.c([axis], np.int64)])
            nodes.append(sl)
            parts.append(t)
        r.hit("flip")
        return nodes + [helper.make_node("Concat", parts, list(n.output), name=n.name + "/flip", axis=axis)]
    core.rewrite(r, f)


def rule_cumsum(r, core):
    def f(n):
        if n.op_type != "CumSum" or r.attr(n, "exclusive", 0) or r.attr(n, "reverse", 0):
            return None
        ax = r.val(n.input[1])
        s = r.shapes.get(n.input[0])
        if ax is None or not s or len(s) != 4:
            return None
        axis = int(np.asarray(ax).ravel()[0]) % 4
        if axis not in (2, 3) or s[axis] is None:
            return None
        L = s[axis]
        nodes, y, k = [], n.input[0], 1
        while k < L:  # Hillis-Steele: y += shift(y, k)
            sl, ts = r.node("Slice", [y, r.c([0], np.int64), r.c([L - k], np.int64), r.c([axis], np.int64)])
            pads = [0] * 8
            pads[axis] = k
            pd, tp = r.node("Pad", [ts, r.c(np.array(pads, np.int64), np.int64)], mode="constant")
            ad, y = r.node("Add", [y, tp])
            nodes += [sl, pd, ad]
            k *= 2
        nodes.append(helper.make_node("Identity", [y], list(n.output), name=n.name + "/cumsum"))
        r.hit("cumsum")
        return nodes
    core.rewrite(r, f)


def rule_hardmax(r, core):
    def f(n):
        if n.op_type != "Hardmax":
            return None
        ax = r.attr(n, "axis", -1)
        x = n.input[0]
        m, tm = r.node("ReduceMax", [x], axes=[ax], keepdims=1)
        s, ts = r.node("Sub", [x, tm])  # <= 0, == 0 at the max
        # steep ramp: relu(1 + K t) - relu(K t): 1 at t = 0, 0 for t <= -1/K  (K: far above 1 / input step)
        k_ = r.c(np.array(1e3, np.float32))
        mu, tk = r.node("Mul", [ts, k_])
        a1, t1 = r.node("Add", [tk, r.c(np.array(1.0, np.float32))])
        r1, tr1 = r.node("Relu", [t1])
        r.hit("hardmax")
        return [m, s, mu, a1, r1, helper.make_node("Clip", [tr1, r.c(np.array(0.0, np.float32)), r.c(np.array(1.0, np.float32))],
                                                     list(n.output), name=n.name + "/hardmax")]
    core.rewrite(r, f)


def rule_matmul_const(r, core):
    def f(n):
        if n.op_type != "MatMul" or n.name.startswith("compat_"):
            return None
        w = r.val(n.input[1])
        s = r.shapes.get(n.input[0])
        if w is None or w.ndim != 2 or not s or len(s) != 4:
            return None
        k, m = w.shape
        nb, c, h, wd = s
        if None in (nb, c, h, wd) or wd != k or m > 128:
            return None
        # one depthwise conv per output column, kernel (1, W): col_j[c] = sum_w x[c, :, w] * W[w, j]; concat on W
        # (a Reshape of channels into width is mistranslated by the DFC: -3 dB)
        nodes, cols = [], []
        for j in range(m):
            kern = np.tile(w[:, j].reshape(1, 1, 1, k), (c, 1, 1, 1)).astype(np.float32)
            cv, t = r.node("Conv", [n.input[0], r.c(kern)], kernel_shape=[1, k], group=c)
            nodes.append(cv)
            cols.append(t)
        r.hit("matmul_const")
        return nodes + [helper.make_node("Concat", cols, list(n.output), name=n.name + "/mmc", axis=3)]
    core.rewrite(r, f)


def rule_lrn(r, core):
    def f(n):
        if n.op_type != "LRN":
            return None
        s = r.shapes.get(n.input[0])
        if not s or s[1] is None:
            return None
        c, size = s[1], r.attr(n, "size")
        alpha, beta, bias = r.attr(n, "alpha", 1e-4), r.attr(n, "beta", 0.75), r.attr(n, "bias", 1.0)
        band = np.zeros((c, c, 1, 1), np.float32)
        for i in range(c):
            lo, hi = max(0, i - (size - 1) // 2), min(c - 1, i + int(math.ceil((size - 1) / 2)))
            band[i, lo:hi + 1] = alpha / size
        x = n.input[0]
        sq, ts = r.node("Mul", [x, x])
        cv, tc = r.node("Conv", [ts, r.c(band), r.c(np.full(c, bias, np.float32))], kernel_shape=[1, 1])
        pw, tp = r.node("Pow", [tc, r.c(np.array(beta, np.float32))])
        r.hit("lrn")
        return [sq, cv, pw, helper.make_node("Div", [x, tp], list(n.output), name=n.name + "/lrn")]
    core.rewrite(r, f)


def rule_gelu_any(r, core):
    """Erf(x / sqrt2 | x * 1/sqrt2) followed by Add(1) and Muls by x and 0.5 in any order -> torch's GELU order.
    The candidate output is verified numerically against gelu(x) before the rewrite."""
    prod = {o: n for n in r.g.node for o in n.output}
    cons = {}
    for n in r.g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)

    def scalar(t):
        v = r.val(t)
        return None if v is None or v.size != 1 else float(v.ravel()[0])

    for e in [n for n in r.g.node if n.op_type == "Erf"]:
        d = prod.get(e.input[0])
        if d is None or d.op_type not in ("Div", "Mul"):
            continue
        c = scalar(d.input[1])
        if c is None or not (abs(c - math.sqrt(2)) < 1e-3 if d.op_type == "Div" else abs(c - 1 / math.sqrt(2)) < 1e-3):
            continue
        x = d.input[0]
        chain, cur = [], e.output[0]
        for _ in range(4):
            nx = cons.get(cur, [])
            if len(nx) != 1 or nx[0].op_type not in ("Add", "Mul"):
                break
            chain.append(nx[0])
            cur = nx[0].output[0]
        # candidate outputs: after 3 or 4 nodes (Add, Mul, Mul or with an extra Mul(x, 0.5) branch merged)
        for k in range(len(chain), 1, -1):
            y = chain[k - 1].output[0]
            if _is_gelu(r, x, y):
                chain[k - 1].output[0] = y + "__dead"
                r.g.node.extend(core._torch_gelu(r, x, y, e.name + "_any"))
                r.hit("gelu_any")
                break


def _is_gelu(r, x, y):
    """Numerically: does tensor y equal gelu(x) on a random input? (onnxruntime on a copy with both as outputs)"""
    import onnxruntime as ort
    m = onnx.ModelProto()
    m.CopyFrom(r.m)
    del m.graph.output[:]
    m.graph.output.extend([helper.make_empty_tensor_value_info(x), helper.make_empty_tensor_value_info(y)])
    try:
        s = ort.InferenceSession(m.SerializeToString())
        inp = s.get_inputs()[0]
        shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        a, b = s.run(None, {inp.name: np.random.default_rng(0).normal(0, 1.5, shape).astype(np.float32)})
        from math import erf
        g = 0.5 * a * (1 + np.vectorize(erf)(a / math.sqrt(2)))
        return a.shape == b.shape and float(np.abs(g - b).max()) < 1e-4 * (1 + float(np.abs(g).max()))
    except Exception:  # noqa: BLE001
        return False


def rule_equal(r, core):
    cons = {}
    for n in r.g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)

    def f(n):
        if n.op_type != "Cast" or r.attr(n, "to") != TensorProto.FLOAT:
            return None
        prod = next((m for m in r.g.node if n.input[0] in m.output), None)
        if prod is None or prod.op_type != "Equal" or len(cons.get(prod.output[0], [])) != 1:
            return None
        c = r.val(prod.input[1])
        if c is None or c.size != 1:
            return None
        c = float(c.ravel()[0])
        x = prod.input[0]
        g, tg = r.node("Greater", [x, r.c(np.array(c - 0.5, np.float32))])
        cg, tcg = r.node("Cast", [tg], to=TensorProto.FLOAT)
        l_, tl = r.node("Less", [x, r.c(np.array(c + 0.5, np.float32))])
        cl, tcl = r.node("Cast", [tl], to=TensorProto.FLOAT)
        r.hit("equal")
        return [g, cg, l_, cl, helper.make_node("Mul", [tcg, tcl], list(n.output), name=n.name + "/eq")]
    core.rewrite(r, f)


def rule_group_norm(r, core):
    """GroupNormalization (opset 18/21) -> torch's export: Reshape(N, G, -1) -> InstanceNorm -> Reshape -> affine."""
    def f(n):
        if n.op_type != "GroupNormalization":
            return None
        s = r.shapes.get(n.input[0])
        if not s or len(s) != 4 or any(v is None for v in s):
            return None
        g, c = r.attr(n, "num_groups"), s[1]
        sc, bi = r.val(n.input[1]), r.val(n.input[2])
        if sc is None or bi is None:
            return None
        if sc.size == g:  # opset 18: per-group affine -> per channel
            sc, bi = np.repeat(sc, c // g), np.repeat(bi, c // g)
        x = n.input[0]
        a, t1 = r.node("Reshape", [x, r.c(np.array([s[0], g, -1], np.int64), np.int64)])
        b, t2 = r.node("InstanceNormalization", [t1, r.c(np.ones(g)), r.c(np.zeros(g))], epsilon=r.attr(n, "epsilon", 1e-5))
        c_, t3 = r.node("Reshape", [t2, r.c(np.array(s, np.int64), np.int64)])
        d, t4 = r.node("Mul", [t3, r.c(sc.reshape(1, c, 1, 1).astype(np.float32))])
        r.hit("group_norm")
        return [a, b, c_, d, helper.make_node("Add", [t4, r.c(bi.reshape(1, c, 1, 1).astype(np.float32))], list(n.output),
                                               name=n.name + "/gn")]
    core.rewrite(r, f)
