"""Batch-3 ONNX -> Hailo-8 rewrites: the rest of the default domain (registered into onnx_hailo_compat.ALL).
These run BEFORE constant folding (onnxruntime may lack kernels for the newest ops). Exact in float unless noted.

  sum_mean     Sum / Mean (n-ary) -> Add chain (* 1/n)
  swish        Swish(alpha) (opset 24) -> x * Sigmoid(alpha x)
  cmp_eq       GreaterOrEqual / LessOrEqual -> Greater / Less          (differ only on exact ties)
  bool_float   Cast(bool graph -> float) of Not/And/Or/Xor/comparisons -> 0/1 arithmetic
  nan_inf      IsNaN / IsInf -> all-false (finite tensors); Where(IsNaN(x), a, b) -> b
  gl_lp_pool   GlobalLpPool p=1/2 -> ReduceL1 / ReduceL2 over H, W
  reduce_log_sum  ReduceLogSum -> Log(ReduceSum)
  cumprod      CumProd (opset 26) over H/W -> log-step shift-and-multiply (Slice + Mul + Concat)
  trilu        Trilu -> Mul by a constant 0/1 mask
  rms_norm     RMSNormalization (opset 23) -> torch's pattern Pow2 / ReduceMean / Add / Sqrt / Div / Mul
  attention    Attention (opset 23/24, 4-D or 3-D + heads, optional const mask, is_causal) -> torch's bhnd
               MatMul / Mul / Softmax / MatMul (the layout the DFC parses)
  rotary       RotaryEmbedding (const caches/positions) -> x * cos + rotate_half(x) * sin (Slice/Neg/Concat/Mul)
  gather_el    GatherElements with indices constant except along `axis` -> Gather
  gather_nd    GatherND with const indices [0, c_k] on (1, C, ...) -> Gather over channels
  scatter_blk  ScatterND / ScatterElements / TensorScatter with const indices writing one contiguous block along
               one axis -> Concat(x[:a], update, x[b:])
  onehot       OneHot (depth, values const) -> per-class window masks (x in (k - .5, k + .5)) + Concat
  compress     Compress with a const condition -> Gather
  crop_pad     CenterCropPad (const shape) -> Slice / Pad
  rev_seq      ReverseSequence (const lengths, batch 1) -> reversed single slices + Concat
  col2im       Col2Im -> Reshape + ConvTranspose with one-hot kernels
  maxunpool    MaxUnpool(MaxPool(x)) (kernel = stride) -> nearest upsample * Cast(x >= up)   (ties: all marked)
  deform_zero  DeformConv with const zero offsets (and unit mask) -> Conv
  causal_conv  CausalConvWithState (opset 27, zero/const past) -> left Pad + depthwise Conv (+ SiLU), state = Slice
  lin_attn     LinearAttention 'linear' (no past, T > 1) -> scale * ((Q K^T) * causal mask) V
  space2depth  SpaceToDepth -> Reshape / Transpose / Reshape (ONNX DCR order)
  dft          DFT of a real signal over W -> MatMul with const cos / sin matrices; consumers that reduce or pick
               the complex axis (ReduceSum, ReduceSumSquare, ReduceL2, Gather) act on the two real planes
  stft         STFT (const window/step) of a 1-D signal -> strided Conv with windowed cos / sin kernels (same planes)
  einsum       Einsum (2 operands) -> ReduceSum of private indices, then Mul (no contraction) / MatMul (contraction)
               / 1x1 Conv (one const operand contracting channels)
  qdq          QDQ / QOperator / dynamic-quant models -> float: Q->DQ pairs removed, DQ(const) folded,
               QLinearConv/QLinearMatMul -> Conv/MatMul, MatMulInteger/ConvInteger(DynamicQuantizeLinear) * scales
               -> MatMul/Conv with dequantized weights   (drops the fake-quant rounding: the DFC requantizes)
"""
import math

import numpy as np
from onnx import TensorProto, helper, numpy_helper

RULES = ["add_matmul", "shape_noop", "eq_step", "cmp_zoom", "greater_neg", "prod_tree", "squeeze_unsq", "reshape_chain", "matmul_heads", "qdq", "sum_mean", "swish", "cmp_eq", "nan_inf", "bool_float", "gl_lp_pool", "reduce_log_sum", "cumprod", "trilu",
         "rms_norm", "attention", "rotary", "gather_el", "gather_nd", "scatter_blk", "onehot", "compress", "crop_pad",
         "rev_seq", "col2im", "maxunpool", "deform_zero", "causal_conv", "lin_attn", "space2depth", "dft", "stft",
         "einsum"]
PRE_FOLD = set(RULES)

F = lambda v: np.array(v, np.float32)  # noqa: E731
I64 = lambda v: np.array(v, np.int64)  # noqa: E731


def _a(r, n, name, default=None):
    """Attribute value with strings decoded (onnx returns bytes)."""
    v = r.attr(n, name, default)
    return v.decode() if isinstance(v, bytes) else v


def _cons(r):
    c = {}
    for n in r.g.node:
        for i in n.input:
            c.setdefault(i, []).append(n)
    return c


def _prod(r):
    return {o: n for n in r.g.node for o in n.output}


def _out(n, nodes, last_op, ins, **attrs):
    """Final node writing n's first output."""
    return nodes + [helper.make_node(last_op, ins, [n.output[0]], name=n.name + "/c3", **attrs)]


def _chain(r, op, ts):
    nodes, y = [], ts[0]
    for t in ts[1:]:
        a, y = r.node(op, [y, t])
        nodes.append(a)
    return nodes, y


def _slice(r, x, a, b, axis):
    return r.node("Slice", [x, r.c(I64([a]), np.int64), r.c(I64([b]), np.int64), r.c(I64([axis]), np.int64)])


def _reduce(r, op, x, axes, keepdims=1):
    """Reduce node valid for the model's opset (axes attribute <= 17, input >= 18; ReduceSum input >= 13)."""
    opset = next(o.version for o in r.m.opset_import if o.domain in ("", "ai.onnx"))
    if op == "ReduceSum" or opset >= 18:
        return r.node(op, [x, r.c(I64(axes), np.int64)], keepdims=keepdims)
    return r.node(op, [x], axes=list(axes), keepdims=keepdims)


# ---------------------------------------------------------------- element-wise / logic
def rule_add_matmul(r, core):
    """Add(MatMul(x, A), MatMul(x, B)) with constant A, B -> MatMul(x, A + B) (DFT -> ReduceSum; also avoids the
    parser merging identical conv columns: cos is symmetric)."""
    prod = _prod(r)

    def f(n):
        if n.op_type != "Add":
            return None
        a, b = (prod.get(i) for i in n.input)
        if a is None or b is None or a.op_type != "MatMul" or b.op_type != "MatMul" or a.input[0] != b.input[0]:
            return None
        wa, wb = r.val(a.input[1]), r.val(b.input[1])
        if wa is None or wb is None or wa.shape != wb.shape:
            return None
        r.hit("add_matmul")
        return [helper.make_node("MatMul", [a.input[0], r.c((wa + wb).astype(np.float32))], list(n.output),
                                 name=n.name + "/am")]
    core.rewrite(r, f)


def rule_shape_noop(r, core):
    """A chain of Reshape / Unsqueeze / Squeeze / Flatten whose output shape equals its input's -> Identity."""
    prod, cons = _prod(r), _cons(r)
    kinds = ("Reshape", "Unsqueeze", "Squeeze", "Flatten")

    def f(n):
        if n.op_type not in kinds:
            return None
        src, p = n.input[0], prod.get(n.input[0])
        while p is not None and p.op_type in kinds and len(cons.get(p.output[0], [])) == 1:
            src, p = p.input[0], prod.get(p.input[0])
        a, b = r.shapes.get(src), r.shapes.get(n.output[0])
        if src == n.input[0] and a != b:
            return None
        if not a or not b or None in a or a != b:
            return None
        r.hit("shape_noop")
        return [helper.make_node("Identity", [src], list(n.output), name=n.name + "/c3")]
    core.rewrite(r, f)


def rule_eq_step(r, core):
    """Equal(Round|Floor|Ceil(x), c) -> And(x >= lo, x < hi) on x itself (no step function on the chip):
    Round [c-.5, c+.5), Floor [c, c+1), Ceil (c-1, c]."""
    prod = _prod(r)

    def f(n):
        if n.op_type != "Equal":
            return None
        p = prod.get(n.input[0])
        c = r.val(n.input[1])
        if p is None or p.op_type not in ("Round", "Floor", "Ceil") or c is None or c.size != 1:
            return None
        c = float(np.asarray(c).ravel()[0])
        lo, hi = {"Round": (c - .5, c + .5), "Floor": (c, c + 1), "Ceil": (c - 1, c)}[p.op_type]
        x = p.input[0]
        g, tg = r.node("Greater", [x, r.c(F(lo))])
        l_, tl = r.node("Less", [x, r.c(F(hi))])
        r.hit("eq_step")
        return [g, l_, helper.make_node("And", [tg, tl], list(n.output), name=n.name + "/c3")]
    core.rewrite(r, f)


def rule_cmp_zoom(r, core):
    """Greater/Less(x, t) -> Greater/Less(Clip(K (x - t), -1, 1), 0)  (CMP_ZOOM=K, for 16-bit builds).
    The chip's compare activations are 8-bit only: a 16-bit x is requantized to 8 bit over its whole range and
    every value within half a step of t flips (a 0/1 error of size |x|). The zoom layer (16-bit capable) puts the
    8-bit grid of the compare input around the threshold: resolution 2 / (255 K) in x units."""
    import os
    K = float(os.environ.get("CMP_ZOOM", "0"))
    if not K:
        return

    def f(n):
        if n.op_type not in ("Greater", "Less") or n.name.endswith("/z"):
            return None
        t = r.val(n.input[1])
        if t is None or t.size != 1:
            return None
        sb, ts = r.node("Sub", [n.input[0], r.c(F(float(np.asarray(t).ravel()[0])))])
        mu, tm = r.node("Mul", [ts, r.c(F(K))])
        cl, tc = r.node("Clip", [tm, r.c(F(-1.0)), r.c(F(1.0))])
        r.hit("cmp_zoom")
        return [sb, mu, cl, helper.make_node(n.op_type, [tc, r.c(F(0.0))], list(n.output), name=n.name + "/z")]
    core.rewrite(r, f)


def rule_greater_neg(r, core):
    """Greater(x, t < 0) -> Not(Less(x, t)): the chip's `greater` is a thresholded ReLU (theta must be >= 0; a negative
    theta fails to quantize alone and computes garbage inside a graph: And/Xor 13 dB). Less works for any t."""
    def f(n):
        if n.op_type != "Greater" or n.name.endswith("/gn"):
            return None
        t = r.val(n.input[1])
        if t is None or t.size != 1 or float(np.asarray(t).ravel()[0]) >= 0:
            return None
        ls, tl = r.node("Less", [n.input[0], n.input[1]])
        r.hit("greater_neg")
        return [ls, helper.make_node("Not", [tl], list(n.output), name=n.name + "/gn")]
    core.rewrite(r, f)


def rule_sum_mean(r, core):
    def f(n):
        if n.op_type not in ("Sum", "Mean") or n.name.endswith("/c3"):
            return None
        nodes, y = _chain(r, "Add", list(n.input))
        r.hit("sum_mean")
        if n.op_type == "Sum":
            return _out(n, nodes, "Identity", [y])
        return _out(n, nodes, "Mul", [y, r.c(F(1.0 / len(n.input)))])
    core.rewrite(r, f)


def rule_swish(r, core):
    def f(n):
        if n.op_type != "Swish":
            return None
        a = _a(r, n, "alpha", 1.0)
        x = n.input[0]
        nodes = []
        if a != 1.0:
            m, x2 = r.node("Mul", [x, r.c(F(a))])
            nodes.append(m)
        else:
            x2 = x
        s, ts = r.node("Sigmoid", [x2])
        r.hit("swish")
        return _out(n, nodes + [s], "Mul", [x, ts])
    core.rewrite(r, f)


def rule_cmp_eq(r, core):
    def f(n):
        if n.op_type not in ("GreaterOrEqual", "LessOrEqual"):
            return None
        r.hit("cmp_eq")
        return [helper.make_node(n.op_type.replace("OrEqual", ""), list(n.input), list(n.output), name=n.name + "/c3")]
    core.rewrite(r, f)


def rule_nan_inf(r, core):
    prod = _prod(r)

    def f(n):
        if n.op_type == "Where" and prod.get(n.input[0]) is not None and prod[n.input[0]].op_type in ("IsNaN", "IsInf"):
            r.hit("nan_inf")
            return [helper.make_node("Identity", [n.input[2]], list(n.output), name=n.name + "/c3")]
        if n.op_type in ("IsNaN", "IsInf"):  # all false on the finite tensors a quantized chip computes
            r.hit("nan_inf")
            return [helper.make_node("Less", [n.input[0], r.c(F(-3.0e38))], list(n.output), name=n.name + "/c3")]
        return None
    core.rewrite(r, f)


_CMP = ("Greater", "Less", "GreaterOrEqual", "LessOrEqual", "Equal", "IsNaN", "IsInf")


def bool_linear(r, op, m1, m2, out):
    """0/1 masks without a mask x mask product: And = Relu(m1 + m2 - 1), Or = min(m1 + m2, 1) = 1 - Relu(1 - m1 - m2),
    Xor = |m1 - m2|. (Product of two compare masks quantizes to 13 dB on the chip at 8 and 16 bit.)"""
    if op == "Xor":
        a, d = r.node("Sub", [m1, m2])
        b, y = r.node("Abs", [d])
        out += [a, b]
        return y
    a, s_ = r.node("Add", [m1, m2])
    out.append(a)
    if op == "And":
        b, t = r.node("Sub", [s_, r.c(F(1.0))])
        c, y = r.node("Relu", [t])
        out += [b, c]
        return y
    b, t = r.node("Sub", [r.c(F(1.0)), s_])
    c, t2 = r.node("Relu", [t])
    d, y = r.node("Sub", [r.c(F(1.0)), t2])
    out += [b, c, d]
    return y


def rule_bool_float(r, core):
    """Cast(boolean graph, to=float) -> 0/1 arithmetic (Not = 1 - m, And = m1 m2, Or = m1 + m2 - m1 m2,
    Xor = m1 + m2 - 2 m1 m2). Comparisons stay Cast(cmp) (the parser's mask)."""
    prod = _prod(r)
    memo = {}

    def mask(t, out):
        if t in memo:
            return memo[t]
        n = prod.get(t)
        if n is None:
            raise ValueError
        if n.op_type in _CMP:
            a, m = r.node("Cast", [t], to=TensorProto.FLOAT)
            out.append(a)
        elif n.op_type == "Not":
            a, m = r.node("Sub", [r.c(F(1.0)), mask(n.input[0], out)])
            out.append(a)
        elif n.op_type in ("And", "Or", "Xor"):
            m = bool_linear(r, n.op_type, mask(n.input[0], out), mask(n.input[1], out), out)
        elif n.op_type == "Cast":
            m = mask(n.input[0], out)
        else:
            raise ValueError
        memo[t] = m
        return m

    def f(n):
        if n.op_type != "Cast" or _a(r, n, "to") != TensorProto.FLOAT:
            return None
        p = prod.get(n.input[0])
        if p is None or p.op_type not in ("Not", "And", "Or", "Xor"):
            return None
        out = []
        try:
            m = mask(n.input[0], out)
        except ValueError:
            return None
        r.hit("bool_float")
        return _out(n, out, "Identity", [m])
    core.rewrite(r, f)


# ---------------------------------------------------------------- reductions / pooling
def rule_gl_lp_pool(r, core):
    def f(n):
        if n.op_type != "GlobalLpPool":
            return None
        p = _a(r, n, "p", 2)
        if p not in (1, 2):
            return None
        a, t = _reduce(r, "ReduceL2" if p == 2 else "ReduceL1", n.input[0], [2, 3])
        r.hit("gl_lp_pool")
        return _out(n, [a], "Identity", [t])
    core.rewrite(r, f)


def rule_reduce_log_sum(r, core):
    def f(n):
        if n.op_type != "ReduceLogSum":
            return None
        ax = r.axes_of(n)
        if ax is None:
            return None
        a, t = _reduce(r, "ReduceSum", n.input[0], ax, _a(r, n, "keepdims", 1))
        r.hit("reduce_log_sum")
        return _out(n, [a], "Log", [t])
    core.rewrite(r, f)


def rule_cumprod(r, core):
    def f(n):
        if n.op_type != "CumProd" or _a(r, n, "exclusive", 0) or _a(r, n, "reverse", 0):
            return None
        ax, s = r.val(n.input[1]), r.shapes.get(n.input[0])
        if ax is None or not s or len(s) != 4:
            return None
        axis = int(np.asarray(ax).ravel()[0]) % 4
        L = s[axis]
        if axis not in (2, 3) or L is None:
            return None
        nodes, y, k = [], n.input[0], 1
        while k < L:  # Hillis-Steele: y[k:] *= y[:-k]
            a, head = _slice(r, y, 0, k, axis)
            b, tail = _slice(r, y, k, L, axis)
            c, prev = _slice(r, y, 0, L - k, axis)
            d, mul = r.node("Mul", [tail, prev])
            e, y = r.node("Concat", [head, mul], axis=axis)
            nodes += [a, b, c, d, e]
            k *= 2
        r.hit("cumprod")
        return _out(n, nodes, "Identity", [y])
    core.rewrite(r, f)


def rule_prod_tree(r, core):
    """ReduceProd over one axis -> pairwise Mul tree of Slices (exact for any sign; exp(sum log) needs x > 0)."""
    def f(n):
        if n.op_type != "ReduceProd" or not _a(r, n, "keepdims", 1):
            return None
        ax, s_ = r.axes_of(n), r.shapes.get(n.input[0])
        if not ax or len(ax) != 1 or not s_ or s_[ax[0] % len(s_)] is None:
            return None
        axis = ax[0] % len(s_)
        parts, nodes = [], []
        for i in range(s_[axis]):
            sl, t = _slice(r, n.input[0], i, i + 1, axis)
            nodes.append(sl)
            parts.append(t)
        while len(parts) > 1:
            nxt = []
            for i in range(0, len(parts) - 1, 2):
                mu, t = r.node("Mul", [parts[i], parts[i + 1]])
                nodes.append(mu)
                nxt.append(t)
            parts = nxt + (parts[-1:] if len(parts) % 2 else [])
        r.hit("prod_tree")
        return _out(n, nodes, "Identity", parts)
    core.rewrite(r, f)


def rule_squeeze_unsq(r, core):
    """Squeeze(Unsqueeze(x, a), a) -> x (left by rewrites of rank-5 complex tensors)."""
    prod = _prod(r)

    def axes(n):
        v = r.val(n.input[1]) if len(n.input) > 1 else None
        return None if v is None else [int(a) for a in np.asarray(v).ravel()]

    def f(n):
        if n.op_type != "Squeeze":
            return None
        u = prod.get(n.input[0])
        if u is None or u.op_type != "Unsqueeze" or axes(n) is None or axes(n) != axes(u):
            return None
        r.hit("squeeze_unsq")
        return [helper.make_node("Identity", [u.input[0]], list(n.output), name=n.name + "/c3")]
    core.rewrite(r, f)


def rule_reshape_chain(r, core):
    """Reshape(Reshape(x, s1), s2) -> Reshape(x, s2) when the inner result has no other consumer (rank-3
    intermediates confuse the parser's channel mapping: CausalConvWithState's (B, C, L))."""
    prod, cons = _prod(r), _cons(r)

    def f(n):
        if n.op_type != "Reshape":
            return None
        p = prod.get(n.input[0])
        if p is None or p.op_type != "Reshape" or len(cons.get(p.output[0], [])) != 1:
            return None
        sh = r.val(n.input[1])
        if sh is None or (np.asarray(sh) <= 0).any():
            return None
        r.hit("reshape_chain")
        return [helper.make_node("Reshape", [p.input[0], n.input[1]], list(n.output), name=n.name + "/c3")]
    core.rewrite(r, f)


def rule_matmul_heads(r, core):
    """MatMul (1, h, M, K) x (1, h, K, N) of two tensors: the DFC maps the h axis to conv groups and needs M, K, N
    divisible by h (per-head split + Concat and zero-padding do not parse either): per output column a broadcast Mul
    and a ReduceSum over W, Concat over W (N <= 64)."""
    def f(n):
        if n.op_type != "MatMul" or r.val(n.input[1]) is not None or n.name.endswith("/mh"):
            return None
        sa, sb = r.shapes.get(n.input[0]), r.shapes.get(n.input[1])
        if not sa or not sb or len(sa) != 4 or len(sb) != 4 or None in sa + sb or sa[0] != 1:
            return None
        h = sa[1]
        if h == 1 or (sa[2] % h == 0 and sa[3] % h == 0 and sb[3] % h == 0):
            return None
        M, K, Nn = sa[2], sa[3], sb[3]
        if Nn > 64:
            return None
        # out[:, :, :, j] = sum_k A[:, :, :, k] * B[:, :, k, j]: B column -> row (H<->W), broadcast Mul, ReduceSum W
        nodes, cols = [], []
        for j in range(Nn):
            s1, tb = _slice(r, n.input[1], j, j + 1, 3)  # (1, h, K, 1)
            t1, tbt = r.node("Transpose", [tb], perm=[0, 1, 3, 2])  # (1, h, 1, K)
            mu, tm = r.node("Mul", [n.input[0], tbt])  # (1, h, M, K)
            rs, tc = r.node("ReduceSum", [tm, r.c(I64([3]), np.int64)], keepdims=1)  # (1, h, M, 1)
            nodes += [s1, t1, mu, rs]
            cols.append(tc)
        r.hit("matmul_heads")
        return _out(n, nodes, "Concat", cols, axis=3)
    core.rewrite(r, f)


def rule_trilu(r, core):
    def f(n):
        if n.op_type != "Trilu":
            return None
        s = r.shapes.get(n.input[0])
        if not s or s[-1] is None or s[-2] is None:
            return None
        k = 0
        if len(n.input) > 1 and n.input[1]:
            kv = r.val(n.input[1])
            if kv is None:
                return None
            k = int(np.asarray(kv).ravel()[0])
        ones = np.ones((s[-2], s[-1]), np.float32)
        mask = np.triu(ones, k) if _a(r, n, "upper", 1) else np.tril(ones, k)
        r.hit("trilu")
        return _out(n, [], "Mul", [n.input[0], r.c(mask.reshape([1] * (len(s) - 2) + list(mask.shape)))])
    core.rewrite(r, f)


def rule_rms_norm(r, core):
    def f(n):
        if n.op_type != "RMSNormalization":
            return None
        rk = r.rank(n.input[0])
        if rk is None:
            return None
        ax = _a(r, n, "axis", -1) % rk
        eps = _a(r, n, "epsilon", 1e-5)
        x = n.input[0]
        p, t2 = r.node("Pow", [x, r.c(F(2.0))])
        m, tm = _reduce(r, "ReduceMean", t2, list(range(ax, rk)))
        a, ta = r.node("Add", [tm, r.c(F(eps))])
        q, tq = r.node("Sqrt", [ta])
        d, td = r.node("Div", [r.c(F(1.0)), tq])
        mu, ty = r.node("Mul", [x, td])
        r.hit("rms_norm")
        return _out(n, [p, m, a, q, d, mu], "Mul", [ty, n.input[1]])
    core.rewrite(r, f)


# ---------------------------------------------------------------- attention family
def _heads4(r, t, s, h, nodes):
    """(B, S, h*D) -> (B, h, S, D)"""
    b, S, hd = s
    a, t1 = r.node("Reshape", [t, r.c(I64([b, S, h, hd // h]), np.int64)])
    c, t2 = r.node("Transpose", [t1], perm=[0, 2, 1, 3])
    nodes += [a, c]
    return t2


def rule_attention(r, core):
    def f(n):
        if n.op_type != "Attention" or any(i for i in n.input[4:]) or len([o for o in n.output if o]) > 1:
            return None
        q, k, v = n.input[:3]
        sq, sk, sv = (r.shapes.get(t) for t in (q, k, v))
        if not sq or not sk or not sv or None in sq + sk + sv:
            return None
        nodes = []
        three = len(sq) == 3
        if three:
            hq, hk = _a(r, n, "q_num_heads"), _a(r, n, "kv_num_heads")
            if not hq or hq != hk:
                return None
            sv4 = [sv[0], hk, sv[1], sv[2] // hk]
            q, k, v = _heads4(r, q, sq, hq, nodes), _heads4(r, k, sk, hk, nodes), _heads4(r, v, sv, hk, nodes)
            D = sq[2] // hq
            Sq, Sk = sq[1], sk[1]
        else:
            if sq[1] != sk[1]:
                return None  # GQA: not yet
            D, Sq, Sk = sq[3], sq[2], sk[2]
            sv4 = sv
        if _a(r, n, "softcap", 0.0):
            return None
        scale = _a(r, n, "scale", 0.0) or 1.0 / math.sqrt(D)
        tk, tkt = r.node("Transpose", [k], perm=[0, 1, 3, 2])
        mm, ts = r.node("MatMul", [q, tkt])
        mu, tl = r.node("Mul", [ts, r.c(F(scale))])
        nodes += [tk, mm, mu]
        bias = None
        if len(n.input) > 3 and n.input[3]:
            mv = r.val(n.input[3])
            if mv is None:
                return None
            bias = np.where(mv, 0.0, -np.inf) if mv.dtype == bool else mv.astype(np.float64)
        if _a(r, n, "is_causal", 0):
            cm = np.triu(np.full((Sq, Sk), -np.inf), 1 + Sk - Sq)
            bias = cm if bias is None else bias + cm
        if bias is not None:  # additive, tiled over heads (the parser maps (1, h, N, N) consts), -inf -> -30
            hh = sv4[1]
            b30 = np.maximum(np.asarray(bias, np.float64), -30.0)
            eb = np.broadcast_to(b30, (1, hh, Sq, Sk)).astype(np.float32)
            ad, tl = r.node("Add", [tl, r.c(np.ascontiguousarray(eb))])
            nodes.append(ad)
        sm, ta = r.node("Softmax", [tl], axis=-1)
        nodes.append(sm)
        if three:
            m2, to = r.node("MatMul", [ta, v])
            tr, t2 = r.node("Transpose", [to], perm=[0, 2, 1, 3])
            nodes += [m2, tr]
            r.hit("attention")
            return _out(n, nodes, "Reshape", [t2, r.c(I64([sq[0], Sq, sv4[1] * sv4[3]]), np.int64)])
        r.hit("attention")
        return _out(n, nodes, "MatMul", [ta, v])
    core.rewrite(r, f)


def _rot_half(r, x, D, inter, nodes):
    """rotate_half over the last axis: non-interleaved [-x2, x1]; interleaved pairs (-x_{2i+1}, x_{2i})."""
    ax = 3
    if not inter:
        a, x1 = _slice(r, x, 0, D // 2, ax)
        b, x2 = _slice(r, x, D // 2, D, ax)
        c, nx2 = r.node("Neg", [x2])
        d, y = r.node("Concat", [nx2, x1], axis=ax)
        nodes += [a, b, c, d]
        return y
    parts = []
    for i in range(D // 2):
        a, e = _slice(r, x, 2 * i, 2 * i + 1, ax)
        b, o = _slice(r, x, 2 * i + 1, 2 * i + 2, ax)
        c, no = r.node("Neg", [o])
        nodes += [a, b, c]
        parts += [no, e]
    d, y = r.node("Concat", parts, axis=ax)
    nodes.append(d)
    return y


def rule_rotary(r, core):
    def f(n):
        if n.op_type != "RotaryEmbedding":
            return None
        x = n.input[0]
        s = r.shapes.get(x)
        cos, sin = r.val(n.input[1]), r.val(n.input[2])
        pos = r.val(n.input[3]) if len(n.input) > 3 and n.input[3] else None
        if not s or len(s) != 4 or cos is None or sin is None or None in s:
            return None
        if len(n.input) > 3 and n.input[3] and pos is None:
            return None
        B, Hh, S, D = s
        rd = _a(r, n, "rotary_embedding_dim", 0) or D
        inter = _a(r, n, "interleaved", 0)
        if pos is not None:
            cos, sin = cos[pos], sin[pos]  # (B, S, rd/2)
        else:
            cos, sin = cos[:S][None], sin[:S][None]
        rep = (lambda c: np.repeat(c, 2, -1)) if inter else (lambda c: np.concatenate([c, c], -1))
        cf, sf = rep(cos)[:, None].astype(np.float32), rep(sin)[:, None].astype(np.float32)  # (B, 1, S, rd)
        nodes = []
        xr = x
        if rd < D:
            a, xr = _slice(r, x, 0, rd, 3)
            b, xp = _slice(r, x, rd, D, 3)
            nodes += [a, b]
        m1, t1 = r.node("Mul", [xr, r.c(cf)])
        rh = _rot_half(r, xr, rd, inter, nodes)
        m2, t2 = r.node("Mul", [rh, r.c(sf)])
        nodes += [m1, m2]
        r.hit("rotary")
        if rd < D:
            ad, ty = r.node("Add", [t1, t2])
            return _out(n, nodes + [ad], "Concat", [ty, xp], axis=3)
        return _out(n, nodes, "Add", [t1, t2])
    core.rewrite(r, f)


def rule_lin_attn(r, core):
    def f(n):
        if n.op_type != "LinearAttention" or (_a(r, n, "update_rule", "gated_delta") not in ("linear",)):
            return None
        if any(i for i in n.input[3:]) or (len(n.output) > 1 and n.output[1] and _cons(r).get(n.output[1])):
            return None
        q, k, v = n.input[:3]
        sq, sk, sv = (r.shapes.get(t) for t in (q, k, v))
        hq, hk = _a(r, n, "q_num_heads"), _a(r, n, "kv_num_heads")
        if not sq or None in sq + sk + sv or hq != hk:
            return None
        B, T, _ = sq
        nodes = []
        q4, k4, v4 = _heads4(r, q, sq, hq, nodes), _heads4(r, k, sk, hk, nodes), _heads4(r, v, sv, hk, nodes)
        dk = sq[2] // hq
        scale = _a(r, n, "scale", 0.0) or 1.0 / math.sqrt(dk)
        tk, tkt = r.node("Transpose", [k4], perm=[0, 1, 3, 2])
        mm, ts = r.node("MatMul", [q4, tkt])
        mask = np.broadcast_to(np.tril(np.ones((T, T))) * scale, (1, hq, T, T)).astype(np.float32)  # tiled: parses
        nq, q4n = r.node("Neg", [q4])  # Relu(-S) = Relu((-Q) K^T): no Neg between matmuls (the parser fuses it
        mmn, tsn = r.node("MatMul", [q4n, tkt])  # into a signed left operand, which the Hailo matmul rejects)
        rp, tp0 = r.node("Relu", [ts])
        rn, tq0 = r.node("Relu", [tsn])
        mu, tp = r.node("Mul", [tp0, r.c(np.ascontiguousarray(mask))])  # Relu(S) M = Relu(S M), M >= 0
        mu2, tq = r.node("Mul", [tq0, r.c(np.ascontiguousarray(mask))])
        m2, to1 = r.node("MatMul", [tp, v4])  # Hailo matmul: the left operand must be >= 0 (softmax-like)
        m3, to2 = r.node("MatMul", [tq, v4])
        sb, to = r.node("Sub", [to1, to2])
        tr, t2 = r.node("Transpose", [to], perm=[0, 2, 1, 3])
        r.hit("lin_attn")
        return _out(n, nodes + [tk, mm, nq, mmn, rp, rn, mu, mu2, m2, m3, sb, tr], "Reshape",
                    [t2, r.c(I64([B, T, sv[2]]), np.int64)])
    core.rewrite(r, f)


# ---------------------------------------------------------------- indexing
def rule_gather_el(r, core):
    def f(n):
        if n.op_type != "GatherElements":
            return None
        idx = r.val(n.input[1])
        s = r.shapes.get(n.input[0])
        if idx is None or not s:
            return None
        ax = _a(r, n, "axis", 0) % idx.ndim
        moved = np.moveaxis(idx, ax, 0).reshape(idx.shape[ax], -1)
        if not (moved == moved[:, :1]).all() or any(idx.shape[i] != s[i] for i in range(idx.ndim) if i != ax):
            return None
        sel = np.where(moved[:, 0] < 0, moved[:, 0] + s[ax], moved[:, 0])
        r.hit("gather_el")
        return _out(n, [], "Gather", [n.input[0], r.c(I64(sel), np.int64)], axis=ax)
    core.rewrite(r, f)


def rule_gather_nd(r, core):
    def f(n):
        if n.op_type != "GatherND" or _a(r, n, "batch_dims", 0):
            return None
        idx = r.val(n.input[1])
        s = r.shapes.get(n.input[0])
        if idx is None or not s or idx.shape[-1] != 2 or s[0] != 1 or idx.ndim != 3 or idx.shape[0] != 1:
            return None
        if (idx[..., 0] != 0).any():
            return None
        r.hit("gather_nd")
        return _out(n, [], "Gather", [n.input[0], r.c(I64(idx[0, :, 1]), np.int64)], axis=1)
    core.rewrite(r, f)


def _block(r, x, upd, axis, a, L):
    s = r.shapes.get(upd)
    if not s or s[axis] is None:
        return None
    b = a + s[axis]
    nodes, parts = [], []
    if a > 0:
        n1, t = _slice(r, x, 0, a, axis)
        nodes.append(n1)
        parts.append(t)
    parts.append(upd)
    if b < L:
        n2, t = _slice(r, x, b, L, axis)
        nodes.append(n2)
        parts.append(t)
    return nodes, parts


def rule_scatter_blk(r, core):
    def f(n):
        if n.op_type not in ("ScatterND", "ScatterElements", "TensorScatter") or _a(r, n, "reduction", "none") != "none":
            return None
        x, upd = n.input[0], n.input[2] if n.op_type != "TensorScatter" else n.input[1]
        s, su = r.shapes.get(x), r.shapes.get(upd)
        if not s or not su or None in s + su:
            return None
        if n.op_type == "TensorScatter":
            if _a(r, n, "mode", "linear") != "linear":
                return None
            axis = _a(r, n, "axis", -2) % len(s)
            wi = r.val(n.input[2]) if len(n.input) > 2 and n.input[2] else np.zeros(1, np.int64)
            if wi is None or s[0] != 1:
                return None
            a = int(np.asarray(wi).ravel()[0])
        elif n.op_type == "ScatterElements":
            idx = r.val(n.input[1])
            if idx is None:
                return None
            axis = _a(r, n, "axis", 0) % len(s)
            mv = np.moveaxis(idx, axis, 0).reshape(idx.shape[axis], -1)
            a = int(mv[0, 0])
            if not (mv == mv[:, :1]).all() or not (mv[:, 0] == np.arange(a, a + len(mv))).all() or \
                    any(su[i] != s[i] for i in range(len(s)) if i != axis):
                return None
        else:  # ScatterND: indices (K, 1) over axis 0 ... (1, K, 2) over axis 1 with batch 0
            idx = r.val(n.input[1])
            if idx is None:
                return None
            k = idx.shape[-1]
            flat = idx.reshape(-1, k)
            if (flat[:, :k - 1] != flat[:1, :k - 1]).any():
                return None
            axis = k - 1
            a = int(flat[0, -1])
            if not (flat[:, -1] == np.arange(a, a + len(flat))).all() or (k > 1 and (flat[0, :k - 1] != 0).any()):
                return None
            if k > 1 and s[0] != 1:
                return None
            want = list(s)
            want[axis] = len(flat)
            if list(su) != (list(idx.shape[:-1]) + list(s[k:])) or int(np.prod(su)) != int(np.prod(want)):
                return None
            if list(su) != want:
                rs, upd = r.node("Reshape", [upd, r.c(I64(want), np.int64)])
                pre = [rs]
            else:
                pre = []
            got = _block(r, x, upd, axis, a, s[axis])
            if got is None:
                return None
            r.hit("scatter_blk")
            return _out(n, pre + got[0], "Concat", got[1], axis=axis)
        got = _block(r, x, upd, axis, a, s[axis])
        if got is None:
            return None
        r.hit("scatter_blk")
        return _out(n, got[0], "Concat", got[1], axis=axis)
    core.rewrite(r, f)


def rule_onehot(r, core):
    def f(n):
        if n.op_type != "OneHot":
            return None
        depth, vals = r.val(n.input[1]), r.val(n.input[2])
        s = r.shapes.get(n.input[0])
        if depth is None or vals is None or not s:
            return None
        depth = int(np.asarray(depth).ravel()[0])
        axis = _a(r, n, "axis", -1) % (len(s) + 1)
        off, on = float(vals[0]), float(vals[1])
        nodes = []
        x = n.input[0]
        cx, xf = r.node("Cast", [x], to=TensorProto.FLOAT)
        un, xu = r.node("Unsqueeze", [xf, r.c(I64([axis]), np.int64)])
        nodes += [cx, un]
        parts = []
        for k in range(depth):  # window (k - .5, k + .5): Less(x, k+.5) - Less(x, k-.5) (no mask product, theta >= 0)
            l_, tl = r.node("Less", [xu, r.c(F(k + 0.5))])
            cl, tcl = r.node("Cast", [tl], to=TensorProto.FLOAT)
            nodes += [l_, cl]
            if k == 0:  # indices are >= 0
                parts.append(tcl)
                continue
            l2, tl2 = r.node("Less", [xu, r.c(F(k - 0.5))])
            c2, tc2 = r.node("Cast", [tl2], to=TensorProto.FLOAT)
            sb, tm = r.node("Sub", [tcl, tc2])
            nodes += [l2, c2, sb]
            parts.append(tm)
        cc, tc = r.node("Concat", parts, axis=axis)
        nodes.append(cc)
        r.hit("onehot")
        if off == 0.0 and on == 1.0:
            return _out(n, nodes, "Identity", [tc])
        mu, tm = r.node("Mul", [tc, r.c(F(on - off))])
        return _out(n, nodes + [mu], "Add", [tm, r.c(F(off))])
    core.rewrite(r, f)


def rule_compress(r, core):
    def f(n):
        if n.op_type != "Compress" or _a(r, n, "axis") is None:
            return None
        cond = r.val(n.input[1])
        if cond is None:
            return None
        r.hit("compress")
        return _out(n, [], "Gather", [n.input[0], r.c(I64(np.nonzero(cond)[0]), np.int64)], axis=_a(r, n, "axis"))
    core.rewrite(r, f)


def rule_crop_pad(r, core):
    def f(n):
        if n.op_type != "CenterCropPad":
            return None
        shp = r.val(n.input[1])
        s = r.shapes.get(n.input[0])
        if shp is None or not s or None in s:
            return None
        axes = _a(r, n, "axes") or list(range(len(s)))
        axes = [a % len(s) for a in axes]
        nodes, y = [], n.input[0]
        pads = [0] * (2 * len(s))
        for a, want in zip(axes, shp.tolist()):
            have = s[a]
            if want < have:  # crop: start = (have - want) // 2
                st = (have - want) // 2
                sl, y = _slice(r, y, st, st + want, a)
                nodes.append(sl)
            elif want > have:  # pad: before = (want - have) // 2
                pads[a] = (want - have) // 2
                pads[a + len(s)] = want - have - pads[a]
        r.hit("crop_pad")
        if any(pads):
            return _out(n, nodes, "Pad", [y, r.c(I64(pads), np.int64)], mode="constant")
        return _out(n, nodes, "Identity", [y])
    core.rewrite(r, f)


def rule_rev_seq(r, core):
    def f(n):
        if n.op_type != "ReverseSequence":
            return None
        lens = r.val(n.input[1])
        s = r.shapes.get(n.input[0])
        ta, ba = _a(r, n, "time_axis", 0), _a(r, n, "batch_axis", 1)
        if lens is None or not s or s[ba] != 1 or s[ta] is None:
            return None
        L, T = int(lens.ravel()[0]), s[ta]
        nodes, parts = [], []
        for i in range(L - 1, -1, -1):
            sl, t = _slice(r, n.input[0], i, i + 1, ta)
            nodes.append(sl)
            parts.append(t)
        if L < T:
            sl, t = _slice(r, n.input[0], L, T, ta)
            nodes.append(sl)
            parts.append(t)
        r.hit("rev_seq")
        return _out(n, nodes, "Concat", parts, axis=ta)
    core.rewrite(r, f)


# ---------------------------------------------------------------- conv / pooling family
def rule_col2im(r, core):
    def f(n):
        if n.op_type != "Col2Im":
            return None
        img, blk = r.val(n.input[1]), r.val(n.input[2])
        s = r.shapes.get(n.input[0])
        if img is None or blk is None or not s or None in s or len(blk) != 2:
            return None
        kh, kw = (int(v) for v in blk)
        st = _a(r, n, "strides", [1, 1])
        dl = _a(r, n, "dilations", [1, 1])
        pd = _a(r, n, "pads", [0, 0, 0, 0])
        N, CK, L = s
        C = CK // (kh * kw)
        Lh = (int(img[0]) + pd[0] + pd[2] - (dl[0] * (kh - 1) + 1)) // st[0] + 1
        Lw = (int(img[1]) + pd[1] + pd[3] - (dl[1] * (kw - 1) + 1)) // st[1] + 1
        if Lh * Lw != L:
            return None
        w = np.zeros((CK, C, kh, kw), np.float32)
        for c in range(C):
            for i in range(kh):
                for j in range(kw):
                    w[c * kh * kw + i * kw + j, c, i, j] = 1.0
        a, t = r.node("Reshape", [n.input[0], r.c(I64([N, CK, Lh, Lw]), np.int64)])
        r.hit("col2im")
        return _out(n, [a], "ConvTranspose", [t, r.c(w)], kernel_shape=[kh, kw], strides=list(st),
                    dilations=list(dl), pads=list(pd))
    core.rewrite(r, f)


def rule_maxunpool(r, core):
    prod = _prod(r)

    def f(n):
        if n.op_type != "MaxUnpool":
            return None
        mp = prod.get(n.input[0])
        if mp is None or mp.op_type != "MaxPool" or mp.output[1:2] != [n.input[1]]:
            return None
        k, st = _a(r, mp, "kernel_shape"), _a(r, mp, "strides", [1, 1])
        if list(k) != list(st) or any(_a(r, mp, "pads", [0, 0, 0, 0])) or list(_a(r, n, "kernel_shape")) != list(k):
            return None
        x = mp.input[0]
        up, tu = r.node("Resize", [n.input[0], "", r.c(F([1, 1, k[0], k[1]]))], mode="nearest")
        sb, ts = r.node("Sub", [x, tu])  # <= 0, == 0 at the max
        ls, tl = r.node("Less", [ts, r.c(F(-1e-6))])  # threshold rounds to 0 on the chip: still "< 0"
        cs, tc = r.node("Cast", [tl], to=TensorProto.FLOAT)
        nt, tm = r.node("Sub", [r.c(F(1.0)), tc])
        r.hit("maxunpool")
        return _out(n, [up, sb, ls, cs, nt], "Mul", [tu, tm])
    core.rewrite(r, f)


def rule_deform_zero(r, core):
    def f(n):
        if n.op_type != "DeformConv":
            return None
        off = r.val(n.input[2])
        mask = r.val(n.input[4]) if len(n.input) > 4 and n.input[4] else None
        if off is None or np.abs(off).max() != 0 or (len(n.input) > 4 and n.input[4] and
                                                     (mask is None or (mask != 1).any())):
            return None
        ins = [n.input[0], n.input[1]] + ([n.input[3]] if len(n.input) > 3 and n.input[3] else [])
        attrs = {a: _a(r, n, a) for a in ("dilations", "group", "kernel_shape", "pads", "strides") if _a(r, n, a) is not None}
        r.hit("deform_zero")
        return _out(n, [], "Conv", ins, **attrs)
    core.rewrite(r, f)


def rule_causal_conv(r, core):
    def f(n):
        if n.op_type != "CausalConvWithState":
            return None
        x, w = n.input[0], r.val(n.input[1])
        s = r.shapes.get(x)
        if w is None or not s or None in s:
            return None
        B, C, L = s
        k = w.shape[-1]
        past = r.val(n.input[3]) if len(n.input) > 3 and n.input[3] else np.zeros((B, C, k - 1), np.float32)
        if past is None:
            return None
        nodes = []
        if np.abs(past).max() == 0:  # zero carry: causal = left padding only
            xp, pads = x, [k - 1, 0]
        else:
            b, xp = r.node("Concat", [r.c(past.astype(np.float32)), x], axis=2)
            nodes.append(b)
            pads = [0, 0]
        ins = [xp, r.c(w.astype(np.float32))]
        if len(n.input) > 2 and n.input[2]:
            ins.append(n.input[2])
        cv, y = r.node("Conv", ins, kernel_shape=[k], group=C, pads=pads)  # rank-3 Conv1d: parses natively
        nodes.append(cv)
        if _a(r, n, "activation", "none") in ("silu", "swish"):
            sg, tsg = r.node("Sigmoid", [y])
            mu, y = r.node("Mul", [y, tsg])
            nodes += [sg, mu]
        out = [helper.make_node("Identity", [y], [n.output[0]], name=n.name + "/c3")]
        if len(n.output) > 1 and n.output[1] and _cons(r).get(n.output[1]):  # present state: last k-1 of [past, x]
            src = x if pads[0] else xp
            Ls = L if pads[0] else L + k - 1
            sl, ts = _slice(r, src, Ls - (k - 1), Ls, 2)
            out += [sl, helper.make_node("Identity", [ts], [n.output[1]], name=n.name + "/c3s")]
        r.hit("causal_conv")
        return nodes + out
    core.rewrite(r, f)


def rule_space2depth(r, core):
    def f(n):
        if n.op_type != "SpaceToDepth":
            return None
        s = r.shapes.get(n.input[0])
        b = _a(r, n, "blocksize")
        if not s or None in s:
            return None
        N, C, H, W = s
        # torch pixel_unshuffle order (c, i, j) parses; ONNX wants (i, j, c): one channel Gather
        a, t1 = r.node("Reshape", [n.input[0], r.c(I64([N, C, H // b, b, W // b, b]), np.int64)])
        tr, t2 = r.node("Transpose", [t1], perm=[0, 1, 3, 5, 2, 4])
        rs, t3 = r.node("Reshape", [t2, r.c(I64([N, C * b * b, H // b, W // b]), np.int64)])
        idx = [c * b * b + i * b + j for i in range(b) for j in range(b) for c in range(C)]
        r.hit("space2depth")
        return _out(n, [a, tr, rs], "Gather", [t3, r.c(I64(idx), np.int64)], axis=1)
    core.rewrite(r, f)


# ---------------------------------------------------------------- spectral: complex results as two real planes
def _complex_consumers(r, n, re, im, nodes, post=None):
    """Rewrite the consumers of a (..., 2) complex output that reduce or select the last axis; returns the
    replacement map {consumer name: nodes} or None if some consumer needs the stacked tensor."""
    cons = _cons(r).get(n.output[0], [])
    rk = r.rank(n.output[0])
    rep = {}
    for c in cons:
        if post is not None:  # the planes carry an extra size-1 axis: compute into a temp, Reshape to the ONNX shape
            final = list(c.output)
            c = helper.make_node(c.op_type, list(c.input), [c.output[0] + "__p"], name=c.name, **{
                a.name: helper.get_attribute_value(a) for a in c.attribute})
            c.output[0] = final[0] + "__p"
        ax = r.axes_of(c) if c.op_type.startswith("Reduce") else None
        last = ax is not None and [a % rk for a in ax] == [rk - 1]
        kd = _a(r, c, "keepdims", 1)
        if c.op_type == "ReduceSum" and last and not kd:
            rep[c.name] = [helper.make_node("Add", [re, im], list(c.output), name=c.name + "/c3")]
        elif c.op_type in ("ReduceSumSquare", "ReduceL2") and last and not kd:
            a, t1 = r.node("Mul", [re, re])
            b, t2 = r.node("Mul", [im, im])
            if c.op_type == "ReduceL2":
                d, t3 = r.node("Add", [t1, t2])
                rep[c.name] = [a, b, d, helper.make_node("Sqrt", [t3], list(c.output), name=c.name + "/c3")]
            else:
                rep[c.name] = [a, b, helper.make_node("Add", [t1, t2], list(c.output), name=c.name + "/c3")]
        elif c.op_type == "Gather" and _a(r, c, "axis", 0) % rk == rk - 1 and r.val(c.input[1]) is not None \
                and np.asarray(r.val(c.input[1])).ndim == 0:
            i = int(r.val(c.input[1]))
            rep[c.name] = [helper.make_node("Identity", [re if i == 0 else im], list(c.output), name=c.name + "/c3")]
        else:
            return None
        if post is not None:
            rep[c.name].append(helper.make_node("Reshape", [c.output[0], r.c(I64(post), np.int64)], final,
                                                name=c.name + "/c3p"))
    return rep


def _apply_complex(r, core, n, nodes, re, im, post=None):
    rep = _complex_consumers(r, n, re, im, nodes, post)
    if rep is None:
        return False
    first = {"done": False}

    def g(m):
        if m.name == n.name:
            return nodes
        return rep.get(m.name)
    core.rewrite(r, g)
    first["done"] = True
    return True


def rule_dft(r, core):
    for n in list(r.g.node):
        if n.op_type != "DFT" or _a(r, n, "inverse", 0):
            continue
        s = r.shapes.get(n.input[0])
        if not s or None in s or s[-1] != 1 or len(s) != 5:
            continue
        opset = next(o.version for o in r.m.opset_import if o.domain in ("", "ai.onnx"))
        if opset >= 20:
            axv = r.val(n.input[2]) if len(n.input) > 2 and n.input[2] else np.array(-2)
            if axv is None:
                continue
            axis = int(np.asarray(axv).ravel()[0]) % 5
        else:
            axis = _a(r, n, "axis", 1) % 5
        if axis != 3:
            continue
        L = s[3]
        nb = L // 2 + 1 if _a(r, n, "onesided", 0) else L
        t = np.arange(L)[:, None] * np.arange(nb)[None] * 2 * np.pi / L
        sq, x4 = r.node("Squeeze", [n.input[0], r.c(I64([4]), np.int64)])
        re, im = n.name + "/re", n.name + "/im"
        m1 = helper.make_node("MatMul", [x4, r.c(np.cos(t).astype(np.float32))], [re], name=n.name + "/mre")
        m2 = helper.make_node("MatMul", [x4, r.c((-np.sin(t)).astype(np.float32))], [im], name=n.name + "/mim")
        if _apply_complex(r, core, n, [sq, m1, m2], re, im):
            r.hit("dft")


def rule_stft(r, core):
    for n in list(r.g.node):
        if n.op_type != "STFT":
            continue
        s = r.shapes.get(n.input[0])
        step = r.val(n.input[1])
        win = r.val(n.input[2]) if len(n.input) > 2 and n.input[2] else None
        fl = r.val(n.input[3]) if len(n.input) > 3 and n.input[3] else None
        if not s or None in s or step is None or (win is None and fl is None) or s[-1] != 1 or s[0] != 1:
            continue
        nf = int(len(win) if win is not None else int(np.asarray(fl).ravel()[0]))
        win = np.ones(nf) if win is None else win
        step = int(np.asarray(step).ravel()[0])
        L = s[1]
        nb = nf // 2 + 1 if _a(r, n, "onesided", 1) else nf
        frames = (L - nf) // step + 1
        t = np.arange(nb)[:, None] * np.arange(nf)[None] * 2 * np.pi / nf
        # 2 identical input channels with half kernels: a 1-input-channel conv is classed depthwise and rejected
        kc = np.repeat((np.cos(t) * win)[:, None, None, :] / 2, 2, 1).astype(np.float32)
        ks = np.repeat((-np.sin(t) * win)[:, None, None, :] / 2, 2, 1).astype(np.float32)
        a, x4 = r.node("Reshape", [n.input[0], r.c(I64([1, 1, 1, L]), np.int64)])
        a2, x2 = r.node("Concat", [x4, x4], axis=1)
        c1, yr = r.node("Conv", [x2, r.c(kc)], kernel_shape=[1, nf], strides=[1, step])  # (1, nb, 1, frames)
        c2, yi = r.node("Conv", [x2, r.c(ks)], kernel_shape=[1, nf], strides=[1, step])
        nodes = [a, a2, c1, c2]
        planes = []
        for y in (yr, yi):  # C<->W (parses): (1, frames, 1, nb); the consumers reshape to (1, frames, nb) after
            tr, t1 = r.node("Transpose", [y], perm=[0, 3, 2, 1])
            nodes.append(tr)
            planes.append(t1)
        if _apply_complex(r, core, n, nodes, *planes, post=[1, frames, nb]):
            r.hit("stft")


# ---------------------------------------------------------------- einsum
def rule_einsum(r, core):
    def f(n):
        if n.op_type != "Einsum" or len(n.input) != 2:
            return None
        eq = _a(r, n, "equation").decode() if isinstance(_a(r, n, "equation"), bytes) else _a(r, n, "equation")
        eq = eq.replace(" ", "")
        if "..." in eq or "->" not in eq:
            return None
        lhs, out = eq.split("->")
        a, b = lhs.split(",")
        if len(set(a)) != len(a) or len(set(b)) != len(b):
            return None
        A, B = n.input
        nodes = []
        # 1) indices private to one operand and absent from the output: sum them out first
        for which in (0, 1):
            sub, t = (a, A) if which == 0 else (b, B)
            other = b if which == 0 else a
            priv = [i for i, ch in enumerate(sub) if ch not in other and ch not in out]
            if priv:
                rd, t = r.node("ReduceSum", [t, r.c(I64(priv), np.int64)], keepdims=1)
                nodes.append(rd)
                sub = "".join("1" if i in priv else ch for i, ch in enumerate(sub))  # keep rank: size-1 dims
            if which == 0:
                a, A = sub, t
            else:
                b, B = sub, t
        contr = [ch for ch in a if ch in b and ch not in out and ch != "1"]
        wv = r.val(B)
        sa = r.shapes.get(n.input[0])
        if not contr:  # element-wise product with broadcasting: transpose both to the output order (+ size-1 dims)
            def to_out(sub, t):
                if len(sub) == len(out) and sub.count("1") == sum(ch not in sub for ch in out):
                    ones = iter([i for i, ch in enumerate(sub) if ch == "1"])
                    perm = [sub.index(ch) if ch in sub else next(ones) for ch in out]
                    if perm == list(range(len(sub))):
                        return [], t
                    tr, t = r.node("Transpose", [t], perm=perm)
                    return [tr], t
                letters = [ch for ch in sub if ch != "1"]
                order = [sub.index(ch) for ch in out if ch in letters]
                ones = [i for i, ch in enumerate(sub) if ch == "1"]
                perm = order + ones
                nd = []
                if perm != list(range(len(sub))):
                    tr, t = r.node("Transpose", [t], perm=perm)
                    nd.append(tr)
                    sub = "".join(sub[p] for p in perm)
                # insert size-1 dims for output letters this operand lacks
                shape_sub = [ch for ch in out]
                missing = [i for i, ch in enumerate(shape_sub) if ch not in sub]
                if ones:  # drop the reduced dims (they are trailing after the permutation)
                    sq, t = r.node("Squeeze", [t, r.c(I64(list(range(len(order), len(sub)))), np.int64)])
                    nd.append(sq)
                if missing:
                    un, t = r.node("Unsqueeze", [t, r.c(I64(missing), np.int64)])
                    nd.append(un)
                return nd, t
            n1, ta = to_out(a, A)
            n2, tb = to_out(b, B)
            r.hit("einsum")
            return _out(n, nodes + n1 + n2, "Mul", [ta, tb])
        # 2) contraction with a constant 2-D operand over the channel axis of NCHW -> 1x1 conv
        if wv is not None and len(b) == 2 and len(a) == 4 and a[1] in contr and len(contr) == 1 and \
                out == a[0] + b.replace(a[1], "") + a[2:]:
            w = wv if b.index(a[1]) == 1 else wv.T  # (out_ch, in_ch)
            r.hit("einsum")
            return _out(n, nodes, "Conv", [A, r.c(w.reshape(w.shape[0], w.shape[1], 1, 1).astype(np.float32))],
                        kernel_shape=[1, 1])
        # 3) general: batch letters, then MatMul over one contracted letter (both rank 4, attention-like)
        if len(contr) == 1 and len(a) == len(b) == len(out) == 4 and "1" not in a + b:
            c = contr[0]
            batch = [ch for ch in out if ch in a and ch in b]
            fa = [ch for ch in out if ch in a and ch not in b]
            fb = [ch for ch in out if ch in b and ch not in a]
            if len(batch) == 2 and len(fa) == 1 and len(fb) == 1:
                pa = [a.index(ch) for ch in batch + fa + [c]]
                pb = [b.index(ch) for ch in batch + [c] + fb]
                t1, ta = r.node("Transpose", [A], perm=pa)
                t2, tb = r.node("Transpose", [B], perm=pb)
                mm, tm = r.node("MatMul", [ta, tb])
                cur = batch + fa + fb
                po = [cur.index(ch) for ch in out]
                r.hit("einsum")
                return _out(n, nodes + [t1, t2, mm], "Transpose", [tm], perm=po)
        return None
    core.rewrite(r, f)


# ---------------------------------------------------------------- quantized models -> float
def rule_qdq(r, core):
    prod = _prod(r)
    cons = _cons(r)

    def deq(name, scale, zp, axis=1):
        q = r.val(name)
        sc = r.val(scale)
        z = r.val(zp) if zp else np.zeros_like(sc)
        if q is None or sc is None or z is None:
            return None
        if sc.size > 1:
            shp = [1] * q.ndim
            shp[axis] = -1
            sc, z = sc.reshape(shp), z.reshape(shp)
        return ((q.astype(np.float32) - z.astype(np.float32)) * sc).astype(np.float32)

    def f(n):
        t = n.op_type
        if t == "DequantizeLinear":
            w = deq(n.input[0], n.input[1], n.input[2] if len(n.input) > 2 else "", _a(r, n, "axis", 1))
            if w is not None:
                r.hit("qdq")
                return [helper.make_node("Identity", [r.c(w)], list(n.output), name=n.name + "/c3")]
            p = prod.get(n.input[0])
            if p is not None and p.op_type == "QuantizeLinear":  # Q -> DQ on activations: the float value itself
                r.hit("qdq")
                return [helper.make_node("Identity", [p.input[0]], list(n.output), name=n.name + "/c3")]
            return None
        if t == "QLinearConv":
            w = deq(n.input[3], n.input[4], n.input[5], 0)
            if w is None:
                return None
            xs, ys = r.val(n.input[1]), r.val(n.input[6])
            nodes = []
            dq, x = r.node("DequantizeLinear", [n.input[0], n.input[1], n.input[2]])
            ins = [x, r.c(w)]
            if len(n.input) > 8 and n.input[8]:
                bq = r.val(n.input[8])
                ws = r.val(n.input[4])
                ins.append(r.c((bq.astype(np.float32) * (xs * ws).ravel()).astype(np.float32)))
            attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute}
            cv, y = r.node("Conv", ins, **attrs)
            nodes += [dq, cv]
            r.hit("qdq")
            del ys
            return _out(n, nodes, "QuantizeLinear", [y, n.input[6], n.input[7]])
        if t == "QLinearMatMul":
            w = deq(n.input[3], n.input[4], n.input[5], 1)
            if w is None:
                return None
            dq, x = r.node("DequantizeLinear", [n.input[0], n.input[1], n.input[2]])
            y = n.name + "/mm_out"  # not compat_-named: matmul_const must still rewrite it
            mm = helper.make_node("MatMul", [x, r.c(w)], [y], name=n.name + "/mm")
            r.hit("qdq")
            return _out(n, [dq, mm], "QuantizeLinear", [y, n.input[6], n.input[7]])
        if t == "Mul":  # Mul(Cast(MatMulInteger|ConvInteger(DynamicQuantizeLinear(x), Wq)), xs * ws)
            for ci, si in ((0, 1), (1, 0)):
                c = prod.get(n.input[ci])
                if c is None or c.op_type != "Cast":
                    continue
                mi = prod.get(c.input[0])
                if mi is None or mi.op_type not in ("MatMulInteger", "ConvInteger"):
                    continue
                dql = prod.get(mi.input[0])
                sm = prod.get(n.input[si])
                if dql is None or dql.op_type != "DynamicQuantizeLinear" or sm is None or sm.op_type != "Mul":
                    continue
                if dql.output[1] not in sm.input:
                    continue
                wsn = [i for i in sm.input if i != dql.output[1]][0]
                ws = r.val(wsn)
                wq = r.val(mi.input[1])
                wz = r.val(mi.input[3]) if len(mi.input) > 3 and mi.input[3] else np.zeros(1)
                if ws is None or wq is None or wz is None:
                    continue
                x = dql.input[0]
                if mi.op_type == "MatMulInteger":
                    wf = ((wq.astype(np.float32) - wz.astype(np.float32)) * ws).astype(np.float32)
                    r.hit("qdq")
                    return _out(n, [], "MatMul", [x, r.c(wf)])
                wf = ((wq.astype(np.float32) - wz.astype(np.float32).reshape(-1, 1, 1, 1)) *
                      np.asarray(ws, np.float32).reshape(-1, 1, 1, 1)).astype(np.float32)
                attrs = {a.name: helper.get_attribute_value(a) for a in mi.attribute}
                r.hit("qdq")
                return _out(n, [], "Conv", [x, r.c(wf)], **attrs)
        return None
    core.rewrite(r, f)
    # graph-input Quantize -> ... : a float model input feeding QuantizeLinear whose only consumer chain is DQ is
    # handled above; a final QuantizeLinear on a graph output is left (the model's own integer output contract)
    del cons


def reference(path, feeds):
    """Float reference outputs: onnxruntime, or onnx's reference evaluator for ops onnxruntime lacks."""
    import onnxruntime as ort
    try:
        return ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, feeds)
    except Exception:  # noqa: BLE001
        from onnx.reference import ReferenceEvaluator
        return ReferenceEvaluator(path).run(None, feeds)


_AXES_INPUT_18 = {"ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd", "ReduceL1", "ReduceL2", "ReduceLogSum",
                  "ReduceLogSumExp", "ReduceSumSquare"}


def lower_opset(m, target=17):
    """Rewritten graph -> opset `target` when every node exists unchanged there (Reduce* axes input -> attribute):
    the DFC env's onnx 1.16 / parser predate the newest opsets."""
    import onnx
    from onnx import numpy_helper as nh
    cur = next(o for o in m.opset_import if o.domain in ("", "ai.onnx"))
    if cur.version <= target:
        return False
    inits = {t.name: t for t in m.graph.initializer}
    consts = {n.output[0]: n for n in m.graph.node if n.op_type == "Constant"}
    for n in m.graph.node:
        if n.domain not in ("", "ai.onnx"):
            continue
        try:
            a = onnx.defs.get_schema(n.op_type, cur.version, "")
            b = onnx.defs.get_schema(n.op_type, target, "")
        except onnx.defs.SchemaError:
            return False
        if a.since_version != b.since_version and not (n.op_type in _AXES_INPUT_18 and b.since_version <= target):
            if n.op_type not in ("Constant", "Identity", "Cast", "Reshape", "Shape", "Pad", "Squeeze", "Unsqueeze",
                                 "Resize", "Equal", "Where", "Mul", "Add", "Sub", "Div", "Split", "MaxPool",
                                 "Conv", "ConvTranspose", "Transpose", "Concat", "Slice", "Gather", "Softmax",
                                 "MatMul", "Relu", "Sigmoid", "Tanh", "Neg", "Abs", "Sqrt", "Exp", "Log", "Pow",
                                 "Greater", "Less", "Clip", "AveragePool", "GlobalAveragePool", "Flatten",
                                 "BatchNormalization", "InstanceNormalization", "Tile", "ReduceSum", "Expand"):
                return False
    for n in m.graph.node:  # attributes added after `target` (defaults written by the defaults rule)
        if n.domain in ("", "ai.onnx"):
            known = onnx.defs.get_schema(n.op_type, target, "").attributes
            keep = [a for a in n.attribute if a.name in known]
            del n.attribute[:]
            n.attribute.extend(keep)
    for n in m.graph.node:  # Reduce* (not ReduceSum) axes input -> attribute
        if n.op_type in _AXES_INPUT_18 and len(n.input) > 1 and n.input[1]:
            t = inits.get(n.input[1])
            v = nh.to_array(t) if t is not None else (nh.to_array(consts[n.input[1]].attribute[0].t)
                                                      if n.input[1] in consts else None)
            if v is None:
                return False
            del n.input[1:]
            n.attribute.append(helper.make_attribute("axes", [int(x) for x in np.asarray(v).ravel()]))
        if n.op_type == "Split" and any(a.name == "num_outputs" for a in n.attribute):
            return False
    cur.version = target
    m.ir_version = min(m.ir_version, 8)
    return True
