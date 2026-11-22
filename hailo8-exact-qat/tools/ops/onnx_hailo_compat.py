"""Rewrite ONNX ops the Hailo DFC 3.33 parser rejects or silently mistranslates into equivalent supported ops.

Every rule is exact in float unless marked (approx). Found by zoo/ops/probe.py (see FIX_LATER.md):
  fold         constant folding (onnxruntime basic level: standard ops only) so computed pads/shapes become consts
  defaults     write schema-default attributes (parser reads attributes without defaults: BatchNorm epsilon)
  pow          Pow(x,2)->x*x, 3->x*x*x, 0.5->Sqrt, -0.5->Reciprocal(Sqrt) (inv_sqrt; native pow -0.5 is wrong on
               chip), -1->Reciprocal, 1.5->x*Sqrt(x)
  reduce_pool  ReduceMax/ReduceMin over H, W or H*W -> MaxPool / GlobalMaxPool (min = -max(-x))
  reduce_l1    ReduceL1 -> ReduceSum(Abs)
  pad_mode     Pad reflect/edge (parsed as ZERO padding!) -> Slice + Concat
  ew_minmax    Max(a,b)/Min(a,b) of two tensors -> a + Relu(b-a) / a - Relu(a-b) (native: 10-27 dB)
  argmin       ArgMin -> ArgMax(Neg)
  split        Split -> Slices (parser loses a Split feeding one node: 'bias_add1' is not in list)
  expand       Expand of a size-1 axis -> Tile
  gelu         Gelu (opset 20, erf or tanh) and non-torch erf orders -> torch's Div/Erf/Add/Mul/Mul (native gelu);
               tanh GELU -> erf GELU (approx, max |diff| 5e-4)
  mish         Mish op (opset 18) -> x * Tanh(Softplus(x)) (the parser's native mish pattern)
  gn1          GroupNorm(1) (Reshape [0,1,-1] -> InstanceNorm -> Reshape) -> spatial-then-channel statistics
               (the parser's channel-first whole-tensor stats are WRONG on the chip; this order is exact)
  convt_outpad ConvTranspose output_padding (computed WRONG on chip: -3 dB) -> kernel zero-padded at the end by
               output_padding, attribute dropped (same output size and taps: exact)
  shrink       Shrink op -> x - Clip(x); torch's Where(u > t, u - t, 0) -> Relu(u - t) and
               Where(u < -t, u + t, 0) -> -Relu(-u - t) (Softshrink, thresholded ramps)

Usage: python onnx_hailo_compat.py IN.onnx OUT.onnx [--rules a,b,...] [--no-check]
"""
import argparse
import math
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ALL = ["fold", "defaults", "pow", "reduce_pool", "reduce_l1", "pad_mode", "ew_minmax", "argmin", "split", "expand", "gelu",
       "shrink", "convt_outpad", "mish", "gn1"]


class Rw:
    def __init__(self, m):
        self.m, self.g = m, m.graph
        self.inits = {t.name: t for t in self.g.initializer}
        self.consts = {}
        for n in self.g.node:
            if n.op_type == "Constant":
                for a in n.attribute:
                    if a.name == "value":
                        self.consts[n.output[0]] = numpy_helper.to_array(a.t)
        import re as _re  # continue the numbering of an earlier pass (names must stay unique)
        used = [int(x) for t in [n.name for n in self.g.node] + [o for n in self.g.node for o in n.output] +
                list(self.inits) for x in _re.findall(r"^compat_(?:[A-Za-z]+_|t|c)(\d+)$", t)]
        self.k = max(used, default=0)
        self.shapes = {}
        try:
            inf = onnx.shape_inference.infer_shapes(m)
            for v in list(inf.graph.value_info) + list(inf.graph.input) + list(inf.graph.output):
                d = v.type.tensor_type.shape.dim
                self.shapes[v.name] = [x.dim_value if x.HasField("dim_value") else None for x in d]
        except Exception:  # noqa: BLE001
            pass
        self.count = {}

    def val(self, name):
        if name in self.inits:
            return numpy_helper.to_array(self.inits[name])
        return self.consts.get(name)

    def c(self, arr, dtype=np.float32):
        self.k += 1
        nm = f"compat_c{self.k}"
        t = numpy_helper.from_array(np.asarray(arr, dtype), nm)
        self.g.initializer.append(t)
        self.inits[nm] = t
        return nm

    def node(self, op, ins, out=None, **attrs):
        self.k += 1
        out = out or f"compat_t{self.k}"
        return helper.make_node(op, ins, [out], name=f"compat_{op}_{self.k}", **attrs), out

    def attr(self, n, name, default=None):
        for a in n.attribute:
            if a.name == name:
                return helper.get_attribute_value(a)
        return default

    def hit(self, rule):
        self.count[rule] = self.count.get(rule, 0) + 1

    def axes_of(self, n):
        ax = self.attr(n, "axes")
        if ax is None and len(n.input) > 1 and n.input[1]:
            v = self.val(n.input[1])
            ax = None if v is None else v.tolist()
        return ax

    def rank(self, t):
        s = self.shapes.get(t)
        return len(s) if s else None


def fold_model(m):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        so.optimized_model_filepath = os.path.join(d, "folded.onnx")
        ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
        return onnx.load(so.optimized_model_filepath)


def rule_defaults(r):
    opset = next(o.version for o in r.m.opset_import if o.domain in ("", "ai.onnx"))
    for n in r.g.node:
        try:
            sch = onnx.defs.get_schema(n.op_type, opset, n.domain)
        except onnx.defs.SchemaError:
            continue
        have = {a.name for a in n.attribute}
        for an, a in sch.attributes.items():
            if an not in have and a.default_value is not None and a.default_value.type != 0:
                n.attribute.append(a.default_value)
                n.attribute[-1].name = an
                r.hit("defaults")


def rewrite(r, fn):
    """Apply fn(node) -> list of replacement nodes or None, over the graph in order."""
    out = []
    for n in list(r.g.node):
        rep = fn(n)
        out.extend([n] if rep is None else rep)
    del r.g.node[:]
    r.g.node.extend(out)


def rule_pow(r):
    def f(n):
        if n.op_type != "Pow":
            return None
        e = r.val(n.input[1])
        if e is None or e.size != 1:
            return None
        e = float(e.ravel()[0])
        x, y = n.input[0], n.output[0]
        if e == 2:
            nodes = [helper.make_node("Mul", [x, x], [y], name=n.name + "/sq")]
        elif e == 3:
            a, t = r.node("Mul", [x, x])
            nodes = [a, helper.make_node("Mul", [t, x], [y], name=n.name + "/cube")]
        elif e == 0.5:
            nodes = [helper.make_node("Sqrt", [x], [y], name=n.name + "/sqrt")]
        elif e == -0.5:
            a, t = r.node("Sqrt", [x])
            nodes = [a, helper.make_node("Reciprocal", [t], [y], name=n.name + "/rsqrt")]
        elif e == -1:
            nodes = [helper.make_node("Reciprocal", [x], [y], name=n.name + "/inv")]
        elif e == 1.5:
            a, t = r.node("Sqrt", [x])
            nodes = [a, helper.make_node("Mul", [x, t], [y], name=n.name + "/x_sqrtx")]
        elif e == 1:
            nodes = [helper.make_node("Identity", [x], [y], name=n.name + "/id")]
        else:
            return None
        r.hit("pow")
        return nodes
    rewrite(r, f)


def rule_reduce_pool(r):
    def f(n):
        if n.op_type not in ("ReduceMax", "ReduceMin") or r.rank(n.input[0]) != 4:
            return None
        ax = r.axes_of(n)
        if ax is None:
            return None
        ax = sorted(a % 4 for a in ax)
        if ax not in ([2], [3], [2, 3]) or not r.attr(n, "keepdims", 1):
            return None
        H, W = r.shapes[n.input[0]][2:4]
        x, y = n.input[0], n.output[0]
        nodes = []
        if n.op_type == "ReduceMin":
            a, x = r.node("Neg", [x])
            nodes.append(a)
        if ax == [2, 3]:
            p, t = r.node("GlobalMaxPool", [x])
        else:
            k = [H, 1] if ax == [2] else [1, W]
            p, t = r.node("MaxPool", [x], kernel_shape=k, strides=k)
        nodes.append(p)
        nodes.append(helper.make_node("Neg" if n.op_type == "ReduceMin" else "Identity", [t], [y], name=n.name + "/out"))
        r.hit("reduce_pool")
        return nodes
    rewrite(r, f)


def rule_reduce_l1(r):
    def f(n):
        if n.op_type != "ReduceL1":
            return None
        a, t = r.node("Abs", [n.input[0]])
        ax = r.axes_of(n)
        rs = helper.make_node("ReduceSum", [t, r.c(np.array(ax, np.int64), np.int64)] if ax is not None else [t],
                              list(n.output), name=n.name + "/sum", keepdims=r.attr(n, "keepdims", 1))
        r.hit("reduce_l1")
        return [a, rs]
    rewrite(r, f)


def rule_pad_mode(r):
    def f(n):
        if n.op_type != "Pad" or r.attr(n, "mode", b"constant") not in (b"reflect", b"edge", "reflect", "edge"):
            return None
        mode = r.attr(n, "mode")
        mode = mode.decode() if isinstance(mode, bytes) else mode
        pads = r.val(n.input[1])
        s = r.shapes.get(n.input[0])
        if pads is None or not s or len(s) != 4 or any(v is None for v in s):
            return None
        pads = pads.tolist()
        nodes, x = [], n.input[0]
        for axis in (2, 3):
            b, e = pads[axis], pads[4 + axis]
            if not (b or e):
                continue
            L = s[axis]
            parts = []

            def sl(i0, i1, step=1):
                a, t = r.node("Slice", [x, r.c([i0], np.int64), r.c([i1], np.int64), r.c([axis], np.int64),
                                        r.c([step], np.int64)])
                nodes.append(a)
                return t
            for i in range(b, 0, -1):  # left: reflect -> x[i], edge -> x[0]
                j = i if mode == "reflect" else 0
                parts.append(sl(j, j + 1))
            parts.append(x)
            for i in range(1, e + 1):  # right: reflect -> x[L-1-i], edge -> x[L-1]
                j = L - 1 - i if mode == "reflect" else L - 1
                parts.append(sl(j, j + 1))
            a, x = r.node("Concat", parts, axis=axis)
            nodes.append(a)
            s = list(s)
            s[axis] = L + b + e
        nodes.append(helper.make_node("Identity", [x], [n.output[0]], name=n.name + "/out"))
        r.hit("pad_mode")
        return nodes
    rewrite(r, f)


def rule_ew_minmax(r):
    def f(n):
        if n.op_type not in ("Max", "Min") or len(n.input) != 2:
            return None
        if r.val(n.input[0]) is not None or r.val(n.input[1]) is not None:
            return None  # with a constant it is an activation the parser handles
        a, b = n.input
        if n.op_type == "Max":  # max(a,b) = a + relu(b - a)
            s, d = r.node("Sub", [b, a])
            q, t = r.node("Relu", [d])
            o = helper.make_node("Add", [a, t], list(n.output), name=n.name + "/max")
        else:  # min(a,b) = a - relu(a - b)
            s, d = r.node("Sub", [a, b])
            q, t = r.node("Relu", [d])
            o = helper.make_node("Sub", [a, t], list(n.output), name=n.name + "/min")
        r.hit("ew_minmax")
        return [s, q, o]
    rewrite(r, f)


def rule_argmin(r):
    def f(n):
        if n.op_type != "ArgMin":
            return None
        a, t = r.node("Neg", [n.input[0]])
        o = helper.make_node("ArgMax", [t], list(n.output), name=n.name + "/argmax",
                             axis=r.attr(n, "axis", 0), keepdims=r.attr(n, "keepdims", 1))
        r.hit("argmin")
        return [a, o]
    rewrite(r, f)


def rule_split(r):
    def f(n):
        if n.op_type != "Split":
            return None
        axis = r.attr(n, "axis", 0)
        s = r.shapes.get(n.input[0])
        if not s or s[axis] is None:
            return None
        sizes = r.val(n.input[1]) if len(n.input) > 1 and n.input[1] else r.attr(n, "split")
        if sizes is None:
            k = len(n.output)
            sizes = [s[axis] // k] * k
        sizes = list(np.asarray(sizes).ravel())
        nodes, start = [], 0
        for o, sz in zip(n.output, sizes):
            nodes.append(helper.make_node("Slice", [n.input[0], r.c([start], np.int64), r.c([start + int(sz)], np.int64),
                                                    r.c([axis], np.int64)], [o], name=f"{n.name}/slice{start}"))
            start += int(sz)
        r.hit("split")
        return nodes
    rewrite(r, f)


def rule_expand(r):
    def f(n):
        if n.op_type != "Expand":
            return None
        s, tgt = r.shapes.get(n.input[0]), r.val(n.input[1])
        if not s or tgt is None or len(s) != len(tgt) or any(v is None for v in s):
            return None
        reps = [int(t // d) if d == 1 else 1 for d, t in zip(s, tgt.tolist())]
        if any(d != 1 and d != t for d, t in zip(s, tgt.tolist())):
            return None
        r.hit("expand")
        return [helper.make_node("Tile", [n.input[0], r.c(reps, np.int64)], list(n.output), name=n.name + "/tile")]
    rewrite(r, f)


def _torch_gelu(r, x, y, base):
    """torch.onnx.export's GELU: Div(x, sqrt2) -> Erf -> Add(1) -> Mul(x) -> Mul(0.5) (the parser's gelu pattern)."""
    d, t1 = r.node("Div", [x, r.c(np.array(math.sqrt(2), np.float32))])
    e, t2 = r.node("Erf", [t1])
    a, t3 = r.node("Add", [t2, r.c(np.array(1.0, np.float32))])
    m1, t4 = r.node("Mul", [x, t3])
    m2 = helper.make_node("Mul", [t4, r.c(np.array(0.5, np.float32))], [y], name=base + "/gelu")
    return [d, e, a, m1, m2]


def rule_gelu(r):
    def f(n):
        if n.op_type == "Gelu":  # opset 20 op, erf or tanh (tanh -> erf: approx, |diff| <= 5e-4)
            r.hit("gelu")
            return _torch_gelu(r, n.input[0], n.output[0], n.name)
        return None
    rewrite(r, f)
    # tanh-GELU pattern: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715 x^3))) -> erf GELU (approx)
    prod = {o: n for n in r.g.node for o in n.output}
    for n in list(r.g.node):
        if n.op_type != "Tanh":
            continue
        inner = prod.get(n.input[0])
        if inner is None or inner.op_type != "Mul":
            continue
        consts = [r.val(i) for i in inner.input]
        if not any(c is not None and c.size == 1 and abs(float(c.ravel()[0]) - math.sqrt(2 / math.pi)) < 1e-3
                   for c in consts):
            continue
        # find x: the Add(x, 0.044715*x^3) input to the inner Mul
        add = prod.get(next(i for i, c in zip(inner.input, consts) if c is None))
        if add is None or add.op_type != "Add":
            continue
        x = next((i for i in add.input if prod.get(i) is None or prod[i].op_type != "Mul"), None)
        cons = [m for m in r.g.node if n.output[0] in m.input]
        # walk forward to the final 0.5*x*(1+t) Mul: the last Mul downstream of tanh within 3 hops
        chain, cur = [], n.output[0]
        for _ in range(4):  # Add(1, t) then one or two Muls (x and 0.5, in either order)
            nxt = [m for m in r.g.node if cur in m.input]
            if len(nxt) != 1 or nxt[0].op_type not in ("Add", "Mul"):
                break
            chain.append(nxt[0])
            cur = nxt[0].output[0]
        while chain and chain[-1].op_type != "Mul":
            chain.pop()
        if x is None or len(chain) < 2:
            continue
        y = chain[-1].output[0]
        chain[-1].output[0] = y + "__dead"
        r.g.node.extend(_torch_gelu(r, x, y, n.name + "_erf"))
        r.hit("gelu")


def rule_shrink(r):
    def f(n):
        if n.op_type != "Shrink":
            return None
        lam, bias = r.attr(n, "lambd", 0.5), r.attr(n, "bias", 0.0)
        if abs(lam - bias) > 1e-9:
            return None  # softshrink is lambd == bias
        c, t = r.node("Clip", [n.input[0], r.c(np.array(-lam, np.float32)), r.c(np.array(lam, np.float32))])
        r.hit("shrink")
        return [c, helper.make_node("Sub", [n.input[0], t], list(n.output), name=n.name + "/softshrink")]
    rewrite(r, f)
    prod = {o: n for n in r.g.node for o in n.output}

    def scalar(t):
        v = r.val(t)
        return None if v is None or v.size != 1 else float(v.ravel()[0])

    def g(n):
        if n.op_type != "Where" or scalar(n.input[2]) != 0.0:
            return None
        cmp, val = prod.get(n.input[0]), prod.get(n.input[1])
        if cmp is None or val is None or cmp.op_type not in ("Greater", "Less") or val.op_type not in ("Sub", "Add"):
            return None
        u, t = cmp.input[0], scalar(cmp.input[1])
        if t is None or val.input[0] != u:
            return None
        c = scalar(val.input[1])
        if c is None:
            return None
        off = -c if val.op_type == "Sub" else c  # val = u + off
        if cmp.op_type == "Greater" and abs(off + t) < 1e-12:  # (u > t) * (u - t) = relu(u - t)
            r.hit("shrink")
            return [helper.make_node("Relu", [n.input[1]], list(n.output), name=n.name + "/relu")]
        if cmp.op_type == "Less" and abs(off + t) < 1e-12:  # (u < t) * (u - t) = min(u - t, 0) = v - relu(v)
            # (not -relu(-v): the Neg -> Relu -> Neg chain quantizes to 7.9 dB on Hailo-8)
            q, rt = r.node("Relu", [n.input[1]])
            r.hit("shrink")
            return [q, helper.make_node("Sub", [n.input[1], rt], list(n.output), name=n.name + "/minzero")]
        return None
    rewrite(r, g)


def rule_convt_outpad(r):
    def f(n):
        if n.op_type != "ConvTranspose":
            return None
        op = r.attr(n, "output_padding")
        if not op or not any(op):
            return None
        w = r.val(n.input[1])
        if w is None or r.attr(n, "dilations", [1, 1]) not in ([1, 1], None):
            return None
        pad = [(0, 0), (0, 0)] + [(0, int(o)) for o in op]
        w2 = np.pad(w, pad)
        k = list(w2.shape[2:])
        attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute if a.name not in ("output_padding", "kernel_shape")}
        ins = [n.input[0], r.c(w2)] + list(n.input[2:])
        r.hit("convt_outpad")
        return [helper.make_node("ConvTranspose", ins, list(n.output), name=n.name + "/nopad", kernel_shape=k, **attrs)]
    rewrite(r, f)


def rule_gn1(r):
    """GroupNorm(1) / whole-tensor norm, torch form Reshape(x,[0,1,-1]) -> InstanceNormalization -> Reshape(back):
    the parser's channel-first statistics ([.,1] -> [1,1,1]) are computed WRONG on the chip (~98% of bytes vs
    SDK_BIT_EXACT, probe/RULES.md 18). Spatial mean first, then channel mean (same value, equal group sizes) is
    exact on the chip: m = mean_C(mean_HW x), d = x - m, y = d / sqrt(mean_C(mean_HW d^2) + eps) * scale + bias."""
    prod = {o: n for n in r.g.node for o in n.output}
    cons = {}
    for n in r.g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    drop, add_after = set(), {}
    for n in r.g.node:
        if n.op_type != "InstanceNormalization":
            continue
        rs = prod.get(n.input[0])
        if rs is None or rs.op_type != "Reshape":
            continue
        tgt = r.val(rs.input[1])
        x = rs.input[0]
        if tgt is None or len(tgt) != 3 or int(tgt[1]) != 1 or r.rank(x) != 4:
            continue
        back = cons.get(n.output[0], [])
        if len(back) != 1 or back[0].op_type != "Reshape":
            continue
        eps = r.attr(n, "epsilon", 1e-5)
        sc, bi = r.val(n.input[1]), r.val(n.input[2])
        new = []

        def mean2(t):
            a, t1 = r.node("ReduceMean", [t], axes=[2, 3], keepdims=1)
            b, t2 = r.node("ReduceMean", [t1], axes=[1], keepdims=1)
            new.extend([a, b])
            return t2
        m_ = mean2(x)
        a, d = r.node("Sub", [x, m_])
        b, sq = r.node("Mul", [d, d])
        new.extend([a, b])
        v = mean2(sq)
        a, ve = r.node("Add", [v, r.c(np.array(eps, np.float32).reshape(1, 1, 1, 1))])
        b, s = r.node("Sqrt", [ve])
        c, inv = r.node("Reciprocal", [s])
        new.extend([a, b, c])
        out = back[0].output[0]
        aff = sc is not None and bi is not None and not (np.allclose(sc, 1) and np.allclose(bi, 0))
        if aff:
            a, y = r.node("Mul", [d, inv])
            b, y2 = r.node("Mul", [y, r.c(np.asarray(sc, np.float32).reshape(1, 1, 1, 1))])
            new.extend([a, b, helper.make_node("Add", [y2, r.c(np.asarray(bi, np.float32).reshape(1, 1, 1, 1))],
                                                [out], name=n.name + "/gn1")])
        else:
            new.append(helper.make_node("Mul", [d, inv], [out], name=n.name + "/gn1"))
        drop.update([id(n), id(back[0])])
        add_after[id(back[0])] = new
        r.hit("gn1")
    if not drop:
        return
    out = []
    for n in list(r.g.node):
        if id(n) in add_after:
            out.extend(add_after[id(n)])
        elif id(n) not in drop:
            out.append(n)
    del r.g.node[:]
    r.g.node.extend(out)


def rule_mish(r):
    def f(n):
        if n.op_type != "Mish":
            return None
        a, t1 = r.node("Softplus", [n.input[0]])
        b, t2 = r.node("Tanh", [t1])
        r.hit("mish")
        return [a, b, helper.make_node("Mul", [n.input[0], t2], list(n.output), name=n.name + "/mish")]
    rewrite(r, f)


def prune(m):
    """topologically sort (rewrites append nodes), then drop nodes whose outputs nobody uses."""
    g = m.graph
    _toposort(g)
    live = {o.name for o in g.output}
    keep = []
    for n in reversed(list(g.node)):
        if any(o in live for o in n.output):
            keep.append(n)
            live.update(n.input)
    del g.node[:]
    g.node.extend(reversed(keep))


def _toposort(g):
    prod = {o: n for n in g.node for o in n.output}
    seen, order = set(), []

    def visit(n):
        if id(n) in seen:
            return
        seen.add(id(n))
        for i in n.input:
            if i in prod:
                visit(prod[i])
        order.append(n)
    for n in list(g.node):
        visit(n)
    del g.node[:]
    g.node.extend(order)


import compat2  # noqa: E402
import compat3  # noqa: E402
import compat4  # noqa: E402

for _r in compat2.RULES:
    if _r not in ALL:
        ALL.append(_r)
    globals()["rule_" + _r] = (lambda fn: (lambda r: fn(r, sys.modules[__name__])))(getattr(compat2, "rule_" + _r))
for _r in compat3.RULES:
    if _r not in ALL:
        ALL.append(_r)
    globals()["rule_" + _r] = (lambda fn: (lambda r: fn(r, sys.modules[__name__])))(getattr(compat3, "rule_" + _r))
ALL += [p for p in compat4.MODEL_PASSES if p not in ALL]  # model-level passes, applied after the node rules


def fix_output_shapes(m):
    """Graph outputs without a shape (the DFC's checker requires one): fill them from one onnxruntime run."""
    missing = [o for o in m.graph.output if not o.type.tensor_type.HasField("shape")]
    if not missing:
        return
    import onnxruntime as ort
    ort.set_default_logger_severity(3)
    s = ort.InferenceSession(m.SerializeToString())
    inp = s.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    outs = dict(zip([o.name for o in s.get_outputs()],
                    s.run(None, {inp.name: np.zeros(shape, np.float32)})))
    for o in missing:
        o.type.tensor_type.elem_type = o.type.tensor_type.elem_type or TensorProto.FLOAT
        for d in outs[o.name].shape:
            o.type.tensor_type.shape.dim.add().dim_value = int(d)


def run(src, dst, rules, check=True):
    m = onnx.load(src)
    if "group_norm" in rules:  # before folding: onnxruntime inlines GroupNormalization's function body
        r0 = Rw(m)
        rule_group_norm(r0)
        prune(m)
    pre = [x for x in rules if x in compat3.PRE_FOLD]
    pre_count = {}
    if pre:  # newest ops first: onnxruntime cannot fold a graph holding ops it has no kernel for
        r0 = Rw(m)
        for rule in pre + pre:
            globals()["rule_" + rule](r0)
            r0.shapes.update(Rw(m).shapes)
        prune(m)
        pre_count = r0.count
    if "fold" in rules:
        try:
            m = fold_model(m)
        except Exception as e:  # noqa: BLE001  (an op onnxruntime cannot run is left: host side)
            print("fold skipped:", str(e).splitlines()[0][:160])
        rules = [x for x in rules if x != "fold"]
    r = Rw(m)
    for k_, v_ in pre_count.items():
        r.count[k_] = r.count.get(k_, 0) + v_
    node_rules = [x for x in rules if x not in compat4.MODEL_PASSES and x not in compat4.ALIAS]
    for rule in node_rules + [x for x in node_rules if x not in ("defaults",)]:  # 2nd pass: ops created by later rules
        globals()["rule_" + rule](r)
        if rule != "defaults":
            r.shapes.update(Rw(m).shapes)
    prune(m)
    m, c4 = compat4.apply(m, rules)
    r.count.update(c4)
    compat3.lower_opset(m)
    fix_output_shapes(m)
    onnx.checker.check_model(m)
    onnx.save(m, dst)
    print("rewrites:", r.count or "none")
    if check:
        import onnxruntime as ort
        ort.set_default_logger_severity(3)
        inp = onnx.load(src).graph.input[0]
        shape = [d.dim_value or 1 for d in inp.type.tensor_type.shape.dim]
        x = np.random.default_rng(0).normal(0, 1.5, shape).astype(np.float32)
        if any(o in r.count for o in ("pow",)) or "Sqrt" in str(m.graph):
            x = np.abs(x) + 0.05
        ya, yb = compat3.reference(src, {inp.name: x}), compat3.reference(dst, {inp.name: x})
        err = max(float(np.abs(p - q).max() / (np.abs(p).max() + 1e-9)) for p, q in zip(ya, yb))
        print(f"max relative output difference {err:.2e}")
        return err


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--rules", default=",".join(ALL))
    ap.add_argument("--no-check", action="store_true")
    a = ap.parse_args()
    run(a.src, a.dst, a.rules.split(","), not a.no_check)
