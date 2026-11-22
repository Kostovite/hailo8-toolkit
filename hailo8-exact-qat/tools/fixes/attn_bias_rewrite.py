"""Exact attention bias rewrite (RULES.md section 26): the quantizer gives an offset K / V operand of a dynamic matmul
a non-zero zero point that is never compensated (zp_comp_none), so K and V are best centred before quantization.
  K side: softmax(Q.(K + B)^T) = softmax(Q.K^T + Q.B^T): with B constant along the key axis, Q.B^T is constant along
          the softmax axis -> drop the bias elements that reach only K.
  V side: P.(V + B) = P.V + B because every row of P sums to 1 -> move the bias elements that reach only V after P.V.
Bias candidates are the constant vectors added upstream of K / V (Conv / Gemm bias inputs, Add constants), found by
walking back from the operand through layout ops (Reshape, Transpose, Gather, Split, Slice, ...) to the first linear op.
Which operand each element reaches is measured by perturbing it (onnxruntime), so fused projections (one in_proj for
Q, K, V as in nn.MultiheadAttention) are handled element by element. Every change is kept only if the float output
is unchanged (rtol 1e-4) on random inputs.
Usage: python attn_bias_rewrite.py IN.onnx OUT.onnx"""
import copy
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper as nh

LAYOUT = {"Reshape", "Transpose", "Flatten", "Squeeze", "Unsqueeze", "Gather", "Split", "Slice", "Identity"}
SCALE = {"Mul", "Div"}


class Net:
    def __init__(self, m):
        self.m = m
        g = m.graph
        self.prod = {o: n for n in g.node for o in n.output}
        self.inits = {i.name: i for i in g.initializer}
        self.feeds = None

    def const(self, t):
        return t in self.inits or (t in self.prod and self.prod[t].op_type == "Constant")

    def run(self, tensors, inits=None):
        m = copy.deepcopy(self.m)
        if inits:
            for i in m.graph.initializer:
                if i.name in inits:
                    i.CopyFrom(nh.from_array(inits[i.name].astype(nh.to_array(i).dtype), i.name))
        have = {o.name for o in m.graph.output}
        for t in tensors:
            if t not in have:
                m.graph.output.append(helper.make_tensor_value_info(t, 1, None))
        s = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
        return dict(zip(tensors, s.run(list(tensors), self.feeds)))

    def bias_sources(self, t):
        """Constant vectors added upstream of tensor t before the first linear op: [(init name)]."""
        out, stack, seen = [], [t], set()
        while stack:
            t = stack.pop()
            if t in seen or t not in self.prod:
                continue
            seen.add(t)
            n = self.prod[t]
            if n.op_type in LAYOUT or n.op_type in SCALE:
                stack += [x for x in n.input[:1] if not self.const(x)]
            elif n.op_type == "Add":
                for x in n.input:
                    if x in self.inits:
                        out.append(x)
                    elif not self.const(x):
                        stack.append(x)
            elif n.op_type in ("Conv", "Gemm") and len(n.input) > 2 and n.input[2] in self.inits:
                out.append(n.input[2])
        return out


def attentions(net):
    g = net.m.graph
    cons = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    for sm in [n for n in g.node if n.op_type == "Softmax"]:
        ax = next((a.i for a in sm.attribute if a.name == "axis"), -1)
        t = sm.input[0]
        while t in net.prod and net.prod[t].op_type in SCALE and any(net.const(x) for x in net.prod[t].input):
            t = next(x for x in net.prod[t].input if not net.const(x))
        qk = net.prod.get(t)
        if qk is None or qk.op_type != "MatMul" or ax not in (-1, 3, 2):
            continue
        pv = [n for n in cons.get(sm.output[0], []) if n.op_type == "MatMul" and n.input[0] == sm.output[0]]
        yield qk, (pv[0] if pv else None)


def main(src, dst):
    m, moved = rewrite_model(onnx.load(src))
    onnx.save(m, dst)
    print(f"attn_bias_rewrite: {len(moved)} bias move(s) {moved} -> {dst}")


def rewrite_model(m):
    """ModelProto -> (rewritten ModelProto, list of moves)."""
    net = Net(m)
    rng = np.random.default_rng(0)
    net.feeds = {i.name: rng.normal(0, 1, [d.dim_value or 1 for d in i.type.tensor_type.shape.dim]).astype(np.float32)
                 for i in m.graph.input if i.name not in net.inits}
    ref = net.run([o.name for o in m.graph.output])
    moved = []
    for qk, pv in list(attentions(net)):
        ops = {"Q": qk.input[0], "K": qk.input[1]}
        if pv is not None:
            ops["V"] = pv.input[1]
        cands = sorted(set(net.bias_sources(ops["K"]) + (net.bias_sources(ops["V"]) if pv is not None else [])))
        base = net.run(list(ops.values()))
        for b in cands:
            vec = nh.to_array(net.inits[b]).astype(np.float64)
            if vec.ndim != 1:
                continue
            reach = np.zeros((vec.size, 3), bool)  # element j changes Q / K / V
            for j in range(vec.size):
                p = vec.copy(); p[j] += 1.0
                r = net.run(list(ops.values()), {b: p})
                reach[j] = [not np.allclose(r[ops[k]], base[ops[k]]) if k in ops else False for k in ("Q", "K", "V")]
            kmask = reach[:, 1] & ~reach[:, 0] & ~reach[:, 2]
            vmask = reach[:, 2] & ~reach[:, 0] & ~reach[:, 1]
            for kind, mask in (("K", kmask), ("V", vmask)):
                if not mask.any() or not np.any(vec[mask]):
                    continue
                new = vec.copy(); new[mask] = 0.0
                m2 = copy.deepcopy(net.m)
                for i in m2.graph.initializer:
                    if i.name == b:
                        i.CopyFrom(nh.from_array(new.astype(nh.to_array(i).dtype), b))
                if kind == "V":  # add the removed contribution after P.V: it is constant along the token axis
                    dv = base[ops["V"]] - net.run([ops["V"]], {b: new})[ops["V"]]
                    row = dv[..., :1, :]
                    if not np.allclose(dv, np.broadcast_to(row, dv.shape), atol=1e-5):
                        print(f"skip V bias {b}: not constant along the token axis")
                        continue
                    g = m2.graph
                    pvn = next(n for n in g.node if list(n.output) == list(pv.output))
                    out = pvn.output[0]
                    pvn.output[0] = out + "_pre_vbias"
                    g.initializer.append(nh.from_array(row.astype(np.float32), out + "_vbias"))
                    g.node.insert(list(g.node).index(pvn) + 1, helper.make_node("Add", [out + "_pre_vbias", out + "_vbias"], [out]))
                trial = Net(m2); trial.feeds = net.feeds
                got = trial.run([o.name for o in m.graph.output])
                if all(np.allclose(got[k], ref[k], rtol=1e-4, atol=1e-4 * max(1.0, float(np.abs(ref[k]).max()))) for k in ref):
                    net = trial
                    moved.append(f"{kind}:{b}[{int(mask.sum())}]")
                    base = net.run(list(ops.values()))
                    vec = new
                else:
                    print(f"reject {kind} bias {b}: float output changed")
    return net.m, moved


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
