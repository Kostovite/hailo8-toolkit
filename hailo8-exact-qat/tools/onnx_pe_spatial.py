"""Rewire YOLO26 attention so attn.pe reads V in spatial layout (no transpose through host RAM on Hailo-8).

Ultralytics Attention: qkv = Conv1x1(x); q, k, v = qkv.view(B, nh, 2kd + hd, N).split([kd, kd, hd], 2);
out = (v @ attn^T).view(B, C, H, W) + pe(v.reshape(B, C, H, W)). The Hailo parser runs the qkv conv in token layout
for the matmuls, so pe's spatial V needs a transpose that the DFC routes through a DDR portal (~2 MiB each way per
attention block per frame). Here pe gets V from its own 1x1 conv on x with the V rows of the qkv weights: identical
math, one small extra conv, no layout round trip.

Usage: python onnx_pe_spatial.py IN.onnx OUT.onnx
"""
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper

src, dst = sys.argv[1:3]
m = onnx.load(src)
g = m.graph
prod = {o: n for n in g.node for o in n.output}
inits = {t.name: t for t in g.initializer}


def const(name):
    if name in inits:
        return numpy_helper.to_array(inits[name])
    n = prod[name]
    assert n.op_type == "Constant", (name, n.op_type)
    return numpy_helper.to_array(n.attribute[0].t)


new_nodes = []
for pe in [n for n in g.node if n.op_type == "Conv" and n.name.endswith("attn/pe/conv/Conv")]:
    r2 = prod[pe.input[0]]                      # Reshape_2: v -> (B, C, H, W)
    sp = prod[r2.input[0]]                      # Split [kd, kd, hd] on axis 2
    r1 = prod[sp.input[0]]                      # Reshape: qkv -> (B, nh, 2kd + hd, N)
    qkv = prod[r1.input[0]]                     # qkv Conv1x1
    assert (r2.op_type, sp.op_type, r1.op_type, qkv.op_type) == ("Reshape", "Split", "Reshape", "Conv"), pe.name
    assert list(sp.output).index(r2.input[0]) == 2, "pe must consume the third split (v)"
    shape = const(r1.input[1])
    nh, per = int(shape[1]), int(shape[2])
    sizes = const(sp.input[1]) if len(sp.input) > 1 else [a.ints for a in sp.attribute if a.name == "split"][0]
    kd, hd = int(sizes[0]), int(sizes[2])
    assert per == 2 * kd + hd
    rows = np.concatenate([np.arange(h * per + 2 * kd, h * per + 2 * kd + hd) for h in range(nh)])
    W = const(qkv.input[1])
    B = const(qkv.input[2]) if len(qkv.input) > 2 else np.zeros(W.shape[0], W.dtype)
    base = pe.name.rsplit("/pe/", 1)[0] + "/pe_v"
    wn, bn, out = base + "/W", base + "/B", base + "/conv_output_0"
    g.initializer.extend([numpy_helper.from_array(W[rows].copy(), wn), numpy_helper.from_array(B[rows].copy(), bn)])
    conv = helper.make_node("Conv", [qkv.input[0], wn, bn], [out], name=base + "/conv/Conv")
    conv.attribute.extend([a for a in qkv.attribute])
    new_nodes.append((qkv, conv))
    pe.input[0] = out
    print(f"{pe.name}: nh={nh} kd={kd} hd={hd} -> pe_v Conv {W.shape[1]}->{len(rows)} on {qkv.input[0]}")

nodes = list(g.node)
for qkv, conv in new_nodes:  # keep topological order: right after the qkv conv
    nodes.insert(nodes.index(qkv) + 1, conv)
del g.node[:]
g.node.extend(nodes)
onnx.checker.check_model(m)
onnx.save(m, dst)

import onnxruntime as ort  # noqa: E402

a, b = ort.InferenceSession(src), ort.InferenceSession(dst)
inp = a.get_inputs()[0]
x = np.random.default_rng(0).random([1] + [int(d) for d in inp.shape[1:]], dtype=np.float32)
ya, yb = a.run(None, {inp.name: x}), b.run(None, {inp.name: x})
err = max(float(np.abs(p - q).max() / (np.abs(p).max() + 1e-9)) for p, q in zip(ya, yb))
print(f"saved {dst}; max relative output difference vs original {err:.2e}")
assert err < 1e-4
