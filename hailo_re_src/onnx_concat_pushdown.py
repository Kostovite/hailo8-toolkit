"""Push a 1x1 conv through a feature concat: Conv(Concat(Resize(u), s)) = Resize(Conv_u(u)) + Conv_s(s).

The skip input s (a high-resolution backbone feature) then crosses Hailo-8 context cuts as Conv_s(s), with the conv's
output width instead of s's channel count, and the upsampled branch's partial conv runs before the nearest-neighbour
Resize (a 1x1 conv commutes with it) at a quarter of the pixels. Identical math up to float summation order.

--split: build Conv_s as two output-channel halves + Concat. The Hailo DFC fuses an Add into its producing conv
(ew_add_fusing: conv with one consumer, linear activation), which would make Conv_s consume both branches and pull it
back behind the cut; a Concat between them keeps Conv_s standalone so it can run where s is produced. Exact.

Usage: python onnx_concat_pushdown.py IN.onnx OUT.onnx [--split] SKIP_TENSOR [SKIP_TENSOR ...]
"""
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper

src, dst = sys.argv[1], sys.argv[2]
split = "--split" in sys.argv[3:]
skips = [a for a in sys.argv[3:] if a != "--split"]
m = onnx.load(src)
g = m.graph
prod = {o: n for n in g.node for o in n.output}
cons = {}
for n in g.node:
    for i in n.input:
        cons.setdefault(i, []).append(n)
inits = {t.name: t for t in g.initializer}
shapes = {}
for vi in list(g.value_info) + list(g.input) + list(g.output):
    shapes[vi.name] = [d.dim_value for d in vi.type.tensor_type.shape.dim]
if not g.value_info:
    m = onnx.shape_inference.infer_shapes(m)
    g = m.graph
    for vi in g.value_info:
        shapes[vi.name] = [d.dim_value for d in vi.type.tensor_type.shape.dim]

remove, insert_after = set(), []
for skip in skips:
    cat = [c for c in cons[skip] if c.op_type == "Concat"]
    assert len(cat) == 1, (skip, [c.op_type for c in cons[skip]])
    cat = cat[0]
    assert [a.i for a in cat.attribute if a.name == "axis"] == [1], "channel concat expected"
    (conv,) = cons[cat.output[0]]
    assert conv.op_type == "Conv", conv.op_type
    W = numpy_helper.to_array(inits[conv.input[1]])
    assert W.shape[2:] == (1, 1), "1x1 conv expected"
    B = numpy_helper.to_array(inits[conv.input[2]]) if len(conv.input) > 2 else np.zeros(W.shape[0], W.dtype)
    up_t = [t for t in cat.input if t != skip]
    assert len(up_t) == 1 and len(cat.input) == 2
    up = prod[up_t[0]]
    assert up.op_type == "Resize", up.op_type
    cu = shapes[up.input[0]][1]
    cs = shapes[skip][1]
    off = list(cat.input).index(up_t[0])
    Wu = W[:, :cu] if off == 0 else W[:, cs:]
    Ws = W[:, cu:] if off == 0 else W[:, :cs]
    base = conv.name.rsplit("/", 1)[0] + "/pushdown"
    g.initializer.extend([numpy_helper.from_array(Wu.copy(), base + "/Wu"),
                          numpy_helper.from_array(Ws.copy(), base + "/Ws"),
                          numpy_helper.from_array(B.copy(), base + "/B")])
    attrs = [a for a in conv.attribute]
    n_u = helper.make_node("Conv", [up.input[0], base + "/Wu"], [base + "/u_out"], name=base + "/conv_u")
    n_u.attribute.extend(attrs)
    n_r = helper.make_node("Resize", [base + "/u_out"] + list(up.input[1:]), [base + "/u_up"], name=base + "/resize")
    n_r.attribute.extend(up.attribute)
    if split:
        h = W.shape[0] // 2
        g.initializer.extend([numpy_helper.from_array(Ws[:h].copy(), base + "/Ws0"), numpy_helper.from_array(B[:h].copy(), base + "/B0"),
                              numpy_helper.from_array(Ws[h:].copy(), base + "/Ws1"), numpy_helper.from_array(B[h:].copy(), base + "/B1")])
        s0 = helper.make_node("Conv", [skip, base + "/Ws0", base + "/B0"], [base + "/s_out0"], name=base + "/conv_s0")
        s1 = helper.make_node("Conv", [skip, base + "/Ws1", base + "/B1"], [base + "/s_out1"], name=base + "/conv_s1")
        s0.attribute.extend(attrs)
        s1.attribute.extend(attrs)
        sc = helper.make_node("Concat", [base + "/s_out0", base + "/s_out1"], [base + "/s_out"], name=base + "/s_cat", axis=1)
        n_s = [s0, s1, sc]
    else:
        n_s = [helper.make_node("Conv", [skip, base + "/Ws", base + "/B"], [base + "/s_out"], name=base + "/conv_s")]
        n_s[0].attribute.extend(attrs)
    n_a = helper.make_node("Add", [base + "/u_up", base + "/s_out"], [conv.output[0]], name=base + "/add")
    remove.add(id(conv))
    if len(cons[cat.output[0]]) == 1:
        remove.add(id(cat))
    if len(cons[up.output[0]]) == 1:
        remove.add(id(up))
    insert_after.append((conv, [n_u, n_r] + n_s + [n_a]))
    print(f"{conv.name}: concat({cu} up + {cs} skip) -> 1x1 {W.shape[0]}: skip crosses as {W.shape[0]} channels")

nodes = []
for n in g.node:
    for c, new in insert_after:
        if n is c:
            nodes.extend(new)
    if id(n) not in remove:
        nodes.append(n)
del g.node[:]
g.node.extend(nodes)
m = onnx.shape_inference.infer_shapes(m)
onnx.checker.check_model(m)
onnx.save(m, dst)

import onnxruntime as ort  # noqa: E402

x = np.random.default_rng(0).random([1] + [int(d) for d in ort.InferenceSession(src).get_inputs()[0].shape[1:]], dtype=np.float32)


def heads(path):
    """The six one2one head convs (what the Hailo HEF outputs). The end-to-end top-k decode reorders near-tied
    detections on float-rounding differences, so the model's final output is not a usable check."""
    mm = onnx.load(path)
    names = [n.output[0] for n in mm.graph.node if n.op_type == "Conv" and "one2one_cv" in n.name and n.name.endswith(".2/Conv")]
    del mm.graph.output[:]
    mm.graph.output.extend([helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None) for t in names])
    sess = ort.InferenceSession(mm.SerializeToString())
    return dict(zip(names, sess.run(None, {sess.get_inputs()[0].name: x})))


ya, yb = heads(src), heads(dst)
err = max(float(np.abs(ya[k] - yb[k]).max() / (np.abs(ya[k]).max() + 1e-9)) for k in ya)
print(f"saved {dst}; max relative head-conv difference vs original {err:.2e}")
assert err < 1e-4
