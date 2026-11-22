"""Split a wide Conv+SiLU into two output-channel halves and rewrite its Conv consumers as sums of partial convs.

  y = SiLU(Conv(x, W))            ->  a = SiLU(Conv(x, W[:h])),  b = SiLU(Conv(x, W[h:]))
  z = Conv(y, V, c)               ->  z = Conv(a, V[:, :h], c) + Conv(b, V[:, h:])

Exact (SiLU is per-channel, a conv is linear in its input channels). On Hailo-8 a wide high-resolution tensor is
produced as several cluster outputs that the compiler concatenates back into ONE stream before its consumers; at
~1.7 B/cycle per stream the YOLO26m@896 P3 tensor (112x112x512) alone costs ~9-11 ms/frame. After the split no
consumer needs the full-width tensor, so the halves stream in parallel.

Usage: python onnx_channel_split.py IN.onnx OUT.onnx TENSOR [TENSOR ...]   (TENSOR = the SiLU Mul output)
"""
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper

src, dst, targets = sys.argv[1], sys.argv[2], sys.argv[3:]
m = onnx.load(src)
g = m.graph
prod = {o: n for n in g.node for o in n.output}
inits = {t.name: t for t in g.initializer}


def consumers(t):
    return [n for n in g.node if t in n.input]


new_after, remove = [], set()
for t in targets:
    mul = prod[t]
    assert mul.op_type == "Mul", mul.op_type
    conv = next(prod[i] for i in mul.input if i in prod and prod[i].op_type == "Conv")
    sig = next(prod[i] for i in mul.input if i in prod and prod[i].op_type == "Sigmoid")
    W = numpy_helper.to_array(inits[conv.input[1]])
    B = numpy_helper.to_array(inits[conv.input[2]]) if len(conv.input) > 2 else np.zeros(W.shape[0], W.dtype)
    h = W.shape[0] // 2
    base = conv.name.rsplit("/", 1)[0] + "/csplit"
    halves = []
    nodes = []
    for k, sl in enumerate((slice(0, h), slice(h, None))):
        wn, bn = f"{base}/W{k}", f"{base}/B{k}"
        g.initializer.extend([numpy_helper.from_array(W[sl].copy(), wn), numpy_helper.from_array(B[sl].copy(), bn)])
        c = helper.make_node("Conv", [conv.input[0], wn, bn], [f"{base}/c{k}"], name=f"{base}/conv{k}")
        c.attribute.extend(conv.attribute)
        s = helper.make_node("Sigmoid", [f"{base}/c{k}"], [f"{base}/s{k}"], name=f"{base}/sig{k}")
        mm = helper.make_node("Mul", [f"{base}/c{k}", f"{base}/s{k}"], [f"{base}/y{k}"], name=f"{base}/mul{k}")
        nodes += [c, s, mm]
        halves.append(f"{base}/y{k}")
    new_after.append((mul, nodes))
    remove |= {id(conv), id(sig), id(mul)}
    for cons in consumers(t):
        if cons.op_type != "Conv" or [a.i for a in cons.attribute if a.name == "group"] not in ([], [1]):
            # anything else keeps reading the full tensor through a concat
            cat = helper.make_node("Concat", halves, [t], name=f"{base}/cat", axis=1)
            new_after.append((mul, [cat]))
            remove.discard(id(mul))
            remove.add(id(mul))
            print(f"  {cons.name} ({cons.op_type}) reads the concatenated tensor")
            continue
        V = numpy_helper.to_array(inits[cons.input[1]])
        cb = numpy_helper.to_array(inits[cons.input[2]]) if len(cons.input) > 2 else np.zeros(V.shape[0], V.dtype)
        cbase = cons.name + "/csplit_in"
        g.initializer.extend([numpy_helper.from_array(V[:, :h].copy(), cbase + "/V0"),
                              numpy_helper.from_array(V[:, h:].copy(), cbase + "/V1"),
                              numpy_helper.from_array(cb.copy(), cbase + "/b")])
        p0 = helper.make_node("Conv", [halves[0], cbase + "/V0", cbase + "/b"], [cbase + "/p0"], name=cbase + "/conv0")
        p1 = helper.make_node("Conv", [halves[1], cbase + "/V1"], [cbase + "/p1"], name=cbase + "/conv1")
        p0.attribute.extend(cons.attribute)
        p1.attribute.extend(cons.attribute)
        add = helper.make_node("Add", [cbase + "/p0", cbase + "/p1"], [cons.output[0]], name=cbase + "/add")
        new_after.append((cons, [p0, p1, add]))
        remove.add(id(cons))
        print(f"  {cons.name}: {V.shape} -> two partial convs over {h} + {V.shape[1] - h} channels")
    print(f"{t}: {W.shape[0]} channels -> 2 x {h}")

nodes = []
for n in g.node:
    for anchor, new in new_after:
        if n is anchor:
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
    """The six one2one head convs (what the HEF outputs); the end-to-end top-k decode is not a usable check."""
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
