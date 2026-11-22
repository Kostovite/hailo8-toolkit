"""Insert an identity Linear (MatMul I + Add 0) after AIFI q/k/v projections so the projection keeps a16_w16 and the
identity conv (a16_w8_a8) does the 16->8 requantisation the Hailo-8 matmul needs (signed 16->8 shortcut unsupported).
usage: python add_qkv_identity.py IN.onnx OUT.onnx"""
import sys, numpy as np, onnx
from onnx import helper, numpy_helper
m = onnx.load(sys.argv[1]); g = m.graph
cons = {}
for n in g.node:
    for i in n.input: cons.setdefault(i, []).append(n)
new = []
for n in list(g.node):
    if n.op_type == "Add" and any(f"self_attn/{p}_proj/Add" in n.name for p in "qkv"):
        out = n.output[0]; d = 256; tag = n.name.replace("/", "_") + "_id"
        I = numpy_helper.from_array(np.eye(d, dtype=np.float32), tag + "_W"); Z = numpy_helper.from_array(np.zeros(d, np.float32), tag + "_b")
        g.initializer.extend([I, Z])
        mm = helper.make_node("MatMul", [out, tag + "_W"], [tag + "_mm"], name=n.name.replace("/Add", "/IdMatMul"))
        ad = helper.make_node("Add", [Z.name, tag + "_mm"], [tag + "_out"], name=n.name.replace("/Add", "/IdAdd"))
        for c in cons[out]:
            for k, x in enumerate(c.input):
                if x == out: c.input[k] = tag + "_out"
        new.append((n, [mm, ad])); print("inserted after", n.name)
nodes = list(g.node); del g.node[:]
for n in nodes:
    g.node.append(n)
    for src, extra in new:
        if src is n: g.node.extend(extra)
onnx.checker.check_model(m); onnx.save(m, sys.argv[2])
