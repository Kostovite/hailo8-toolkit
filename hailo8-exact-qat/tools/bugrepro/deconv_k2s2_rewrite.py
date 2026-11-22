"""Report 3 workaround: rewrite ConvTranspose(k=2, s=2) as Conv1x1 (C -> 4C) + DepthToSpace(2, CRD); exact in float.
Usage: python deconv_k2s2_rewrite.py IN.onnx OUT.onnx"""
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper as nh
m = onnx.load(sys.argv[1]); g = m.graph; inits = {i.name: nh.to_array(i) for i in g.initializer}
nodes = []
for n in g.node:
    a = {x.name: helper.get_attribute_value(x) for x in n.attribute}
    if n.op_type != "ConvTranspose" or list(a.get("kernel_shape", [])) != list(a.get("strides", [None])) or \
            any(a.get("pads", [0])) or a.get("group", 1) != 1:
        nodes.append(n); continue
    W = inits[n.input[1]]; cin, cout, kh, kw = W.shape
    B = inits[n.input[2]] if len(n.input) > 2 and n.input[2] in inits else np.zeros(cout, np.float32)
    wn, bn, cn = n.output[0] + "_w_d2s", n.output[0] + "_b_d2s", n.output[0] + "_c_d2s"
    g.initializer.extend([nh.from_array(W.transpose(1, 2, 3, 0).reshape(cout * kh * kw, cin, 1, 1).astype(np.float32), wn),
                          nh.from_array(np.repeat(B, kh * kw).astype(np.float32), bn)])
    nodes += [helper.make_node("Conv", [n.input[0], wn, bn], [cn], kernel_shape=[1, 1]),
              helper.make_node("DepthToSpace", [cn], [n.output[0]], blocksize=kh, mode="CRD")]
del g.node[:]; g.node.extend(nodes); onnx.save(m, sys.argv[2]); print("saved", sys.argv[2])
