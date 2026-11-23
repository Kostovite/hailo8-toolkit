import numpy as np, onnx, sys, os
from onnx import helper as oh, numpy_helper as nh, TensorProto as T
def make(d, edits):
    o, c, h = 8, 8, 16
    w = np.array([[((oi * 8 + ii) * 37 % 255 - 127) / 127 * 0.25 for ii in range(c)] for oi in range(o)], np.float32)
    w[:, 0] = 0.5  # per-channel max anchor: scales stay fixed
    for (oi, ii), v in edits.items(): w[oi, ii] = v
    W = nh.from_array(w.reshape(o, c, 1, 1), "W"); B = nh.from_array(np.zeros(o, np.float32), "B")
    n = oh.make_node("Conv", ["x", "W", "B"], ["y"], kernel_shape=[1, 1])
    g = oh.make_graph([n], "g", [oh.make_tensor_value_info("x", T.FLOAT, [1, c, h, h])],
                      [oh.make_tensor_value_info("y", T.FLOAT, [1, o, h, h])], [W, B])
    m = oh.make_model(g, opset_imports=[oh.make_opsetid("", 13)]); m.ir_version = 8
    os.makedirs(d, exist_ok=True); onnx.save(m, d + "/m.onnx"); np.save(d + "/w.npy", w)
make("A", {}); make("B", {(3, 5): 0.40}); make("C", {(6, 2): -0.40})
