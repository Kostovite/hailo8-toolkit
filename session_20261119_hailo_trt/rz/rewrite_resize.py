"""Replace half_pixel bilinear 2x Resize -> Pad(edge,1) + depthwise ConvTranspose(k4,s2,pads3); drop same-size linear Resize.
usage: rewrite_resize.py IN.onnx OUT.onnx [edge|concat]"""
import onnx, sys, numpy as np
from onnx import helper, numpy_helper, shape_inference
src, dst = sys.argv[1:3]; padmode = sys.argv[3] if len(sys.argv) > 3 else "edge"
m = shape_inference.infer_shapes(onnx.load(src)); g = m.graph
vi = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim] for v in list(g.value_info) + list(g.input) + list(g.output)}
k1 = np.array([.25, .75, .75, .25], np.float32); K = np.outer(k1, k1)
new, rename, n_up, n_id = [], {}, 0, 0
for n in g.node:
    n.input[:] = [rename.get(i, i) for i in n.input]
    a = {x.name: helper.get_attribute_value(x) for x in n.attribute}
    if n.op_type == "Resize" and a.get("mode") == b"linear":
        si, so = vi[n.input[0]], vi[n.output[0]]
        if si == so: rename[n.output[0]] = n.input[0]; n_id += 1; continue
        assert so[2] == 2 * si[2] and so[3] == 2 * si[3] and a["coordinate_transformation_mode"] == b"half_pixel", (n.name, si, so)
        C, p = si[1], n.name.replace("/", "_")
        if padmode == "nnz":   # nearest x2 + dw3x3 ZERO pad: exact except the 1-px border (edge term .25*x lost)
            sc = numpy_helper.from_array(np.array([1, 1, 2, 2], np.float32), p + "_sc"); g.initializer.append(sc)
            new.append(helper.make_node("Resize", [n.input[0], "", sc.name], [p + "_nn"], mode="nearest", coordinate_transformation_mode="asymmetric", nearest_mode="floor", name=p + "_nn"))
            k3 = np.array([.25, .5, .25], np.float32)
            w = numpy_helper.from_array(np.tile(np.outer(k3, k3)[None, None], (C, 1, 1, 1)), p + "_dw"); g.initializer.append(w)
            new.append(helper.make_node("Conv", [p + "_nn", w.name], [n.output[0]], group=C, kernel_shape=[3, 3], pads=[1, 1, 1, 1], name=p + "_dwc"))
            n_up += 1; continue
        if padmode == "nn2":   # edge-pad the SMALL map (concat), nearest x2, dw3x3 valid, crop 1
            x = n.input[0]; H, W = si[2], si[3]
            def sl(inp, s, e, ax, out):
                t = [numpy_helper.from_array(np.array([v], np.int64), f"{out}_{nm}") for nm, v in (("s", s), ("e", e), ("a", ax))]
                g.initializer.extend(t); new.append(helper.make_node("Slice", [inp] + [q.name for q in t], [out], name=out))
            sl(x, 0, 1, 2, p + "_t"); sl(x, H - 1, H, 2, p + "_b")
            new.append(helper.make_node("Concat", [p + "_t", x, p + "_b"], [p + "_h"], axis=2, name=p + "_h"))
            sl(p + "_h", 0, 1, 3, p + "_l"); sl(p + "_h", W - 1, W, 3, p + "_r")
            new.append(helper.make_node("Concat", [p + "_l", p + "_h", p + "_r"], [p + "_pad"], axis=3, name=p + "_pad"))
            sc = numpy_helper.from_array(np.array([1, 1, 2, 2], np.float32), p + "_sc"); g.initializer.append(sc)
            new.append(helper.make_node("Resize", [p + "_pad", "", sc.name], [p + "_nn"], mode="nearest", coordinate_transformation_mode="asymmetric", nearest_mode="floor", name=p + "_nn"))
            k3 = np.array([.25, .5, .25], np.float32)
            w = numpy_helper.from_array(np.tile(np.outer(k3, k3)[None, None], (C, 1, 1, 1)), p + "_dw"); g.initializer.append(w)
            new.append(helper.make_node("Conv", [p + "_nn", w.name], [p + "_dwo"], group=C, kernel_shape=[3, 3], pads=[0, 0, 0, 0], name=p + "_dwc"))
            t = [numpy_helper.from_array(np.array(v, np.int64), f"{p}_crop{k}") for k, v in (("s", [1, 1]), ("e", [2 * H + 1, 2 * W + 1]), ("a", [2, 3]))]
            g.initializer.extend(t); new.append(helper.make_node("Slice", [p + "_dwo"] + [q.name for q in t], [n.output[0]], name=p + "_crop"))
            n_up += 1; continue
        if padmode == "nn":
            sc = numpy_helper.from_array(np.array([1, 1, 2, 2], np.float32), p + "_sc"); g.initializer.append(sc)
            new.append(helper.make_node("Resize", [n.input[0], "", sc.name], [p + "_nn"], mode="nearest", coordinate_transformation_mode="asymmetric", nearest_mode="floor", name=p + "_nn"))
            x = p + "_nn"; H, W = 2 * si[2], 2 * si[3]
            def sl(inp, s, e, ax, out):
                t = [numpy_helper.from_array(np.array([v], np.int64), f"{out}_{nm}") for nm, v in (("s", s), ("e", e), ("a", ax))]
                g.initializer.extend(t); new.append(helper.make_node("Slice", [inp] + [q.name for q in t], [out], name=out))
            sl(x, 0, 1, 2, p + "_t"); sl(x, H - 1, H, 2, p + "_b")
            new.append(helper.make_node("Concat", [p + "_t", x, p + "_b"], [p + "_h"], axis=2, name=p + "_h"))
            sl(p + "_h", 0, 1, 3, p + "_l"); sl(p + "_h", W - 1, W, 3, p + "_r")
            new.append(helper.make_node("Concat", [p + "_l", p + "_h", p + "_r"], [p + "_pad"], axis=3, name=p + "_pad"))
            k3 = np.array([.25, .5, .25], np.float32)
            w = numpy_helper.from_array(np.tile(np.outer(k3, k3)[None, None], (C, 1, 1, 1)), p + "_dw"); g.initializer.append(w)
            new.append(helper.make_node("Conv", [p + "_pad", w.name], [n.output[0]], group=C, kernel_shape=[3, 3], pads=[0, 0, 0, 0], name=p + "_dwc"))
            n_up += 1; continue
        w = numpy_helper.from_array(np.tile(K[None, None], (C, 1, 1, 1)), p + "_upw"); g.initializer.append(w)
        if padmode == "edge":
            pads = numpy_helper.from_array(np.array([0, 0, 1, 1, 0, 0, 1, 1], np.int64), p + "_pads"); g.initializer.append(pads)
            new.append(helper.make_node("Pad", [n.input[0], p + "_pads"], [p + "_pad"], mode="edge", name=p + "_pad"))
        else:  # concat of edge slices
            x = n.input[0]; H, W = si[2], si[3]
            def sl(inp, s, e, ax, out):
                t = [numpy_helper.from_array(np.array([v], np.int64), f"{out}_{nm}") for nm, v in (("s", s), ("e", e), ("a", ax))]
                g.initializer.extend(t); new.append(helper.make_node("Slice", [inp] + [q.name for q in t], [out], name=out))
            sl(x, 0, 1, 2, p + "_t"); sl(x, H - 1, H, 2, p + "_b")
            new.append(helper.make_node("Concat", [p + "_t", x, p + "_b"], [p + "_h"], axis=2, name=p + "_h"))
            sl(p + "_h", 0, 1, 3, p + "_l"); sl(p + "_h", W - 1, W, 3, p + "_r")
            new.append(helper.make_node("Concat", [p + "_l", p + "_h", p + "_r"], [p + "_pad"], axis=3, name=p + "_pad"))
        new.append(helper.make_node("ConvTranspose", [p + "_pad", w.name], [p + "_dc"], group=C, kernel_shape=[4, 4], strides=[2, 2], pads=[1, 1, 1, 1], name=p + "_up"))
        H2, W2 = 2 * si[2], 2 * si[3]   # deconv out = 2(n+2); keep [2 : 2n+2]
        t = [numpy_helper.from_array(np.array(v, np.int64), f"{p}_crop{k}") for k, v in (("s", [2, 2]), ("e", [H2 + 2, W2 + 2]), ("a", [2, 3]))]
        g.initializer.extend(t); new.append(helper.make_node("Slice", [p + "_dc"] + [q.name for q in t], [n.output[0]], name=p + "_crop"))
        n_up += 1; continue
    new.append(n)
for o in g.output: assert o.name not in rename
del g.node[:]; g.node.extend(new); del g.value_info[:]
onnx.checker.check_model(m); onnx.save(m, dst); print("upsample", n_up, "identity removed", n_id, "->", dst)
