"""iso.py NAME H ADD Q OUT: standalone depthwise 3x3 (c64, pad 1) [+ residual add of its input], all layers at Q bits,
level-0 quantize on random calib, single-context compile. Writes OUT/NAME/{m.onnx,m.hef,res.json}."""
import json, os, sys, time, traceback
import numpy as np, onnx
from onnx import helper as oh, numpy_helper as nh, TensorProto as T
name, H, add, q, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
d = os.path.join(out, name); os.makedirs(d, exist_ok=True); os.chdir(d)
c = 64; rng = np.random.default_rng(0)
W = nh.from_array((rng.normal(0, .3, (c, 1, 3, 3))).astype(np.float32), "W"); B = nh.from_array(np.zeros(c, np.float32), "B")
nodes = [oh.make_node("Conv", ["x", "W", "B"], ["y" if add else "z"], group=c, kernel_shape=[3, 3], pads=[1, 1, 1, 1])]
if add: nodes.append(oh.make_node("Add", ["y", "x"], ["z"]))
g = oh.make_graph(nodes, "g", [oh.make_tensor_value_info("x", T.FLOAT, [1, c, H, H])], [oh.make_tensor_value_info("z", T.FLOAT, [1, c, H, H])], [W, B])
m = oh.make_model(g, opset_imports=[oh.make_opsetid("", 13)]); m.ir_version = 8; onnx.save(m, "m.onnx")
res = {"name": name, "H": H, "add": add, "q": q}; t0 = time.time()
try:
    from hailo_sdk_client import ClientRunner
    r = ClientRunner(hw_arch="hailo8"); r.translate_onnx_model("m.onnx", "iso")
    hn = r.get_hn_dict()["layers"]; res["hn"] = {k.split("/")[-1]: (v["type"], v.get("params", {}).get("elementwise_add")) for k, v in hn.items()}
    s = "model_optimization_flavor(optimization_level=0, compression_level=0)\n"
    if q == 16:
        s += "quantization_param([%s], precision_mode=a16_w16)\n" % ", ".join(k.split("/")[-1] for k, v in hn.items() if v["type"] not in ("input_layer", "output_layer"))
    r.load_model_script(s + "context_switch_param(mode=disabled)\nperformance_param(fps=20)\n")
    r.optimize(rng.normal(0, 1, (8, H, H, c)).astype(np.float32))
    hef = r.compile(); open("m.hef", "wb").write(hef); res["ok"] = True
except Exception as e:
    res.update(ok=False, error=f"{type(e).__name__}: {str(e)[:300]}")
res["s"] = round(time.time() - t0, 1); json.dump(res, open("res.json", "w"))
print(json.dumps({k: res.get(k) for k in ("name", "ok", "s", "error", "hn")}), flush=True)
