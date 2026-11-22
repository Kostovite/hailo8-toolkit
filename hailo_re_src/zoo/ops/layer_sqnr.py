"""Per-layer SQNR of a quantized HAR: bit-exact HSim output of every HN layer (dequantized with its output scale /
zero point) vs the float ONNX tensor it was translated from (onnxruntime). Finds the layer that costs the accuracy.

Usage (DFC env): python layer_sqnr.py Q.har M.onnx INPUT.npy [OUT.json]
  INPUT.npy: one NCHW float input (as fed to the ONNX).  Prints layers in graph order with SQNR (dB).
"""
import json
import sys

import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf
from hailo_model_optimization.acceleras.atomic_ops.base_atomic_op import BaseAtomicOp
from hailo_model_optimization.flows.inference_flow import SimulationInferenceModel
from hailo_sdk_client import ClientRunner, InferenceContext

har, onnx_path, inp = sys.argv[1:4]
out_json = sys.argv[4] if len(sys.argv) > 4 else None
x = np.load(inp).astype(np.float32)
if x.ndim == 3:
    x = x[None]

# float reference: every ONNX tensor
m = onnx.load(onnx_path)
have = {o.name for o in m.graph.output}
for n in m.graph.node:
    for o in n.output:
        if o not in have:
            m.graph.output.append(onnx.helper.make_empty_tensor_value_info(o))
sess = ort.InferenceSession(m.SerializeToString())
ref = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, {sess.get_inputs()[0].name: x})))
node_out = {n.name: n.output[0] for n in m.graph.node}

# bit-exact: last atomic op output of every layer, with its output scale / zero point
last = {}
orig = BaseAtomicOp._bit_exact_run


def hook(self, inputs, training, **kw):
    out = orig(self, inputs, training, **kw)
    ln = self.full_name.rsplit("/", 1)[0]
    try:
        o = out[0] if isinstance(out, (list, tuple)) else out
        last[ln] = (np.asarray(o.numpy() if hasattr(o, "numpy") else o), self)
    except Exception:  # noqa: BLE001
        pass
    return out


BaseAtomicOp._bit_exact_run = hook
tf.config.run_functions_eagerly(True)
runner = ClientRunner(har=har)
hn = runner.get_hn_dict()
model = runner._sdk_backend.build_acceleras_model(InferenceContext.SDK_BIT_EXACT)
flow = SimulationInferenceModel(model)
flow.set_quantized()
flow.set_native(False)
flow.set_bit_exact(True)
xin = np.transpose(x, (0, 2, 3, 1))
flow.run(tf.data.Dataset.from_tensor_slices((xin, np.zeros(len(xin), np.float32))), 1, len(xin))

rows = []
for name, L in hn["layers"].items():
    if name not in last or L["type"] in ("input_layer", "output_layer"):
        continue
    t, op = last[name]
    # bit-exact ops return integers: dequantize with the op's output scale / zero point (per channel)
    sc = np.asarray(op.output_scale.numpy() if hasattr(op.output_scale, "numpy") else op.output_scale, np.float64)
    zp = np.asarray(op.output_zero_point.numpy() if hasattr(op.output_zero_point, "numpy") else op.output_zero_point,
                    np.float64)
    yq = (t.astype(np.float64) - zp) * sc
    cands = [node_out[o] for o in L.get("original_names", []) if o in node_out]
    # the layer's output is one of its original nodes' outputs, in some axis order (HN is NHWC; 3-D attention
    # tensors map to other layouts): try every candidate tensor x axis permutation that fits, keep the best match
    import itertools
    best = None
    for c in cands:
        r = ref[c].astype(np.float64)
        if r.size != yq.size:
            continue
        for perm in itertools.permutations(range(r.ndim)):
            rp = np.transpose(r, perm).reshape(yq.shape)
            e = ((rp - yq) ** 2).sum()
            s_ = float(10 * np.log10((rp ** 2).sum() / max(e, 1e-30)))
            if best is None or s_ > best[2]:
                best = (c, perm, s_)
    if best is None:
        continue
    s = best[2]
    rows.append({"layer": name.split("/")[-1], "type": L["type"], "onnx": best[0], "sqnr_db": s,
                 "act": L.get("params", {}).get("activation"),
                 "prec": L.get("quantization_params", {}).get("precision_mode")})
for r_ in rows:
    print(f'{r_["layer"]:32s} {r_["type"]:20s} {str(r_["prec"]):12s} {r_["sqnr_db"]:7.1f} dB   <- {r_["onnx"]}')
if out_json:
    json.dump(rows, open(out_json, "w"), indent=1)
