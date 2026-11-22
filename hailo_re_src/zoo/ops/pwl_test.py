"""Check a pwl_inject'ed model: parse marked ONNX -> apply_pwl -> level-0 quantize -> bit-exact HSim vs onnxruntime
float of the ORIGINAL model; optional compile.

Usage (DFC env): python pwl_test.py ORIG.onnx MARKED.onnx PWL.json CALIB.npy TEST.npy [--compile]
"""
import sys

import numpy as np
import onnxruntime as ort
from hailo_sdk_client import ClientRunner, InferenceContext

from pwl_inject import apply_pwl

orig, marked, pj, cal, tst = sys.argv[1:6]
calib, test = np.load(cal).astype(np.float32), np.load(tst).astype(np.float32)
sess = ort.InferenceSession(orig)
ref = np.concatenate([sess.run(None, {sess.get_inputs()[0].name: t[None]})[0] for t in test])
r = ClientRunner(hw_arch="hailo8")
r.translate_onnx_model(marked, "pwlnet")
print("pwl layers:", apply_pwl(r, pj))
hn = r.get_hn_dict()
pw = [n.split("/")[-1] for n, v in hn["layers"].items() if v["type"] == "activation" and v["params"].get("activation") == "pwl"]
tail = [n.split("/")[-1] for n, v in hn["layers"].items() if v["type"] in ("ew_add", "ew_sub")
        and any(i.split("/")[-1] in pw for i in v["input"])]
mode = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--mode=")), None)
script = "model_optimization_flavor(optimization_level=0, compression_level=0)\n"
if mode:
    script += f"quantization_param([{', '.join(pw + tail)}], precision_mode={mode})\n"
print("model script:", script.strip().replace(chr(10), " | "))
r.load_model_script(script)
nhwc = lambda a: np.transpose(a, (0, 2, 3, 1))  # noqa: E731
r.optimize(nhwc(calib))
sq = lambda a, b: 10 * np.log10((a ** 2).sum() / ((a - b) ** 2).sum())  # noqa: E731
for ctx_t, tag in ((InferenceContext.SDK_NATIVE, "float"), (InferenceContext.SDK_BIT_EXACT, "hw")):
    with r.infer_context(ctx_t) as ctx:
        y = np.asarray(r.infer(ctx, nhwc(test)))
    print(f"SQNR {tag} vs original: {sq(nhwc(ref), y.reshape(nhwc(ref).shape)):.1f} dB")
if "--compile" in sys.argv:
    hef = r.compile()
    print("compiled", len(hef) / 1e6, "MB")
