import sys, numpy as np, json
from hailo_sdk_client import ClientRunner, InferenceContext
sys.path.insert(0, ".")
from pwl_inject import apply_pwl
r = ClientRunner(hw_arch="hailo8")
r.translate_onnx_model("pwl/sin_marked.onnx", "pwlnet")
apply_pwl(r, "pwl/sin_fit.json")
r.load_model_script("model_optimization_flavor(optimization_level=0, compression_level=0)\n")
cal = np.transpose(np.load("pwl/calib_normal.npy"), (0, 2, 3, 1))
r.optimize(cal)
x = np.linspace(-7, 7, 24 * 24 * 16, dtype=np.float32).reshape(1, 16, 24, 24).transpose(0, 2, 3, 1)
outs = {}
for t, tag in ((InferenceContext.SDK_NATIVE, "float"), (InferenceContext.SDK_QUANTIZED, "quant"), (InferenceContext.SDK_BIT_EXACT, "bitexact")):
    with r.infer_context(t) as ctx:
        outs[tag] = np.asarray(r.infer(ctx, x)).ravel()
xs = x.ravel()
for i in np.linspace(0, xs.size - 1, 15).astype(int):
    print(f"x={xs[i]:7.3f} sin={np.sin(xs[i]):7.3f} float={outs['float'][i]:7.3f} quant={outs['quant'][i]:7.3f} be={outs['bitexact'][i]:7.3f}")
m = r._sdk_backend.get_params_translated() if hasattr(r._sdk_backend, "get_params_translated") else None
p = r.get_params_translated()
for k in sorted(p.keys()):
    if "activation1" in k: print(k, np.asarray(p[k]).ravel()[:12])
