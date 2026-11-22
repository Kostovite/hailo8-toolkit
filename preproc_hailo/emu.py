"""usage: emu.py HAR IN_NPY(uint8 NHWC) OUT fp|q
One output -> OUT (.npy float32).  Several -> OUT_<i>.npy float16 NHWC, i in order of output size (largest first)."""
import sys, numpy as np
from hailo_sdk_client import ClientRunner, InferenceContext
har, src, dst, m = sys.argv[1:5]
x = np.load(src, mmap_mode="r"); r = ClientRunner(har=har)
ctx = InferenceContext.SDK_QUANTIZED if m == "q" else InferenceContext.SDK_FP_OPTIMIZED
outs = None; B = 8
with r.infer_context(ctx) as c:
    for i in range(0, len(x), B):
        y = r.infer(c, np.asarray(x[i:i + B], np.float32), batch_size=B)
        y = [np.asarray(v) for v in (y if isinstance(y, (list, tuple)) else [y])]
        if outs is None:
            order = sorted(range(len(y)), key=lambda k: -y[k][0].size)
            if len(y) == 1: outs = [np.zeros((len(x),) + y[0].shape[1:], np.float32)]
            else: outs = [np.lib.format.open_memmap(dst.replace(".npy", f"_{j}.npy"), "w+", np.float16, (len(x),) + y[k].shape[1:]) for j, k in enumerate(order)]
        for j, k in enumerate(order): outs[j][i:i + len(y[k])] = y[k]
if len(outs) == 1: np.save(dst, outs[0])
else: [o.flush() for o in outs]
print("saved", dst, [o.shape for o in outs])
