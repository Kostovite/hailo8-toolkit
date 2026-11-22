"""usage: emu.py HAR IN_NPY OUT_PREFIX fp|q START END
Emulates rows [START,END) of IN_NPY (NHWC, host units). Writes OUT_PREFIX_<i>.npy float32, outputs ordered by size (largest first)."""
import os, sys, numpy as np
from hailo_sdk_client import ClientRunner, InferenceContext
har, src, pre, m, s, e = sys.argv[1:7]; s, e = int(s), int(e)
x = np.load(src, mmap_mode="r"); r = ClientRunner(har=har)
ctx = InferenceContext.SDK_QUANTIZED if m == "q" else InferenceContext.SDK_FP_OPTIMIZED
outs = order = None; B = 8
with r.infer_context(ctx) as c:
    for i in range(s, e, B):
        xb = np.asarray(x[i:min(i + B, e)], np.float32)
        y = r.infer(c, xb, batch_size=len(xb)); y = [np.asarray(v) for v in (y if isinstance(y, (list, tuple)) else [y])]
        if outs is None:
            sz = [v[0].size for v in y]; assert len(set(sz)) == len(sz), f"outputs need distinct sizes for ordering: {sz}"
            order = sorted(range(len(y)), key=lambda k: -sz[k])
            outs = [np.lib.format.open_memmap(f"{pre}_{j}.npy", "w+", np.float16 if os.environ.get("EMU_F16") == "1" else np.float32, (e - s,) + y[k].shape[1:]) for j, k in enumerate(order)]
        for j, k in enumerate(order): outs[j][i - s:i - s + len(y[k])] = y[k]
[o.flush() for o in outs]; print("saved", pre, [o.shape for o in outs])
