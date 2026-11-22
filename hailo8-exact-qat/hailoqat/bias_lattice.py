"""Add each layer's storable-bias lattice to its capture record (bias_factor, bias_repeats, bias_imax).

a16_w16 layers store the 32-bit accumulator bias as float32(float32(factor * i) * repeats) with a 16-bit signed i
(acceleras BiasQuantElement, single decomposition; HSim rounds both products to float32); factor / repeats are fixed
when the HAR is quantized. QAT must train i, not the accumulator value, or write-back cannot store the result. Layers without decomposition get no keys.

Usage (DFC env): python bias_lattice.py HAR CAPTURE_DIR
"""
import glob
import os
import sys

import numpy as np

from hailo_sdk_client import ClientRunner, InferenceContext

har, cap = sys.argv[1:3]
runner = ClientRunner(har=har)
model = runner._sdk_backend.build_acceleras_model(InferenceContext.SDK_QUANTIZED)
model.build(tuple([1] + list(runner.get_hn_model().get_input_layers()[0].input_shape[1:])))
done = 0
for path in sorted(glob.glob(os.path.join(cap, "*.npz"))):
    lname = os.path.basename(path)[:-4].replace("__", "/")
    layer = model.layers.get(lname)
    if layer is None or not hasattr(layer, "bias_add_op"):
        continue
    e = layer.bias_add_op.weight_lossy_elements.bias_decompose
    ndec = getattr(e, "num_decomposition", 0)
    if not ndec:
        continue
    if not (ndec == 1 and float(e.pre_acc_shift) == 0):
        print(f"skip {lname}: decomposition {ndec} shift {e.pre_acc_shift} (frozen-bias chain in qatfast)")
        continue
    factor, repeats = float(np.asarray(e.factors[0])), float(e.repeats)
    imax = 2 ** (int(e.int_bits) - 1) - 1
    rec = dict(np.load(path, allow_pickle=True))
    b = np.asarray(rec["bias"], np.float64).reshape(-1)
    # acceleras reconstructs f32(f32(factor * i) * repeats): both products are rounded to float32 (HSim bit-exact)
    i = np.round(b / (factor * repeats))
    recon = ((np.float32(factor) * i.astype(np.float32)) * np.float32(repeats)).astype(np.float64)
    if not np.array_equal(recon, b):  # another bias mode: leave unannotated, qatfast runs it as a frozen chain
        print(f"skip {lname}: captured bias not f32(f32(factor*i)*repeats) at {np.flatnonzero(recon != b)[:4]}")
        continue
    rec["bias_factor"], rec["bias_repeats"], rec["bias_imax"] = np.float64(factor), np.float64(repeats), np.int64(imax)
    rec.pop("bias_step", None)
    np.savez(path, **rec)
    done += 1
    print(f"{lname}: bias = f32(f32({factor:.0f} * i) * {repeats:.0f}), |i| <= {imax}")
print(f"{done} layers annotated in {cap}")
