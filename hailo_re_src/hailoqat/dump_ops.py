"""Print atomic-op attributes of chosen layers straight from a quantized HAR (no inference).

Usage: python dump_ops.py HAR LAYER [LAYER ...]
"""
import sys

import numpy as np

from hailo_sdk_client import ClientRunner, InferenceContext

runner = ClientRunner(har=sys.argv[1])
model = runner._sdk_backend.build_acceleras_model(InferenceContext.SDK_QUANTIZED)
model.build(tuple([1] + list(runner.get_hn_model().get_input_layers()[0].input_shape[1:])))
for ln in sys.argv[2:]:
    layer = model.layers[ln]
    print("==", ln, type(layer).__name__)
    for op in layer.atomic_ops:
        attrs = {}
        for a in sorted(set(dir(op))):
            if a.startswith("__") or a in ("trainable_weights", "weights", "variables", "non_trainable_weights"):
                continue
            try:
                v = getattr(op, a)
            except Exception:
                continue
            if callable(v):
                continue
            try:
                arr = np.asarray(v)
            except Exception:
                continue
            if arr.dtype.kind in "biuf" and arr.size <= 8:
                attrs[a] = arr.tolist()
        el = [(e.bits, e.signed, getattr(e, "wraparound", None), type(e).__name__)
              for e in getattr(op, "output_lossy_elements", []) if getattr(e, "bits", None) is not None]
        print("  ", op.full_name.split("/")[-1], type(op).__name__, "out_lossy", el)
        print("     ", {k: v for k, v in attrs.items() if any(t in k for t in ("shift", "factor", "repeat", "zp", "zero", "bits", "kernel"))})
