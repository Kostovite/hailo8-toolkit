"""usage: OPT_LEVEL=0 python quantize.py PARSED_HAR OUT_HAR CALIB_NPY MODE [EXTRA_ALLS]
MODE: int8 | all16 (every conv/dense/ew layer a16_w16) | out16 (output conv a8_w8_a16).  Calib = uint8 HWC (make_calib.py)."""
import os, sys, time, numpy as np
from hailo_sdk_client import ClientRunner
src, dst, cal_p, mode = sys.argv[1:5]; extra = open(sys.argv[5]).read() if len(sys.argv) > 5 else ""
cal = np.load(cal_p).astype(np.float32); n = len(cal)
r = ClientRunner(har=src); hn = r.get_hn_model()
layers = list(hn.stable_toposort())
def short(l): return l.name.split("/")[-1]
q = ""
if mode in ("all16", "full16"):
    T = ("conv", "dense", "ew_add", "elementwise") + (("dw", "layer_normalization", "concat", "ew_mult", "ew_sub") if mode == "full16" else ())
    L = [short(l) for l in layers if any(t in str(l.op).lower().replace("layertype.", "") for t in T)]
    q = f"quantization_param([{', '.join(L)}], precision_mode=a16_w16)\n"
elif mode == "mix16":
    # everything 16-bit except matmul/softmax (DFC: 8-bit only); convs feeding a matmul -> a16_w8 (emits 8-bit natively,
    # avoids the unsupported signed a16_w16_a8 precision_change)
    skip = ("matmul", "softmax", "normalization", "input_layer", "output_layer", "resize", "format_conversion")
    op = lambda l: str(l.op).lower().replace("layertype.", "")
    to8 = [l for l in layers if op(l) not in skip and any(op(s) in ("matmul", "softmax") for s in hn.successors(l))]
    w8 = [short(l) for l in to8 if op(l) in ("conv", "dense")]
    w8 = list(dict.fromkeys(w8 + ["conv98"]))
    L = [short(l) for l in layers if op(l) not in skip and short(l) not in w8 and short(l) not in ("ew_add22",)]
    q = f"quantization_param([{', '.join(L)}], precision_mode=a16_w16)\nquantization_param([{', '.join(w8)}], precision_mode=a16_w8_a8)\n"
elif mode == "out16":
    outs = [short(l) for l in hn.get_output_layers()]; pre = [short(p) for o in hn.get_output_layers() for p in hn.predecessors(o)]
    q = f"quantization_param([{', '.join(pre)}], precision_mode=a8_w8_a16)\n"
alls = f"""in_norm = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])
model_optimization_flavor(optimization_level={os.environ.get("OPT_LEVEL", "0")}, compression_level=0)
model_optimization_config(calibration, batch_size=8, calibset_size={n})
{q}{extra}"""
print(alls); print("ops:", sorted({str(l.op) for l in layers}))
r.load_model_script(alls); t = time.time(); r.optimize(cal)
print(f"optimize took {time.time() - t:.0f} s"); r.save_har(dst); print("saved", dst)
