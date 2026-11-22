"""usage: quantize.py PARSED_HAR OUT_HAR CALIB_NPY MODE NORM OPT_LEVEL [EXTRA_ALLS_FILE]
MODE  int8 | all16 (conv/dense/ew a16_w16) | mix16 (16-bit except matmul/softmax paths, PP-DocLayoutV3 rule)
NORM  rgb255 | gray255 | none   (in-chip input normalisation; calib holds RAW host input in the same units)
Env INPUT16=1 also makes the input layer 16-bit (needed when the host input is a float feature vector)."""
import os, sys, time, numpy as np
from hailo_sdk_client import ClientRunner
src, dst, cal_p, mode, norm, opt = sys.argv[1:7]; extra = open(sys.argv[7]).read() if len(sys.argv) > 7 else ""
cal = np.load(cal_p).astype(np.float32); n = len(cal)
r = ClientRunner(har=src); hn = r.get_hn_model(); layers = list(hn.stable_toposort())
short = lambda l: l.name.split("/")[-1]; op = lambda l: str(l.op).lower().replace("layertype.", "")
q = ""
if mode == "all16":
    L = [short(l) for l in layers if any(t in op(l) for t in ("conv", "dense", "ew_add", "elementwise", "avgpool"))]  # avgpool: a8_w8_a16 is rejected by the compiler
    q = f"quantization_param([{', '.join(L)}], precision_mode=a16_w16)\n"
elif mode == "mix16":
    skip = ("matmul", "softmax", "normalization", "input_layer", "output_layer", "resize", "format_conversion")
    to8 = [l for l in layers if op(l) not in skip and any(op(s) in ("matmul", "softmax") for s in hn.successors(l))]
    w8 = [short(l) for l in to8 if op(l) in ("conv", "dense")]
    L = [short(l) for l in layers if op(l) not in skip and short(l) not in w8]
    q = f"quantization_param([{', '.join(L)}], precision_mode=a16_w16)\n" + (f"quantization_param([{', '.join(w8)}], precision_mode=a16_w8_a8)\n" if w8 else "")
elif mode != "int8": raise SystemExit("bad MODE " + mode)
if os.environ.get("INPUT16") == "1":
    q += "".join(f"quantization_param({short(l)}, precision_mode=a16_w16_a16)\n" for l in hn.get_input_layers())
nl = {"rgb255": "in_norm = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])\n", "gray255": "in_norm = normalization([0.0], [255.0])\n", "none": ""}[norm]
alls = f"""{nl}model_optimization_flavor(optimization_level={opt}, compression_level=0)
model_optimization_config(calibration, batch_size=8, calibset_size={n})
{q}{extra}"""
open(dst.replace(".har", ".alls"), "w").write(alls); print(alls)
r.load_model_script(alls); t = time.time(); r.optimize(cal)
print(f"optimize took {time.time() - t:.0f} s"); r.save_har(dst); print("saved", dst)
