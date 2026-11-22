"""Parse -> level-0 quantize (random calibration) -> compile a micro ONNX, to read the layout the DFC chooses.

Usage (DFC env): python micro_hef.py IN.onnx OUT_PREFIX [ALLS_FILE]
Writes OUT_PREFIX.har / _q.har / .hef and prints the HN layer types.
"""
import collections
import sys

import numpy as np
from hailo_sdk_client import ClientRunner

src, pre = sys.argv[1:3]
alls = open(sys.argv[3]).read() if len(sys.argv) > 3 else ""
runner = ClientRunner(hw_arch="hailo8")
runner.translate_onnx_model(src, "micro")
hn = runner.get_hn_dict()
print("types", dict(collections.Counter(v["type"] for v in hn["layers"].values())))
for n, v in hn["layers"].items():
    if v["type"] == "format_conversion":
        p = v["params"]
        print(" ", n, p.get("conversion_type"), v["input_shapes"], "->", v["output_shapes"], "->", v["output"])
shape = hn["layers"][[n for n, v in hn["layers"].items() if v["type"] == "input_layer"][0]]["output_shapes"][0][1:]
runner.save_har(pre + ".har")
calib = np.random.default_rng(0).standard_normal([64] + list(shape)).astype(np.float32)
runner.load_model_script("model_optimization_flavor(optimization_level=0, compression_level=0)\n" + alls)
runner.optimize(calib)
runner.save_har(pre + "_q.har")
hef = runner.compile()
open(pre + ".hef", "wb").write(hef)
print("compiled", pre + ".hef")
