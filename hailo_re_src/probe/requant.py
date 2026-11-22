"""Re-quantize DIR/m.onnx (calib.npy from the source case) with an extra quantization script, compile with an extra
compile script, into OUT. Usage: python requant.py SRC_CASE_DIR OUT 'quant alls' 'compile alls'"""
import os, shutil, sys, numpy as np
src, out, qtxt, ctxt = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), sys.argv[3], sys.argv[4]
os.makedirs(out, exist_ok=True); os.chdir(out); shutil.copy(f"{src}/m.onnx", "m.onnx")
from hailo_sdk_client import ClientRunner
if os.environ.get("SMX_HEADROOM"):
    sys.path.insert(0, "/mnt/storage/lib_project/models/Hailo_model/hailo_re/turbo"); import smx_headroom; smx_headroom.install(os.environ["SMX_HEADROOM"])
r = ClientRunner(hw_arch="hailo8"); r.translate_onnx_model("m.onnx", "probe")
r.load_model_script("model_optimization_flavor(optimization_level=0, compression_level=0)\n" + qtxt.replace("\\n", "\n"))
r.optimize(np.load(f"{src}/calib.npy").transpose(0, 2, 3, 1).astype(np.float32)); r.save_har("q.har")
if ctxt:
    r.load_model_script(ctxt.replace("\\n", "\n"))
open("m.hef", "wb").write(r.compile()); r.save_har("c.har"); print("OK", out)
