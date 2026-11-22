"""Recompile DIR/q.har with an extra model-script text into OUT (HEF + emulator ref copied): for chip A/B tests.
Usage: python recomp.py DIR OUT 'alls text'"""
import os, shutil, sys
d, out, txt = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), sys.argv[3]
os.makedirs(out, exist_ok=True); os.chdir(out)
from hailo_sdk_client import ClientRunner
r = ClientRunner(har=f"{d}/q.har"); r.load_model_script(txt.replace("\\n", "\n"))
open("extra.alls", "w").write(txt)
hef = r.compile(); open("m.hef", "wb").write(hef); r.save_har("c.har")
for f in ("q.har", "m.onnx", "emu_ref_n.npz", "emu_ref.npz", "result.json"):
    if os.path.exists(f"{d}/{f}") and not os.path.exists(f):
        shutil.copy(f"{d}/{f}", f)
print("OK", out)
