"""Compile a quantized HAR to a Hailo-8 HEF (the compiler only maps/schedules; numerics are fixed by the HAR).

Usage: python compile_hef.py IN_HAR OUT_HEF [EXTRA_ALLS_FILE]
  EXTRA_ALLS_FILE: optional compiler commands, e.g. performance_param(compiler_optimization_level=max)
"""
import os
import sys
import time

from hailo_sdk_client import ClientRunner

src, dst = sys.argv[1], sys.argv[2]
extra = open(sys.argv[3]).read() if len(sys.argv) > 3 else ""
runner = ClientRunner(har=src)
if extra:
    print("compiler script:\n" + extra)
    runner.load_model_script(extra)
t = time.time()
hef = runner.compile()
with open(dst, "wb") as f:
    f.write(hef)
print(f"compiled in {time.time() - t:.0f} s -> {dst} ({os.path.getsize(dst) / 1e6:.1f} MB)")
runner.save_har(os.path.splitext(dst)[0] + "_compiled.har")
