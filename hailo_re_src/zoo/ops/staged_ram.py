"""StagedRunner RAM budget on the real chip: planned vs measured peak host RAM, and FPS, per ram_mb.

Usage (hailo env): python staged_ram.py CASE_DIR MB [MB ...]   (streams 2048 frames through run_iter)
"""
import json
import resource
import sys
import time

import numpy as np

from host_split import StagedRunner

d = sys.argv[1]
mb = float(sys.argv[2])
gen = np.random.default_rng(0)
frame = (gen.random((224, 224, 3)) * 255).astype(np.float32)


def frames(n=2048):  # a camera: one frame at a time, nothing held
    for _ in range(n):
        yield frame


r = StagedRunner(f"{d}/part1.hef", f"{d}/host.onnx", f"{d}/part2.hef", ram_mb=mb)
for _ in r.run_iter(frames(r.K)):  # warm-up (sessions, buffers)
    pass
base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
t = time.perf_counter()
n = sum(len(y) for y in r.run_iter(frames()))
fps = n / (time.perf_counter() - t)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
r.close()
print(json.dumps({"ram_mb": mb, **r.mem_plan, "fps": round(fps), "peak_rss_mb": round(peak),
                  "rss_after_warmup_mb": round(base)}))
