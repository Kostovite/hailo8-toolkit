"""Chip-only map of the softmax reciprocal: row sum S (sum of the chip's own exp outputs) -> reduce_sum_softmax1 output.
Usage: python recip_chip.py NFRAMES SCALE_LO SCALE_HI OUT.npz"""
import fcntl, sys
import numpy as np
sys.path.insert(0, "/mnt/storage/lib_project/models/Hailo_model/hailo_re/probe/silent")
import chip_vs_emu as cv
from hailo_platform import HEF, VDevice
d = "/mnt/storage/lib_project/models/Hailo_model/hailo_re/runs/det/att"
z = np.load(f"{d}/ints2.npz"); base = z["best__ew_add9"].astype(np.int64)
n, lo, hi = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
rng = np.random.default_rng(11); F = []
for k in range(n):
    a, b = base[rng.integers(16)], base[rng.integers(16)]
    perm = rng.permutation(784); m = rng.random((28, 28, 1)) < rng.uniform(0.1, 0.9)
    x = np.where(m, a, b.reshape(784, 512)[perm].reshape(28, 28, 512)) * rng.uniform(lo, hi) + rng.normal(0, rng.uniform(0, 20))
    F.append(np.clip(np.round(x), 0, 255))
X = np.stack(F).astype(np.uint8)
hef = HEF(f"{d}/E.hef"); ii = hef.get_input_vstream_infos()[0]; outs = [o.name for o in hef.get_output_vstream_infos()]
dev = "0000:03:00.0"; lk = open(f"/mnt/storage/lib_project/models/Hailo_model/hailo_re/turbo/.chip_{dev.replace(':', '_')}.lock", "w")
fcntl.flock(lk, fcntl.LOCK_EX); p = VDevice.create_params(); p.device_ids = [dev]; vd = VDevice(p)
try: c = cv.run16(vd, hef, {ii.name: X})
finally: vd.release(); fcntl.flock(lk, fcntl.LOCK_UN)
kexp = next(k for k in outs if "ew_sub" in k); ksum = next(k for k in outs if "reduce_sum" in k)
S = np.asarray(c[kexp]).astype(np.int64).reshape(n, 784, 4, 784).sum(-1).ravel()
C = np.asarray(c[ksum]).astype(np.int64).ravel()
np.savez(sys.argv[4], S=S, C=C); print("rows", S.size, "S range", S.min(), S.max(), "unique S", np.unique(S).size)
