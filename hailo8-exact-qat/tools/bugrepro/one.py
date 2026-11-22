import fcntl, sys, io, tarfile
import numpy as np
sys.path.insert(0, "/mnt/storage/lib_project/models/Hailo_model/hailo_re/probe/silent")
import chip_vs_emu as cv
from hailo_platform import HEF, VDevice
d = "/mnt/storage/lib_project/models/Hailo_model/hailo_re/runs/det/att"
z = np.load(f"{d}/ints2.npz"); nb = int(z["nbad"])
hef = HEF(f"{d}/E.hef"); dev = "0000:03:00.0"
lk = open(f"/mnt/storage/lib_project/models/Hailo_model/hailo_re/turbo/.chip_{dev.replace(':', '_')}.lock", "w"); fcntl.flock(lk, fcntl.LOCK_EX)
p = VDevice.create_params(); p.device_ids = [dev]; vd = VDevice(p)
try: out = cv.run16(vd, hef, {hef.get_input_vstream_infos()[0].name: z["best__ew_add9"].astype(np.uint8)})
finally: vd.release()
o = [x for x in hef.get_output_vstream_infos() if "reduce_sum" in x.name][0]
C = np.asarray(out[o.name]).astype(np.int64).reshape(16, 784, 4); Ez = z["best__reduce_sum_softmax1"].astype(np.int64).reshape(16, 784, 4)
S = z["best__ew_sub_softmax1"].astype(np.int64).reshape(16, 784, 4, 784)  # exp values per (query, head, key)
raw = S.sum(-1)
print("reduce_sum out qp", o.quant_info.qp_zp, o.quant_info.qp_scale)
for i in range(16):
    w = np.argwhere(C[i] != Ez[i])
    for q, h in w: print("bad " if i < nb else "good", i, "q", q, "h", h, "chip", C[i, q, h], "emu", Ez[i, q, h], "raw sum", raw[i, q, h])
r = raw.ravel(); print("raw sum over all rows: min", r.min(), "max", r.max(), " rows with raw>=65536:", int((r >= 65536).sum()), " >=32768:", int((r >= 32768).sum()))
bad_raw = [raw[i][C[i] != Ez[i]][0] for i in range(nb)]; print("bad raw sums sorted:", sorted(bad_raw))
# neighbours: values of raw near the bad ones on other rows -> does the same raw value always fail?
for v in sorted(bad_raw):
    m = (raw == v); print(f"raw {v}: rows with this sum {int(m.sum())}, of which wrong {int((m & (C != Ez)).sum())}")
for v in range(32764, 32776):
    m = raw == v
    if m.any(): print(f"sum {v}: APU in ~{(-2 * v + 2) >> 1}: rows {int(m.sum())}, wrong {int((m & (C != Ez)).sum())}, chip {sorted(set(C[m].tolist()))} emu {sorted(set(Ez[m].tolist()))}")
big = raw >= 32771; print("rows with sum >= 32771:", int(big.sum()), "wrong:", int((big & (C != Ez)).sum()), "emu values there:", sorted(set(Ez[big].tolist()))[:8])
