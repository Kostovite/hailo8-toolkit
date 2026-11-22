"""Recompile the dil3 probe with an alls variant; save HEF; chip vs emulator per column."""
import fcntl, os, sys, numpy as np
sys.path.insert(0, "/mnt/storage/lib_project/models/Hailo_model/hailo_re/probe/silent")
D = "/mnt/storage/lib_project/models/Hailo_model/hailo_re/probe/scan/out/C24_H30_W50/T_Conv_dil3"
v = sys.argv[1]; os.makedirs(v, exist_ok=True); os.chdir(v)
from hailo_sdk_client import ClientRunner
r = ClientRunner(har=f"{D}/q.har"); r.load_model_script(f"../{v}.alls")
try: hef = r.compile()
except Exception as ex: print(v, "COMPILE FAIL", str(ex)[:200]); sys.exit()
open("m.hef", "wb").write(hef)
import chip_vs_emu as cv
from hailo_platform import HEF, VDevice
h = HEF("m.hef"); X = cv.frames(h); o = h.get_output_vstream_infos()[0]; dev = "0000:03:00.0"
lk = open(f"/mnt/storage/lib_project/models/Hailo_model/hailo_re/turbo/.chip_{dev.replace(':', '_')}.lock", "w"); fcntl.flock(lk, fcntl.LOCK_EX)
p = VDevice.create_params(); p.device_ids = [dev]; vd = VDevice(p)
try:
    c = np.asarray(cv.run16(vd, h, X)[o.name]).astype(np.int64)
    c1 = np.concatenate([np.asarray(cv.run16(vd, h, {k: x[i:i + 1] for k, x in X.items()})[o.name]) for i in range(8)]).astype(np.int64)
finally: vd.release()
e = np.load(f"{D}/emu_ref_n.npz")["arr_0"]; q = np.clip(np.round(e / o.quant_info.qp_scale + o.quant_info.qp_zp), 0, 255).astype(np.int64).reshape(c.shape)
bad = c != q; cols = [w for w in range(c.shape[2]) if bad[:, :, w].mean() > 0]
print(v, "mismatch frac", round(float(bad.mean()), 4), "bad cols", cols[:3], "..", cols[-3:] if cols else [], "| batch == 1-by-1:", bool((c1.reshape(c.shape) == c).all()))
