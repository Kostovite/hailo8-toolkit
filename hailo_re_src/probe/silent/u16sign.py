"""Is a raw UINT16 input value >= 32768 read as negative on the chip? Pixels whose channels are all <= 32767 vs
pixels with one channel >= 32768; chip vs SDK_BIT_EXACT mismatch rate per group, and chip vs emulator-with-the-
input-wrapped-to-int16."""
import fcntl, sys
import numpy as np
sys.path.insert(0, "/mnt/storage/lib_project/models/Hailo_model/hailo_re/probe/silent")
import chip_vs_emu as cv
from hailo_platform import HEF, VDevice
from hailo_sdk_client import ClientRunner, InferenceContext
d = sys.argv[1]; dev = "0000:03:00.0"
hef = HEF(f"{d}/m.hef"); i = hef.get_input_vstream_infos()[0]; o = hef.get_output_vstream_infos()[0]; q = i.quant_info
rng = np.random.default_rng(1)
X = rng.integers(int(q.qp_zp) - 4000, int(q.qp_zp) + 4000, (8, *i.shape)).astype(np.int64)
hot = rng.random((8, *i.shape[:-1])) < 0.3
hot[:4] = False  # frames 0-3 entirely cold  # 30% of pixels get one channel pushed above 32767
X[..., 0] = np.where(hot, 33000 + rng.integers(0, 4000, hot.shape), X[..., 0]); X = X.astype(np.uint16)
r = ClientRunner(har=f"{d}/q.har")
def emu(Xr):
    with r.infer_context(InferenceContext.SDK_BIT_EXACT) as ctx:
        e = np.asarray(r.infer(ctx, (Xr.astype(np.float32) - q.qp_zp) * q.qp_scale))
    return np.clip(np.round(e / o.quant_info.qp_scale + o.quant_info.qp_zp), 0, 65535).astype(np.int64)
lk = open(f"/mnt/storage/lib_project/models/Hailo_model/hailo_re/turbo/.chip_{dev.replace(':', '_')}.lock", "w"); fcntl.flock(lk, fcntl.LOCK_EX)
p = VDevice.create_params(); p.device_ids = [dev]; vd = VDevice(p)
try: c = np.asarray(next(iter(cv.run16(vd, hef, {i.name: X}).values()))).astype(np.int64)
finally: vd.release()
E1 = emu(X).reshape(c.shape); E2 = emu(X.astype(np.int64) - 65536 * (X >= 32768)).reshape(c.shape)  # wrapped
if c.shape[1:3] == tuple(i.shape[:2]):
    bad = (c != E1).any(-1); print("pixels hot: mismatch", bad[hot].mean().round(4), " cold:", bad[~hot].mean().round(4))
print("per-frame mismatch (frames 0-3 cold, 4-7 hot):", (c != E1).reshape(8, -1).mean(1).round(4).tolist())
print("chip vs emu(unsigned input):", (c != E1).mean().round(4), " chip vs emu(input wrapped to int16):", (c != E2).mean().round(4), "max|d|", np.abs(c - E2).max())
