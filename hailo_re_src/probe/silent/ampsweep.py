"""Where does chip vs SDK_BIT_EXACT start to diverge? Uniform inputs of growing amplitude around the input range
centre; per amplitude: mismatch fraction, max |diff|. Also where the calibration range sits (normal(0,1) -> q).
Usage: python ampsweep.py DIR [DIR...]   (DEVICE env)"""
import fcntl, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chip_vs_emu as cv
from hailo_platform import HEF, VDevice
from hailo_sdk_client import ClientRunner, InferenceContext
dev = os.environ.get("DEVICE", "0000:03:00.0")
lk = open(f"/mnt/storage/lib_project/models/Hailo_model/hailo_re/turbo/.chip_{dev.replace(':', '_')}.lock", "w")
for d in sys.argv[1:]:
    hef = HEF(f"{d}/m.hef"); i = hef.get_input_vstream_infos()[0]; o = hef.get_output_vstream_infos()[0]
    u16 = "16" in str(i.format.type); hi = 65535 if u16 else 255; q = i.quant_info
    cal = (1 / q.qp_scale)  # one float unit (calibration std) in LSB
    rng = np.random.default_rng(0); runner = ClientRunner(har=f"{d}/q.har")
    print(f"{d.split('/ops/')[-1]}: input {'u16' if u16 else 'u8'} zp {q.qp_zp:.0f} 1.0 float = {cal:.0f} LSB")
    for a in (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0):
        lo, up = max(0, q.qp_zp - a * hi / 2), min(hi, q.qp_zp + a * hi / 2)
        X = np.round(rng.uniform(lo, up, (8, *i.shape))).astype(np.uint16 if u16 else np.uint8)
        with runner.infer_context(InferenceContext.SDK_BIT_EXACT) as ctx:
            e = np.asarray(runner.infer(ctx, (X.astype(np.float32) - q.qp_zp) * q.qp_scale))
        fcntl.flock(lk, fcntl.LOCK_EX); p = VDevice.create_params(); p.device_ids = [dev]; vd = VDevice(p)
        try: c = np.asarray(next(iter(cv.run16(vd, hef, {i.name: X}).values()))).astype(np.int64)
        finally: vd.release(); fcntl.flock(lk, fcntl.LOCK_UN)
        oh = 65535 if c.max() > 255 or "16" in str(o.format.type) else 255
        E = np.clip(np.round(e / o.quant_info.qp_scale + o.quant_info.qp_zp), 0, oh).astype(np.int64).reshape(c.shape)
        print(f"   amplitude {a:4.2f} (+-{(up - lo) / 2 / cal:6.2f} float): mismatch {(c != E).mean():.4f} max|d| {np.abs(c - E).max()}", flush=True)
