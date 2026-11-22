"""Report 5: SDK_BIT_EXACT clamps raw UINT16 inputs at 32767, the chip does not.
One input element set to each raw value in an otherwise in-distribution frame; chip vs emulator in that element's
neighbourhood. Usage (DFC env): python u16_boundary.py PROBE_DIR (m.hef + q.har of a 16-bit-input network)"""
import fcntl, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chip_vs_emu as cv
from hailo_platform import HEF, VDevice
from hailo_sdk_client import ClientRunner, InferenceContext
d = sys.argv[1]; dev = os.environ.get("DEVICE", "0000:03:00.0")
hef = HEF(f"{d}/m.hef"); i = hef.get_input_vstream_infos()[0]; o = hef.get_output_vstream_infos()[0]; q = i.quant_info
vals = [30000, 32000, 32767, 32768, 33000, 40000, 50000, 65535]
X = np.repeat(np.clip(np.round(np.random.default_rng(3).normal(0, 1, (1, *i.shape)) / q.qp_scale + q.qp_zp), 0, 65535),
              len(vals), 0).astype(np.uint16)
for k, v in enumerate(vals):
    X[k, 10, 10, 0] = v
r = ClientRunner(har=f"{d}/q.har")
with r.infer_context(InferenceContext.SDK_BIT_EXACT) as ctx:
    e = np.asarray(r.infer(ctx, (X.astype(np.float32) - q.qp_zp) * q.qp_scale))
p = VDevice.create_params(); p.device_ids = [dev]; vd = VDevice(p)
try:
    c = np.asarray(next(iter(cv.run16(vd, hef, {i.name: X}).values()))).astype(np.int64)
finally:
    vd.release()
E_ = np.clip(np.round(e / o.quant_info.qp_scale + o.quant_info.qp_zp), 0, 65535).astype(np.int64).reshape(c.shape)
for k, v in enumerate(vals):
    nb = slice(9, 12); dd = np.abs(c[k, nb, nb] - E_[k, nb, nb])
    print(f"raw {v:5d}: neighbourhood mismatch {(dd > 0).mean():.3f} max|d| {dd.max():5d}  chip {c[k, 10, 10, 0]} emu {E_[k, 10, 10, 0]}")
