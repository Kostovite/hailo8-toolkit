"""Where do chip and emulator differ? per frame, per row/col (border vs interior), per channel."""
import sys, numpy as np
sys.path.insert(0, "/mnt/storage/lib_project/models/Hailo_model/hailo_re/probe/silent")
import chip_vs_emu as cv
from hailo_platform import HEF, VDevice
d = sys.argv[1]; hef = HEF(f"{d}/m.hef"); o = hef.get_output_vstream_infos()[0]
z = np.load(f"{d}/emu_ref.npz")["arr_0"]
E = np.clip(np.round(z / o.quant_info.qp_scale + o.quant_info.qp_zp), 0, 255).astype(int)
p = VDevice.create_params(); p.device_ids = [os.environ.get("DEVICE", "0000:03:00.0")]; vd = VDevice(p)
X = cv.frames(hef)
c = np.asarray(next(iter(cv.ec.run(vd, f"{d}/m.hef", lambda h: X).values()))).astype(int)
vd.release()
E = E.reshape(c.shape)
dd = np.abs(c - E)
print("shape", c.shape, "frac", (dd > 0).mean().round(4), "max", dd.max())
print("per frame frac", (dd > 0).reshape(len(c), -1).mean(1).round(4))
if c.ndim == 4:
    print("per row frac", (dd > 0).mean((0, 2, 3)).round(3))
    print("per col frac", (dd > 0).mean((0, 1, 3)).round(3))
    ch = (dd > 0).mean((0, 1, 2)); print("channels with diffs", int((ch > 0).sum()), "of", len(ch), "worst", ch.max().round(3))
    print("emu value range where diff", E[dd > 0].min(), E[dd > 0].max(), "chip", c[dd > 0].min(), c[dd > 0].max())
    print("diff histogram", np.bincount(np.minimum(dd[dd > 0], 10))[1:])
