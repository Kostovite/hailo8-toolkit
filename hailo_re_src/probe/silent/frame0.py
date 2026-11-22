"""First-frame test: is output frame 0 (first inference after configure+activate) wrong, and does it depend on what
ran on the chip before? A = HEF under test, B = an unrelated HEF run in between. Compared vs the emulator ref."""
import os
import sys

import numpy as np

sys.path.insert(0, "/mnt/storage/lib_project/models/Hailo_model/hailo_re/cprof/port")
import exact_check as ec  # noqa: E402
from hailo_platform import HEF, VDevice  # noqa: E402

d, other = sys.argv[1], sys.argv[2]
hef = HEF(f"{d}/m.hef")
o = hef.get_output_vstream_infos()[0]
z = np.load(f"{d}/emu_ref.npz")["arr_0"]
E = np.clip(np.round(z / o.quant_info.qp_scale + o.quant_info.qp_zp), 0, 255).astype(int).reshape(8, -1)
rng = np.random.default_rng(0)
X = {i.name: rng.integers(0, 256, (8, *i.shape), dtype=np.uint8) for i in hef.get_input_vstream_infos()}
p = VDevice.create_params()
p.device_ids = [os.environ.get("DEVICE", "0000:03:00.0")]
vd = VDevice(p)


def bad(tag, xs=X):
    c = next(iter(ec.run(vd, f"{d}/m.hef", lambda h: xs).values())).astype(int).reshape(len(next(iter(xs.values()))), -1)
    print(tag, "wrong bytes per frame:", (c != E[:len(c)]).sum(1) if len(c) == 8 else (c != E[:1]).sum(1))
    return c


a = bad("run1")
b = bad("run2 (same HEF again)")
print("frame0 run1==run2:", np.array_equal(a[0], b[0]))
oh = HEF(other)
ec.run(vd, other, lambda h: {i.name: np.random.default_rng(1).integers(0, 256, (4, *i.shape), dtype=np.uint8)
                             for i in oh.get_input_vstream_infos()})
c = bad("run3 (after another HEF)")
print("frame0 run1==run3:", np.array_equal(a[0], c[0]))
bad("single frame", {k: v[:1] for k, v in X.items()})
bad("single frame again", {k: v[:1] for k, v in X.items()})
