import os, sys, numpy as np
sys.path.insert(0, "/mnt/storage/lib_project/models/Hailo_model/hailo_re/cprof/port")
import exact_check as ec
from hailo_platform import HEF, VDevice
d = sys.argv[1]; hef = HEF(f"{d}/m.hef"); o = hef.get_output_vstream_infos()[0]
E = np.clip(np.round(np.load(f"{d}/emu_ref.npz")["arr_0"] / o.quant_info.qp_scale + o.quant_info.qp_zp), 0, 255).astype(int).reshape(8, -1)
rng = np.random.default_rng(0); X = {i.name: rng.integers(0, 256, (8, *i.shape), dtype=np.uint8) for i in hef.get_input_vstream_infos()}
p = VDevice.create_params(); p.device_ids = ["0000:03:00.0"]; vd = VDevice(p)
perm = [3, 0, 1, 2, 4, 5, 6, 7]
c = next(iter(ec.run(vd, f"{d}/m.hef", lambda h: {k: v[perm] for k, v in X.items()}).values())).astype(int).reshape(8, -1)
print("order", perm, "wrong bytes per position:", (c != E[perm]).sum(1))
x0 = X[next(iter(X))][0]
print("frame0 input stats: mean", x0.mean().round(1), "std", x0.std().round(1))
