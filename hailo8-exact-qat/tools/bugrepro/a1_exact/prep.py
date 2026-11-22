"""A1 positive test, step 1 (hailo env): on-grid input, HN dump, chip output of the mm_att probe (unsigned K)."""
import fcntl, json, sys
import numpy as np
from hailo_platform import HEF, VDevice
from hailo_sdk_client import ClientRunner
sys.path.insert(0, "/mnt/storage/lib_project/models/Hailo_model/hailo_re/probe/silent")
import chip_vs_emu as cv
D = "/mnt/storage/lib_project/models/Hailo_model/hailo_re/probe/runs/mm_fc/out/mm_att"
hn = ClientRunner(har=f"{D}/q.har").get_hn(); json.dump(json.loads(hn) if isinstance(hn, str) else hn, open("m.hn", "w"))
hef = HEF(f"{D}/m.hef"); ii = hef.get_input_vstream_infos()[0]; q = ii.quant_info
rng = np.random.default_rng(3)
raw = np.clip(np.round(rng.normal(0, 1, (1, *ii.shape)) / q.qp_scale + q.qp_zp), 0, 255).astype(np.uint8)  # NHWC
np.save("x.npy", np.transpose((raw.astype(np.float32) - q.qp_zp) * q.qp_scale, (0, 3, 1, 2)))  # NCHW, on grid
dev = "0000:02:00.0"; lk = open(f"/mnt/storage/lib_project/models/Hailo_model/hailo_re/turbo/.chip_{dev.replace(':', '_')}.lock", "w")
fcntl.flock(lk, fcntl.LOCK_EX); p = VDevice.create_params(); p.device_ids = [dev]; vd = VDevice(p)
try: c = cv.run16(vd, hef, {ii.name: raw})
finally: vd.release()
np.savez("chip.npz", **{k.replace("/", "|"): np.asarray(v) for k, v in c.items()}); print({k: np.asarray(v).shape for k, v in c.items()})
