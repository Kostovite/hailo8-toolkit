"""Exact executor vs chip, step 1 (hailo env): HN dump, an on-grid input (UINT8 or UINT16), the chip's output.
Usage: python prep.py SRC_DIR OUT_DIR   (SRC_DIR has q.har + m.hef)"""
import fcntl, json, os, sys
import numpy as np
from hailo_platform import HEF, VDevice
from hailo_sdk_client import ClientRunner
sys.path.insert(0, "/mnt/storage/lib_project/models/Hailo_model/hailo_re/probe/silent")
import chip_vs_emu as cv
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
hn = ClientRunner(har=f"{src}/q.har").get_hn(); json.dump(json.loads(hn) if isinstance(hn, str) else hn, open(f"{out}/m.hn", "w"))
hef = HEF(f"{src}/m.hef"); ii = hef.get_input_vstream_infos()[0]; q = ii.quant_info
u16 = "16" in str(ii.format.type); hi = 65535 if u16 else 255
rng = np.random.default_rng(3)
raw = np.clip(np.round(rng.normal(0, 1, (1, *ii.shape)) / q.qp_scale + q.qp_zp), 0, min(hi, 32000)).astype(np.uint16 if u16 else np.uint8)
np.save(f"{out}/x.npy", np.transpose((raw.astype(np.float64) - q.qp_zp) * q.qp_scale, (0, 3, 1, 2)).astype(np.float32))
np.save(f"{out}/raw.npy", raw)
dev = "0000:03:00.0"; lk = open(f"/mnt/storage/lib_project/models/Hailo_model/hailo_re/turbo/.chip_{dev.replace(':', '_')}.lock", "w")
fcntl.flock(lk, fcntl.LOCK_EX); p = VDevice.create_params(); p.device_ids = [dev]; vd = VDevice(p)
try: c = cv.run16(vd, hef, {ii.name: raw})
finally: vd.release()
np.savez(f"{out}/chip.npz", **{k.replace("/", "|"): np.asarray(v) for k, v in c.items()}); print({k: np.asarray(v).shape for k, v in c.items()})
