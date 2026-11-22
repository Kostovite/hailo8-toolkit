"""Feed the exact ew_add9 integers of each page into the attention sub-model HEFs on the chip; compare the chip's
matmul1 / matmul2 (and every sub-model output) with the exact executor's integers."""
import fcntl, sys
import numpy as np
sys.path.insert(0, "/mnt/storage/lib_project/models/Hailo_model/hailo_re/probe/silent")
import chip_vs_emu as cv
from hailo_platform import HEF, VDevice
d = "/mnt/storage/lib_project/models/Hailo_model/hailo_re/runs/det/att"
z = np.load(f"{d}/ints.npz"); nb = int(z["nbad"]); X = z["best__ew_add9"].astype(np.uint8)
dev = sys.argv[1] if len(sys.argv) > 1 else "0000:03:00.0"
lk = open(f"/mnt/storage/lib_project/models/Hailo_model/hailo_re/turbo/.chip_{dev.replace(':', '_')}.lock", "w")
for tag in ("A", "B"):
    hef = HEF(f"{d}/{tag}.hef"); ii = hef.get_input_vstream_infos()[0]
    fcntl.flock(lk, fcntl.LOCK_EX); p = VDevice.create_params(); p.device_ids = [dev]; vd = VDevice(p)
    try: out = cv.run16(vd, hef, {ii.name: X})
    finally: vd.release(); fcntl.flock(lk, fcntl.LOCK_UN)
    for o in hef.get_output_vstream_infos():
        src = o.name.split("/")[-1]
        c = np.asarray(out[o.name]).astype(np.int64)
        cand = [k for k in z.files if k.endswith("__" + src.replace("output_layer", "x"))]
        print(tag, o.name, c.shape, o.quant_info.qp_zp, o.quant_info.qp_scale)
        for k in ("best__matmul1", "best__matmul2", "best__conv_feature_splitter5_2"):
            e = z[k].astype(np.int64)
            if e.reshape(len(e), -1).shape[1] == c.reshape(len(c), -1).shape[1]:
                dd = c.reshape(len(c), -1) != e.reshape(len(e), -1)
                print(f"   vs {k}: per-page mismatch frac bad {np.round(dd[:nb].mean(1), 4).tolist()} good {np.round(dd[nb:].mean(1), 4).tolist()}")
