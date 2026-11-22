import fcntl, sys
import numpy as np
sys.path.insert(0, "/mnt/storage/lib_project/models/Hailo_model/hailo_re/probe/silent")
import chip_vs_emu as cv
from hailo_platform import HEF, VDevice
d = "/mnt/storage/lib_project/models/Hailo_model/hailo_re/runs/det/att"
z = np.load(f"{d}/ints2.npz"); nb = int(z["nbad"])
dev = "0000:03:00.0"; lk = open(f"/mnt/storage/lib_project/models/Hailo_model/hailo_re/turbo/.chip_{dev.replace(':', '_')}.lock", "w")
def run(tag, feeds):
    hef = HEF(f"{d}/{tag}.hef"); fcntl.flock(lk, fcntl.LOCK_EX); p = VDevice.create_params(); p.device_ids = [dev]; vd = VDevice(p)
    try:
        ins = {i.name: feeds(i) for i in hef.get_input_vstream_infos()}
        return hef, cv.run16(vd, hef, ins)
    finally: vd.release(); fcntl.flock(lk, fcntl.LOCK_UN)
def cmp(tag, hef, out):
    for o in hef.get_output_vstream_infos():
        c = np.asarray(out[o.name]).astype(np.int64).reshape(16, -1)
        for k in [x for x in z.files if x.startswith("best__")]:
            e = z[k].astype(np.int64).reshape(16, -1)
            if e.shape == c.shape and (c == e).mean() > 0.5:
                dd = c != e
                print(f"{tag} {o.name} == {k}? per-page mismatch bad {dd[:nb].sum(1).tolist()} good {dd[nb:].sum(1).tolist()}")
#hef, out = run("C", lambda i: z["best__ew_add9"].astype(np.uint8)); cmp("C", hef, out)
def feedD(i):
    k = {"activation1": "best__activation1", "ew_mult_softmax1": "best__ew_mult_softmax1"}
    nm = next(v for kk, v in k.items() if kk in i.name or kk in str(i.shape))
    return z[nm].astype(np.uint8 if z[nm].min() >= 0 else np.int8).reshape((16,) + tuple(i.shape))
hd = HEF(f"{d}/D.hef"); print([(i.name, i.shape, i.quant_info.qp_zp) for i in hd.get_input_vstream_infos()])
hef, out = run("E", lambda i: z["best__ew_add9"].astype(np.uint8)); cmp("E", hef, out)
o = [x for x in HEF(f"{d}/C.hef").get_output_vstream_infos() if "activation1" in x.name][0]
