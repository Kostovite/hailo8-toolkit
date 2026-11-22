"""HailoRT scheduler: several network groups on one chip, inferred concurrently from threads, must equal solo runs.
Usage: python sched.py HEF1 HEF2 [HEF3]  (DEVICE env)"""
import fcntl, os, sys, threading
import numpy as np
from hailo_platform import (HEF, ConfigureParams, FormatType, HailoSchedulingAlgorithm, HailoStreamInterface,
                            InferVStreams, InputVStreamParams, OutputVStreamParams, VDevice)


def frames(hef, n, seed):
    rng = np.random.default_rng(seed); X = {}
    for i in hef.get_input_vstream_infos():
        u16 = "16" in str(i.format.type); hi = 65535 if u16 else 255; q = i.quant_info
        X[i.name] = np.clip(np.round(rng.normal(0, 1, (n, *i.shape)) / q.qp_scale + q.qp_zp), 0, hi).astype(np.uint16 if u16 else np.uint8)
    return X


dev = os.environ.get("DEVICE", "0000:03:00.0")
lk = open(f"/mnt/storage/lib_project/models/Hailo_model/hailo_re/turbo/.chip_{dev.replace(':', '_')}.lock", "w"); fcntl.flock(lk, fcntl.LOCK_EX)
hefs = [HEF(h) for h in sys.argv[1:]]; X = [frames(h, 16, 7) for h in hefs]


def run(ng, x, sched):
    u16 = any(v.dtype == np.uint16 for v in x.values())
    ip = InputVStreamParams.make(ng, format_type=FormatType.UINT16 if u16 else FormatType.UINT8)
    op = OutputVStreamParams.make(ng, format_type=FormatType.AUTO)
    if sched:
        with InferVStreams(ng, ip, op) as pipe:
            return {k: np.asarray(v).copy() for k, v in pipe.infer(x).items()}
    with ng.activate(ng.create_params()), InferVStreams(ng, ip, op) as pipe:
        return {k: np.asarray(v).copy() for k, v in pipe.infer(x).items()}


solo = []
for h, x in zip(hefs, X):
    p = VDevice.create_params(); p.device_ids = [dev]; vd = VDevice(p)
    try: solo.append(run(vd.configure(h, ConfigureParams.create_from_hef(h, interface=HailoStreamInterface.PCIe))[0], x, False))
    finally: vd.release()
p = VDevice.create_params(); p.device_ids = [dev]; p.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
vd = VDevice(p); res = [[] for _ in hefs]
try:
    ngs = [vd.configure(h, ConfigureParams.create_from_hef(h, interface=HailoStreamInterface.PCIe))[0] for h in hefs]
    def worker(j):
        for _ in range(6): res[j].append(run(ngs[j], X[j], True))
    th = [threading.Thread(target=worker, args=(j,)) for j in range(len(hefs))]
    [t.start() for t in th]; [t.join() for t in th]
finally:
    vd.release()
for j, h in enumerate(sys.argv[1:]):
    bad = [sum(int((r[k] != solo[j][k]).sum()) for k in r) for r in res[j]]
    print(os.path.basename(os.path.dirname(h)) or h, "scheduled runs vs solo, mismatching elements per run:", bad)
