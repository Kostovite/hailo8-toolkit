"""Multi-process sharing via hailort_service: several PROCESSES infer different HEFs on one chip, must equal solo runs.
Usage: python mproc.py solo HEF OUT.npz        (no service; direct VDevice)
       python mproc.py mp HEF REF.npz REPS     (multi_process_service=True; hailort_service must be running)
DEVICE env selects the chip; the caller holds the chip lock."""
import os, sys
import numpy as np
from hailo_platform import (HEF, ConfigureParams, FormatType, HailoSchedulingAlgorithm, HailoStreamInterface,
                            InferVStreams, InputVStreamParams, OutputVStreamParams, VDevice)


def frames(hef, n, seed):
    rng = np.random.default_rng(seed); X = {}
    for i in hef.get_input_vstream_infos():
        u16 = "16" in str(i.format.type); hi = 65535 if u16 else 255; q = i.quant_info
        X[i.name] = np.clip(np.round(rng.normal(0, 1, (n, *i.shape)) / q.qp_scale + q.qp_zp), 0, hi).astype(np.uint16 if u16 else np.uint8)
    return X


def infer(ng, x, activate):
    u16 = any(v.dtype == np.uint16 for v in x.values())
    ip = InputVStreamParams.make(ng, format_type=FormatType.UINT16 if u16 else FormatType.UINT8)
    op = OutputVStreamParams.make(ng, format_type=FormatType.AUTO)
    if not activate:
        with InferVStreams(ng, ip, op) as pipe:
            return {k: np.asarray(v).copy() for k, v in pipe.infer(x).items()}
    with ng.activate(ng.create_params()), InferVStreams(ng, ip, op) as pipe:
        return {k: np.asarray(v).copy() for k, v in pipe.infer(x).items()}


mode, path = sys.argv[1], sys.argv[2]
dev = os.environ.get("DEVICE", "0000:03:00.0")
hef = HEF(path); x = frames(hef, 16, 7)
p = VDevice.create_params(); p.device_ids = [dev]
if mode == "mp":
    p.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN; p.multi_process_service = True; p.group_id = "SHARED"
vd = VDevice(p)
try:
    ng = vd.configure(hef, ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe))[0]
    if mode == "solo":
        np.savez(sys.argv[3], **{k.replace("/", "|"): v for k, v in infer(ng, x, True).items()})
    else:
        ref = {k.replace("|", "/"): v for k, v in np.load(sys.argv[3]).items()}
        bad = []
        for _ in range(int(sys.argv[4])):
            r = infer(ng, x, False); bad.append(sum(int((r[k] != ref[k]).sum()) for k in r))
        print(path, "multi-process runs vs solo, mismatching elements per run:", bad, flush=True)
finally:
    vd.release()
