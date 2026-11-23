"""chiprun.py OUT.npz DEV HEF [HEF ...]: run each HEF on Hailo-8 #DEV with the same fixed uint8 input (seed 0),
save raw device outputs (FormatType.AUTO) per HEF."""
import sys, numpy as np
from hailo_platform import (HEF, ConfigureParams, FormatType, HailoStreamInterface, InferVStreams, InputVStreamParams,
                            OutputVStreamParams, VDevice, Device)
out, dev, hefs = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
res = {}
params = VDevice.create_params(); params.device_ids = [Device.scan()[dev]]
with VDevice(params) as vd:
    for i, path in enumerate(hefs):
        hef = HEF(path)
        ng = vd.configure(hef, ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe))[0]
        inf = hef.get_input_vstream_infos()[0]
        x = np.random.default_rng(0).integers(0, 256, (2,) + tuple(inf.shape), dtype=np.uint8)
        ip = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
        op = OutputVStreamParams.make(ng, format_type=FormatType.AUTO)
        with ng.activate(ng.create_params()), InferVStreams(ng, ip, op) as pipe:
            o = pipe.infer({inf.name: x})
        for k, v in o.items():
            res[f"h{i}"] = np.asarray(v)
        vd.release() if False else None
        del ng
        print(path, {k: (np.asarray(v).shape, np.asarray(v).dtype) for k, v in o.items()}, flush=True)
np.savez(out, **res)
