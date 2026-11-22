"""usage: chip.py HEF IN_NPY OUT_PREFIX uint8|float32  -> OUT_PREFIX_<i>.npy float32 from the REAL Hailo-8, ordered like emu.py.
Run under: flock $CHIP_LOCK"""
import sys, numpy as np
from hailo_platform import HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType
hef_p, src, pre, fmt = sys.argv[1:5]; x = np.load(src, mmap_mode="r")
hef = HEF(hef_p); dev = VDevice(); ng = dev.configure(hef, ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe))[0]
ip = InputVStreamParams.make(ng, format_type={"uint8": FormatType.UINT8, "uint16": FormatType.UINT16}.get(fmt, FormatType.FLOAT32))
op = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
iname = hef.get_input_vstream_infos()[0].name; onames = [o.name for o in hef.get_output_vstream_infos()]
out = {n: [] for n in onames}; B = 8
with InferVStreams(ng, ip, op) as pipe, ng.activate(ng.create_params()):
    for i in range(0, len(x), B):
        xb = np.ascontiguousarray(x[i:i + B]).astype({"uint8": np.uint8, "uint16": np.uint16}.get(fmt, np.float32))
        r = pipe.infer({iname: xb})
        for n in onames: out[n].append(r[n])
import os
ys = [np.concatenate(out[n]).astype(np.float16 if os.environ.get("EMU_F16") == "1" else np.float32) for n in onames]; sz = [y[0].size for y in ys]
assert len(set(sz)) == len(sz), sz
for j, k in enumerate(sorted(range(len(ys)), key=lambda k: -sz[k])):
    np.save(f"{pre}_{j}.npy", ys[k].reshape((len(x),) + ys[k].shape[1:]) if ys[k].ndim > 1 else ys[k]); print(j, onames[k], ys[k].shape)
