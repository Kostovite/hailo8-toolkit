"""usage: chip.py HEF IN_NPY(uint8 NHWC) OUT_NPY  -> outputs of the real Hailo-8 (float32 dequantized, first output)"""
import sys, numpy as np
from hailo_platform import HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType
hef_p, src, dst = sys.argv[1:4]; x = np.load(src, mmap_mode="r")
hef = HEF(hef_p); dev = VDevice(); ng = dev.configure(hef, ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe))[0]
ip = InputVStreamParams.make(ng, format_type=FormatType.UINT8); op = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
iname = hef.get_input_vstream_infos()[0].name; onames = [o.name for o in hef.get_output_vstream_infos()]
out = {n: [] for n in onames}
with InferVStreams(ng, ip, op) as pipe, ng.activate(ng.create_params()):
    for i in range(0, len(x), 32):
        r = pipe.infer({iname: np.array(x[i:i + 32])})
        for n in onames: out[n].append(r[n])
np.save(dst, np.concatenate(out[onames[0]]).astype(np.float32)); print("saved", dst, onames)
