"""Run a HEF on a Hailo-8 over the val (or VAL_DIR) pages and save the RAW integer head outputs (no host dequant).

Output npz: one array per output vstream (N, H, W, C) in its native integer type + "files". Pages are letterboxed
exactly like evaluate.py (they are already 896x896 here, so this is the identity).

Usage (hailo env): python chip_dump.py HEF OUT_NPZ [DEVICE_INDEX] [N]
"""
import glob
import os
import sys
import time

import cv2
import numpy as np
from hailo_platform import (HEF, ConfigureParams, FormatType, HailoStreamInterface, InferVStreams, InputVStreamParams,
                            OutputVStreamParams, VDevice)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from letterbox import letterbox  # noqa: E402

hef_path, out = sys.argv[1], sys.argv[2]
dev_idx = int(sys.argv[3]) if len(sys.argv) > 3 else 0
n = int(sys.argv[4]) if len(sys.argv) > 4 else 0
val = os.environ.get("VAL_DIR", "/mnt/storage/lib_project/models/Hailo_model/hailo_re/data_v14sec/val/images")
files = sorted(glob.glob(os.path.join(val, "*")))[: n or None]
hef = HEF(hef_path)
params = VDevice.create_params()
ids = [d for d in __import__("hailo_platform").Device.scan()]
params.device_ids = [ids[dev_idx]]
with VDevice(params) as vd:
    ng = vd.configure(hef, ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe))[0]
    ip = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
    op = OutputVStreamParams.make(ng, format_type=FormatType.AUTO)  # raw device integers
    in_name = hef.get_input_vstream_infos()[0].name
    h, w = hef.get_input_vstream_infos()[0].shape[:2]
    res = {}
    t = time.time()
    with ng.activate(ng.create_params()), InferVStreams(ng, ip, op) as pipe:
        for i in range(0, len(files), 8):
            batch = np.stack([letterbox(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB), (h, w))[0] for f in files[i:i + 8]])
            o = pipe.infer({in_name: batch.astype(np.uint8)})
            for k, v in o.items():
                res.setdefault(k, []).append(np.asarray(v))
            if i % 200 == 0:
                print(f"{i + len(batch)}/{len(files)} {(i + len(batch)) / (time.time() - t):.1f} img/s", flush=True)
np.savez(out, files=np.array(files), **{k.replace("/", "__"): np.concatenate(v) for k, v in res.items()})
print("saved", out, {k: (np.concatenate(v).shape, np.concatenate(v).dtype) for k, v in res.items()})
