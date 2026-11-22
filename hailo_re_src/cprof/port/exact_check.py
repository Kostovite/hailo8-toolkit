"""Are differently-partitioned HEFs of one HAR bit-identical? Same raw (quantized) input frames into each HEF, raw
outputs (no dequant) compared byte for byte against the first HEF.

Usage (DFC env): python exact_check.py REF.hef A.hef B.hef ...   [FRAMES=64 env, DEVICE=0000:02:00.0 env]
"""
import os
import sys

import numpy as np
from hailo_platform import (HEF, ConfigureParams, FormatType, HailoStreamInterface, InferVStreams, InputVStreamParams,
                            OutputVStreamParams, VDevice)

N = int(os.environ.get("FRAMES", 64))


def run(vd, path, frames):
    hef = HEF(path)
    cp = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    ng = vd.configure(hef, cp)[0]
    ip = InputVStreamParams.make(ng, format_type=FormatType.AUTO)  # AUTO = the HEF's own raw integer format
    op = OutputVStreamParams.make(ng, format_type=FormatType.AUTO)
    with ng.activate(ng.create_params()), InferVStreams(ng, ip, op) as pipe:
        return pipe.infer(frames(hef))


def main(paths):
    p = VDevice.create_params()
    p.device_ids = [os.environ.get("DEVICE", "0000:02:00.0")]
    vd = VDevice(p)
    rng = np.random.default_rng(0)
    cache = {}

    def frames(hef):
        if not cache:  # the same frames for every HEF (one HAR: same input layers and formats)
            for vi in hef.get_input_vstream_infos():
                dt = np.uint16 if "16" in str(vi.format.type) else np.uint8
                cache[vi.name] = rng.integers(0, np.iinfo(dt).max + 1, (N, *vi.shape), dtype=dt)
        return cache

    ref = None
    for path in paths:
        out = run(vd, path, frames)
        if ref is None:
            ref = out
            print(f"{path}: reference, {len(out)} outputs, {sum(v.nbytes for v in out.values())} bytes")
            continue
        diffs = []
        for k, v in ref.items():
            w = out[k]
            n_bad = int(np.count_nonzero(v != w)) if v.shape == w.shape else -1
            diffs.append((k, n_bad, int(np.abs(v.astype(np.int64) - w.astype(np.int64)).max()) if n_bad > 0 else 0))
        bad = [d for d in diffs if d[1]]
        print(f"{path}: {'BIT-EXACT' if not bad else 'DIFFERS'}" + "".join(f"\n   {k}: {n} elems, max |d| {m}"
                                                                           for k, n, m in bad))


if __name__ == "__main__":
    main(sys.argv[1:])
