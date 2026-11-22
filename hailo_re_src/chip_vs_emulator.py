"""Compare Hailo-8 hardware output with the DFC emulators on identical inputs.

Usage: python chip_vs_emulator.py HAR HEF IMG [IMG ...]
Writes per-image detection arrays to hailo_re/out/ and prints max deltas.
"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "99")

import cv2
import numpy as np

HAR, HEF_PATH, IMGS = sys.argv[1], sys.argv[2], sys.argv[3:]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)


def load(p):
    img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (896, 896))


batch = np.stack([load(p) for p in IMGS]).astype(np.uint8)


def run_hw():
    from hailo_platform import (HEF, ConfigureParams, FormatType, HailoStreamInterface, InferVStreams,
                                InputVStreamParams, OutputVStreamParams, VDevice)
    hef = HEF(HEF_PATH)
    with VDevice() as vd:
        ng = vd.configure(hef, ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe))[0]
        ip = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
        op = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
        name = hef.get_input_vstream_infos()[0].name
        with ng.activate(ng.create_params()), InferVStreams(ng, ip, op, tf_nms_format=True) as pipe:
            res = pipe.infer({name: batch})
    return {k: np.asarray(v) for k, v in res.items()}


def run_emu(ctx_name):
    from hailo_sdk_client import ClientRunner, InferenceContext
    runner = ClientRunner(har=HAR)
    with runner.infer_context(getattr(InferenceContext, ctx_name)) as ctx:
        res = runner.infer(ctx, batch.astype(np.float32))
    return np.asarray(res if not isinstance(res, list) else res[0])


which = os.environ.get("RUN", "hw,SDK_QUANTIZED,SDK_BIT_EXACT").split(",")
results = {}
for w in which:
    try:
        r = run_hw() if w == "hw" else run_emu(w)
        if isinstance(r, dict):
            r = next(iter(r.values()))
        results[w] = r
        np.save(os.path.join(OUT, f"{w}.npy"), r)
        print(f"[{w}] shape={r.shape} nonzero={np.count_nonzero(r)}", flush=True)
    except Exception as e:  # report and continue so one broken mode doesn't hide the others
        print(f"[{w}] FAILED: {type(e).__name__}: {str(e)[:400]}", flush=True)

ref = results.get("hw")
for w, r in results.items():
    if w == "hw" or ref is None:
        continue
    if r.shape != ref.shape:
        print(f"{w} vs hw: shape mismatch {r.shape} vs {ref.shape}")
        continue
    d = np.abs(r - ref)
    # tf_nms_format: [batch, classes, 5 (y0,x0,y1,x1,score), max_boxes]
    score_d = d[:, :, 4, :]
    print(f"{w} vs hw: max|d|={d.max():.3e} exact={np.array_equal(r, ref)} "
          f"max score d={score_d.max():.3e} boxes hw={int((ref[:, :, 4, :] > 0).sum())} "
          f"emu={int((r[:, :, 4, :] > 0).sum())}")
