"""Real-chip throughput of the host_vs_chip variants (hailo env, both chips idle).

  chip      : chip.hef alone on one chip (batch 1, sync loop)
  host_1chip: part1.hef + part2.hef on ONE chip (HailoRT scheduler switches network groups) + op in onnxruntime
  host_2chip: part1 on chip 0, part2 on chip 1, op on the host, pipelined (threads + queues)
  staged_K  : ONE chip, no scheduler: part1 runs a chunk of K frames -> boundary tensors cached in host RAM ->
              host op on the cache -> part2 runs the chunk. One network switch per chunk instead of per frame.
  SFX=_16 times the 16-bit builds (chip_16.hef, part1_16.hef, part2_16.hef); STAGED=64,256 sets the chunk sizes.

Usage: python hvc_time.py CASE_DIR [N_FRAMES]
"""
import json
import queue
import sys
import threading
import time

import numpy as np
import onnxruntime as ort
from hailo_platform import (HEF, ConfigureParams, FormatType, HailoSchedulingAlgorithm, HailoStreamInterface,
                            InferVStreams, InputVStreamParams, OutputVStreamParams, VDevice)

d = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
SFX = __import__("os").environ.get("SFX", "")
STAGED = [int(k) for k in __import__("os").environ.get("STAGED", "64,256").split(",") if k]
ids = [str(x) for x in __import__("hailo_platform").Device.scan()]


def vdev(dev_ids, sched=False):
    p = VDevice.create_params()
    p.device_ids = dev_ids
    if sched:
        p.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    return VDevice(p)


BATCH = int(__import__("os").environ.get("BATCH", "1"))


def setup(vd, path):
    hef = HEF(path)
    cp = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    for k in cp:
        cp[k].batch_size = BATCH
    ng = vd.configure(hef, cp)[0]
    ip = InputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
    op = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
    shape = hef.get_input_vstream_infos()[0].shape
    return ng, ip, op, hef.get_input_vstream_infos()[0].name, shape


res = {}
x = (np.random.default_rng(0).random((BATCH, 224, 224, 3)) * 255).astype(np.float32)
op_sess = ort.InferenceSession(f"{d}/op.onnx")
_run = op_sess.run


def host_op(f):  # the exported op has batch 1: one call per frame of the batch
    return np.concatenate([_run(None, {"f": f[i:i + 1]})[0] for i in range(len(f))])
# host op cost alone
f0 = np.random.default_rng(1).random((BATCH, 64, 56, 56)).astype(np.float32)
t = time.perf_counter()
for _ in range(50):
    host_op(f0)
res["host_op_ms_per_frame"] = (time.perf_counter() - t) / 50 / BATCH * 1e3

try:  # chip variant
    with vdev([ids[0]]) as vd:
        ng, ip, op, iname, _ = setup(vd, f"{d}/chip{SFX}.hef")
        with ng.activate(ng.create_params()), InferVStreams(ng, ip, op) as pipe:
            for _ in range(10):
                pipe.infer({iname: x})
            t = time.perf_counter()
            for _ in range(N):
                pipe.infer({iname: x})
            res["chip_fps"] = N * BATCH / (time.perf_counter() - t)
            xs = np.repeat(x[:1], 256, 0)  # streamed: 256 frames per call, as the staged chunks
            pipe.infer({iname: xs})
            t = time.perf_counter()
            for _ in range(4):
                pipe.infer({iname: xs})
            res["chip_stream_fps"] = 4 * 256 / (time.perf_counter() - t)
except Exception as e:  # noqa: BLE001
    res["chip_error"] = f"{type(e).__name__}: {str(e)[:150]}"

with vdev([ids[0]], sched=True) as vd:  # host variant, one chip, scheduler switching
    n1, i1, o1, in1, _ = setup(vd, f"{d}/part1{SFX}.hef")
    n2, i2, o2, in2, _ = setup(vd, f"{d}/part2{SFX}.hef")
    with InferVStreams(n1, i1, o1) as p1, InferVStreams(n2, i2, o2) as p2:
        def step():
            f = list(p1.infer({in1: x}).values())[0]  # NHWC float
            g = host_op(np.transpose(f, (0, 3, 1, 2)).astype(np.float32))
            return p2.infer({in2: np.transpose(g, (0, 2, 3, 1)).astype(np.float32)})
        for _ in range(10):
            step()
        t = time.perf_counter()
        for _ in range(N):
            step()
        res["host_1chip_fps"] = N * BATCH / (time.perf_counter() - t)

for K in STAGED:  # staged: chunk through part1, cache, host op, chunk through part2 (one chip)
    n_chunk = max(1, (N * BATCH) // K) if K >= BATCH else 1
    with vdev([ids[0]]) as vd:
        n1, i1, o1, in1, _ = setup(vd, f"{d}/part1{SFX}.hef")
        n2, i2, o2, in2, _ = setup(vd, f"{d}/part2{SFX}.hef")
        xs = np.repeat(x[:1], K, 0)

        def chunk():
            with n1.activate(n1.create_params()), InferVStreams(n1, i1, o1) as p1:
                f = list(p1.infer({in1: xs}).values())[0]  # the whole chunk streams through in one call
            g = np.transpose(host_op(np.transpose(f, (0, 3, 1, 2)).astype(np.float32)), (0, 2, 3, 1))
            g = np.ascontiguousarray(g, np.float32)
            with n2.activate(n2.create_params()), InferVStreams(n2, i2, o2) as p2:
                p2.infer({in2: g})
        chunk()
        t = time.perf_counter()
        for _ in range(n_chunk):
            chunk()
        res[f"staged_{K}_fps"] = n_chunk * K / (time.perf_counter() - t)

if len(ids) > 1:  # host variant, two chips, pipelined
    with vdev([ids[0]]) as va, vdev([ids[1]]) as vb:
        n1, i1, o1, in1, _ = setup(va, f"{d}/part1{SFX}.hef")
        n2, i2, o2, in2, _ = setup(vb, f"{d}/part2{SFX}.hef")
        q = queue.Queue(maxsize=8)
        done = []
        with n1.activate(n1.create_params()), n2.activate(n2.create_params()), \
                InferVStreams(n1, i1, o1) as p1, InferVStreams(n2, i2, o2) as p2:
            def a():
                for _ in range(N + 10):
                    q.put(list(p1.infer({in1: x}).values())[0])
                q.put(None)

            def b():
                while True:
                    f = q.get()
                    if f is None:
                        break
                    g = host_op(np.transpose(f, (0, 3, 1, 2)).astype(np.float32))
                    p2.infer({in2: np.transpose(g, (0, 2, 3, 1)).astype(np.float32)})
                    done.append(time.perf_counter())
            ta, tb = threading.Thread(target=a), threading.Thread(target=b)
            ta.start(); tb.start(); ta.join(); tb.join()
            res["host_2chip_fps"] = (len(done) - 10) * BATCH / (done[-1] - done[9])
print(json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in res.items()}))
json.dump(res, open(f"{d}/timing{SFX}_b{BATCH}.json", "w"), indent=1)
