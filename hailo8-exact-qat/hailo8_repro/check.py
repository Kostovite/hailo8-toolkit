# check.py - build an ONNX probe with the stock DFC flow, then compare Hailo-8 chip vs SDK_BIT_EXACT vs float.
# usage: python check.py m.onnx [extra.alls] [amplitude]
import os
import sys
import numpy as np
import onnx
import onnxruntime as ort
from hailo_sdk_client import ClientRunner, InferenceContext
from hailo_platform import (HEF, VDevice, ConfigureParams, HailoStreamInterface, InferVStreams,
                            InputVStreamParams, OutputVStreamParams, FormatType)

onnx_path = sys.argv[1]
extra = open(sys.argv[2]).read() if len(sys.argv) > 2 and sys.argv[2] != "-" else ""
amp = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
_, c, h, w = [d.dim_value for d in onnx.load(onnx_path).graph.input[0].type.tensor_type.shape.dim]

# 1-3. parse, quantize (level 0, 64 N(0,1) calibration frames), compile
runner = ClientRunner(hw_arch="hailo8")
runner.translate_onnx_model(onnx_path, "probe")
runner.load_model_script("model_optimization_flavor(optimization_level=0, compression_level=0)\n" + extra)
runner.optimize(np.random.default_rng(1).normal(0, 1, (64, h, w, c)).astype(np.float32))
runner.save_har("q.har")
open("m.hef", "wb").write(runner.compile())

# 4. 8 test frames N(0, amp), quantized to the HEF's raw input format
hef = HEF("m.hef")
ii, oi = hef.get_input_vstream_infos()[0], hef.get_output_vstream_infos()[0]
u16 = "16" in str(ii.format.type)
X = np.random.default_rng(0).normal(0, amp, (8, h, w, c)) / ii.quant_info.qp_scale + ii.quant_info.qp_zp
X = np.clip(np.round(X), 0, 65535 if u16 else 255).astype(np.uint16 if u16 else np.uint8)
xf = (X.astype(np.float32) - ii.quant_info.qp_zp) * ii.quant_info.qp_scale

# 5a. float reference (onnxruntime) and SDK_BIT_EXACT emulator on the same dequantized frames
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
ref = np.concatenate([sess.run(None, {sess.get_inputs()[0].name: f[None].transpose(0, 3, 1, 2)})[0]
                      for f in xf]).transpose(0, 2, 3, 1)
q = ClientRunner(har="q.har")
with q.infer_context(InferenceContext.SDK_BIT_EXACT) as ctx:
    emu = np.asarray(q.infer(ctx, xf)).reshape(ref.shape)

# 5b. chip: the whole batch, then frame by frame
p = VDevice.create_params()
if os.environ.get("HAILO_DEVICE_ID"):  # optional: pick one PCIe device, e.g. 0000:03:00.0
    p.device_ids = [os.environ["HAILO_DEVICE_ID"]]
vd = VDevice(p)
try:
    ng = vd.configure(hef, ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe))[0]
    ip = InputVStreamParams.make(ng, format_type=FormatType.UINT16 if u16 else FormatType.UINT8)
    op = OutputVStreamParams.make(ng, format_type=FormatType.AUTO)
    with ng.activate(ng.create_params()), InferVStreams(ng, ip, op) as pipe:
        run = lambda x: np.asarray(pipe.infer({ii.name: x})[oi.name]).copy()
        chip = run(X).reshape(ref.shape).astype(np.int64)
        chip1 = np.concatenate([run(X[i:i + 1]) for i in range(len(X))]).reshape(ref.shape).astype(np.int64)
finally:
    vd.release()

# 6. compare in output LSB and against float
qo = oi.quant_info
hi = 65535 if "16" in str(oi.format.type) else 255
emu_q = np.clip(np.round(emu / qo.qp_scale + qo.qp_zp), 0, hi).astype(np.int64)
deq = lambda a: (a - qo.qp_zp) * qo.qp_scale
snr = lambda a: 10 * np.log10((ref ** 2).mean() / ((deq(a) - ref) ** 2).mean())
d = np.abs(chip - emu_q)
print(f"chip != emulator: {100 * (d > 0).mean():.2f}% of outputs, max {d.max()} LSB")
print(f"SNR vs float: chip {snr(chip):.1f} dB, emulator {snr(emu_q):.1f} dB")
print(f"chip batch == chip frame-by-frame: {(chip == chip1).all()}")
