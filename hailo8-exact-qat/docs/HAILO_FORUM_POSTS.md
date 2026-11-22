# Hailo-8 forum posts — one post per bug

Each post below is self-contained: copy everything between two `=====` lines into a new topic.
The first line of each post is the topic title.
All numbers were measured on 2026-10-26 with exactly the steps in the post.

Common to every post (already included in each one):
- Hailo-8 (PCIe), Dataflow Compiler 3.33.1, HailoRT 4.23.0 (bundled firmware), Ubuntu / Linux 6.1 x86-64, torch 2.4, onnx opset 17.
- `check.py` (inside each post, under "check.py") builds the ONNX with the default DFC flow, then runs the same raw frames
  on the chip, in `SDK_BIT_EXACT` and in onnxruntime.

=====

**[Hailo-8] ConvTranspose with kernel = stride (2×2/s2, 4×4/s4) gives wrong results on the chip, while the emulator is correct**

`ConvTranspose2d(C, C, kernel_size=2, stride=2)` compiles without any warning, but on the Hailo-8 99.5% of its outputs are wrong (SNR −10 dB vs float), while `SDK_BIT_EXACT` on the same HAR is correct (35 dB).

**Environment**
Hailo-8 PCIe · DFC 3.33.1 · HailoRT 4.23.0 · Linux x86-64 · torch 2.4 → ONNX opset 17 · default flow (`optimization_level=0`, `compression_level=0`, no other model-script lines).

**What we measured** (8 frames, N(0,1) input, same raw frames on chip and emulator)

| ConvTranspose | route chosen by the DFC (auto.alls) | chip ≠ emulator | SNR vs float, chip / emulator |
|---|---|---|---|
| k2 s2, C16 24×24 | `defuse(deconv1, 1, defuse_type=DECONV)` | **99.50%** (max 212 LSB) | **−9.6** / 35.1 dB |
| k2 s2, C64 16×16 | DECONV | **99.54%** | **−10.5** / 34.8 dB |
| k4 s4, C16 24×24 | DECONV | **99.48%** | **−6.2** / 38.2 dB |
| k4 s4, C64 12×12 | DECONV | **99.48%** | **−9.1** / 36.2 dB |
| k4 s2 p1 | DECONV | 0% | 36.3 / 36.3 dB |
| k8 s4 p2 | DECONV | 0% | 38.6 / 38.6 dB |
| k16 s8 p4 | DECONV | 0% | 41.4 / 41.4 dB |
| k2 s2 + `defuse_type=SUPER_DECONV` forced (same HAR) | SUPER_DECONV | 0% | 35.1 / 35.1 dB |

The chip output is not a permutation of the correct output. The result is deterministic, and batch output equals frame-by-frame output.

**Why it happens (from the DFC source)**
`hailo_sdk_common/numeric_utils/numeric_utils.py::is_super_deconv()` sends exactly five kernel/stride pairs to the native DECONV (feature-interleave) route and everything else to SUPER_DECONV:
```python
return not ((k == 4 and s == 2) or (k == 4 and s == 4) or (k == 2 and s == 2)
            or (k == 8 and s == 4) or (k == 16 and s == 8))
```
We tested all five on the chip. The three with **kernel = 2 × stride** are bit-exact. The two with **kernel = stride** (2/2 and 4/4), the standard non-overlapping upsamplers, are wrong. The emulator models all five correctly.

**Steps to reproduce**
1. Export the model:
```python
import torch, torch.nn as nn
class M(nn.Module):
    def __init__(s):
        super().__init__(); s.up = nn.ConvTranspose2d(16, 16, 2, 2); s.out = nn.Conv2d(16, 16, 1)
    def forward(s, x): return s.out(s.up(x))
torch.manual_seed(0)
torch.onnx.export(M().eval(), torch.randn(1, 16, 24, 24), "m.onnx", input_names=["x"], opset_version=17, dynamo=False)
```
2. `python check.py m.onnx`. `check.py` is below; it uses the default flow, then compares chip, emulator and float.
3. Output:
```
chip != emulator: 99.50% of outputs, max 212 LSB
SNR vs float: chip -9.6 dB, emulator 35.1 dB
chip batch == chip frame-by-frame: True
```
4. Force the other lowering. Put this line in `super.alls` and run `python check.py m.onnx super.alls`:
```
deconv1_defuse_conv, deconv1_defuse_d2s = defuse(deconv1, 1, defuse_type=SUPER_DECONV)
```
The output becomes `chip != emulator: 0.00%`, with SNR chip 35.1 dB and emulator 35.1 dB.

**Expected**: the chip output equals `SDK_BIT_EXACT`, or the DFC does not route kernel = stride deconvs to DECONV.
**Workaround**: the `SUPER_DECONV` line above, or in ONNX a `Conv 1×1 (C→4C)` followed by `DepthToSpace(2, CRD)`; both are bit-exact on the chip.

[details="check.py"]
```python
# usage: python check.py m.onnx [extra.alls] [amplitude]
import os, sys
import numpy as np, onnx, onnxruntime as ort
from hailo_sdk_client import ClientRunner, InferenceContext
from hailo_platform import (HEF, VDevice, ConfigureParams, HailoStreamInterface, InferVStreams,
                            InputVStreamParams, OutputVStreamParams, FormatType)
onnx_path = sys.argv[1]
extra = open(sys.argv[2]).read() if len(sys.argv) > 2 and sys.argv[2] != "-" else ""
amp = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
_, c, h, w = [d.dim_value for d in onnx.load(onnx_path).graph.input[0].type.tensor_type.shape.dim]
# parse, quantize (level 0, 64 N(0,1) calibration frames), compile
runner = ClientRunner(hw_arch="hailo8")
runner.translate_onnx_model(onnx_path, "probe")
runner.load_model_script("model_optimization_flavor(optimization_level=0, compression_level=0)\n" + extra)
runner.optimize(np.random.default_rng(1).normal(0, 1, (64, h, w, c)).astype(np.float32))
runner.save_har("q.har")
open("m.hef", "wb").write(runner.compile())
# 8 test frames N(0, amp) quantized to the HEF's raw input format
hef = HEF("m.hef")
ii, oi = hef.get_input_vstream_infos()[0], hef.get_output_vstream_infos()[0]
u16 = "16" in str(ii.format.type)
X = np.random.default_rng(0).normal(0, amp, (8, h, w, c)) / ii.quant_info.qp_scale + ii.quant_info.qp_zp
X = np.clip(np.round(X), 0, 65535 if u16 else 255).astype(np.uint16 if u16 else np.uint8)
xf = (X.astype(np.float32) - ii.quant_info.qp_zp) * ii.quant_info.qp_scale
# float reference and SDK_BIT_EXACT on the same dequantized frames
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
ref = np.concatenate([sess.run(None, {sess.get_inputs()[0].name: f[None].transpose(0, 3, 1, 2)})[0]
                      for f in xf]).transpose(0, 2, 3, 1)
q = ClientRunner(har="q.har")
with q.infer_context(InferenceContext.SDK_BIT_EXACT) as ctx:
    emu = np.asarray(q.infer(ctx, xf)).reshape(ref.shape)
# chip: whole batch, then frame by frame
p = VDevice.create_params()
if os.environ.get("HAILO_DEVICE_ID"):
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
qo = oi.quant_info
hi = 65535 if "16" in str(oi.format.type) else 255
emu_q = np.clip(np.round(emu / qo.qp_scale + qo.qp_zp), 0, hi).astype(np.int64)
deq = lambda a: (a - qo.qp_zp) * qo.qp_scale
snr = lambda a: 10 * np.log10((ref ** 2).mean() / ((deq(a) - ref) ** 2).mean())
d = np.abs(chip - emu_q)
print(f"chip != emulator: {100 * (d > 0).mean():.2f}% of outputs, max {d.max()} LSB")
print(f"SNR vs float: chip {snr(chip):.1f} dB, emulator {snr(emu_q):.1f} dB")
print(f"chip batch == chip frame-by-frame: {(chip == chip1).all()}")
```
[/details]

=====

**[Hailo-8] GroupNorm(1, C) gives wrong results on the chip (49% of outputs), while the emulator is correct**

`nn.GroupNorm(1, C)` (whole-tensor normalisation) compiles without warning, but on the Hailo-8 about half of the outputs are wrong (SNR 3.1 dB vs float), while `SDK_BIT_EXACT` gives 36.0 dB. GroupNorm with 2, 4 or 8 groups is exact.

**Environment**
Hailo-8 PCIe · DFC 3.33.1 · HailoRT 4.23.0 · Linux x86-64 · torch 2.4 → ONNX opset 17 · default flow (`optimization_level=0`).

**What we measured** (C = 64, 20×20, N(0,1) input)

| Model | statistics tensor in the DFC's decomposition | chip ≠ emulator | SNR chip / emulator |
|---|---|---|---|
| `GroupNorm(1, 64)` | 1 channel, 16-bit | **49.27%** (max 108 LSB) | **3.1** / 36.0 dB |
| `GroupNorm(2, 64)` | 2 channels | 0% | 35.6 / 35.6 dB |
| `GroupNorm(4, 64)` | 4 channels | 0% | 35.5 / 35.5 dB |
| `GroupNorm(8, 64)` | 8 channels | 0% | 35.2 / 35.2 dB |
| `LayerNorm(C)` over channels (ConvNeXt style) | 1 channel, no spatial reduce | 0.01% (1 LSB) | 36.5 / 36.5 dB |
| same maths rewritten (below) | 1 channel, 8-bit, AVGPOOL | 0% | 35.9 / 35.9 dB |

Earlier we also ran a 35-shape grid (C 8–128 × H = W 8–40): GroupNorm(1) is wrong at every shape, with 37–52% of outputs off by up to 131 LSB. When the HEF has two contexts, the output of a frame also depends on the previous frame.

**Where it goes wrong**
The DFC decomposes the norm itself (`decompose_norm_layer_algo_v2`). For GroupNorm(1) and GroupNorm(4) it produces the **same layer graph**: channel `reduce_mean` → spatial `reduce_mean` [1,2] → `resize` broadcast → `ew_sub` → `square` → `conv_var_inv` → spatial `reduce_mean` → `resize` → `ew_mult`, all a16_w16. The only difference is the width of the statistics tensor, 1 channel against 4.

The 16-bit spatial `reduce_mean`/`resize` of a **1-channel** map is the failing piece:
- With 2, 4 or 8 channels it is exact.
- The channel-only LayerNorm, which has the same 1-channel map but no spatial reduce/resize, is exact.
- The centring half built from plain ops is exact.

None of the `layer_norm_decomposition` options fixes it: equalization, nudging, `optimize_ew_mult` and `square_12_bit` leave it at 49% wrong; `split_precision` gives 31%; `a8_w8` does not compile; `ppu` is not supported on Hailo-8.

**Steps to reproduce**
1. Export the model:
```python
import torch, torch.nn as nn
class M(nn.Module):
    def __init__(s):
        super().__init__(); s.gn = nn.GroupNorm(1, 64); s.out = nn.Conv2d(64, 64, 1)
        with torch.no_grad(): s.gn.weight.uniform_(0.5, 1.5); s.gn.bias.normal_(0, 0.1)
    def forward(s, x): return s.out(s.gn(x))
torch.manual_seed(0)
torch.onnx.export(M().eval(), torch.randn(1, 64, 20, 20), "m.onnx", input_names=["x"], opset_version=17, dynamo=False)
```
2. `python check.py m.onnx` (`check.py` is below):
```
chip != emulator: 49.32% of outputs, max 108 LSB
SNR vs float: chip 3.1 dB, emulator 36.0 dB
chip batch == chip frame-by-frame: True
```
3. Change `nn.GroupNorm(1, 64)` to `nn.GroupNorm(4, 64)` and rerun: `chip != emulator: 0.00%`.

**Expected**: the chip equals `SDK_BIT_EXACT`, as it does for every other group count.
**Workaround**: compute the same function with plain ops (spatial mean first, then the channel mean). On the chip it is identical to the emulator:
```python
m = x.mean((2, 3), keepdim=True).mean(1, keepdim=True)
d = x - m
v = (d * d).mean((2, 3), keepdim=True).mean(1, keepdim=True)
y = d * torch.reciprocal(torch.sqrt(v + eps)) * weight.view(1, -1, 1, 1) + bias.view(1, -1, 1, 1)
```

[details="check.py"]
(same `check.py` as in the ConvTranspose post; paste it here)
[/details]

=====

**[Hailo-8] Dilated Conv2d on a 9–11 pixel wide row gives wrong results on the chip that change with what ran before**

A dilated 3×3 conv compiles without warning, but whenever the compiler gives it a row piece between 9 and 8 + dilation − 1 pixels wide, that piece is ~99% wrong on the Hailo-8, and its values depend on earlier frames (batch ≠ frame-by-frame). `SDK_BIT_EXACT` is correct.

**Environment**
Hailo-8 PCIe · DFC 3.33.1 · HailoRT 4.23.0 · Linux x86-64 · torch 2.4 → ONNX opset 17 · default flow.

**What we measured** (`Conv2d(24, 24, 3, padding=d, dilation=d)`, H = 30, N(0,1) input; "pieces" are the W slices from the DFC's own `defuse(..., defuse_type=SPATIAL)`)

Layer is compiled as one piece (no spatial split):

| W | d = 2 | d = 3 | d = 4 | d = 6 |
|---|---|---|---|---|
| 8 | | exact | | |
| 9 | **98.7% wrong**, −1.8 dB | **98.7% wrong**, −2.3 dB | **98.8% wrong**, −2.7 dB | |
| 10 | exact | **98.7% wrong**, −2.6 dB | **98.9% wrong**, −2.8 dB | |
| 11 | | (split) | **98.8% wrong**, −3.0 dB | |
| 13 | | | | exact |

Layer is split by the compiler (`defuse_type=SPATIAL`):

| W | d = 2 | d = 3 | d = 4 |
|---|---|---|---|
| 11–13 | | exact (2 slices) | exact (W = 12, 2 slices) |
| 17 | **52.2% wrong** (2 slices, one is 9 wide) | | |
| 18 | exact (3 slices) | **54.9% wrong** (2 slices of 9) | **54.8% wrong** (2 slices of 9) |
| 50 | exact (7 slices) | **19.7% wrong**: slices 16+16+8+10, columns 40–49 wrong, all others exact | **19.7% wrong** (4 slices) |
| 50, padded to 56 then cropped | | 0%, 37.9 dB | |

For dilation 2–4 the rule fits every case measured: **a piece is wrong exactly when its width is between 9 and 8 + d − 1**, that is, just past an 8-column boundary by less than the dilation. At dilation 6, width 13 is exact; widths 9–12 at d = 6 were not tested.

Emulator SNR is 37–38 dB in every case. In every wrong case `chip batch == chip frame-by-frame` is **False**, and the first run after loading the HEF also differs from later runs. That points to the piece reading stale memory, most likely right-edge padding columns that are never written. Changing `layer_output_buffers_format` or the defuse count (2, 5, 6) does not help.

**Steps to reproduce**
1. Export the model:
```python
import torch, torch.nn as nn
torch.manual_seed(0)
m = nn.Conv2d(24, 24, 3, padding=3, dilation=3).eval()
torch.onnx.export(m, torch.randn(1, 24, 30, 50), "m.onnx", input_names=["x"], opset_version=17, dynamo=False)
```
2. `python check.py m.onnx` (`check.py` is below):
```
chip != emulator: 19.73% of outputs, max 149 LSB
SNR vs float: chip 4.8 dB, emulator 37.9 dB
chip batch == chip frame-by-frame: False
```
The compiler's `auto.alls` shows the 4-way spatial split. Every output in columns 40–49, the 10-wide slice, is wrong; every other column is bit-exact.

3. Export the same conv at W = 9 (`torch.randn(1, 24, 30, 9)`): the whole output is ~99% wrong.

**Expected**: the chip equals `SDK_BIT_EXACT` and is independent of earlier frames.
**Workaround**: zero-pad W to a multiple of 8, run the conv, then crop. This is the same function, because the conv's own padding is zeros. With it the chip equals the emulator, at 37.9 dB:
```python
y = conv(F.pad(x, (0, 56 - 50)))[..., :50]   # export with static pads
```

[details="check.py"]
(same `check.py` as in the ConvTranspose post; paste it here)
[/details]

=====

**[Hailo-8] Attention softmax×V matmul is wrong on the chip when the compiler splits it with INPUT_FEATURES**

In plain single-head attention, the compiler splits the `softmax(QKᵀ)·V` matmul with `defuse(matmul2, 2, defuse_type=INPUT_FEATURES)`. That split gives 98% wrong outputs on the Hailo-8 (5 dB vs float), while `SDK_BIT_EXACT` gives 37–40 dB. Keeping the matmul whole makes the chip bit-exact.

**Environment**
Hailo-8 PCIe · DFC 3.33.1 · HailoRT 4.23.0 · Linux x86-64 · torch 2.4 → ONNX opset 17 · default flow.

**What we measured** (C = 32, `softmax(QKᵀ·t)·V`, N(0,1) input)

| tokens N, temperature t | default compile (P·V split by INPUT_FEATURES) | P·V pinned to 8 subclusters |
|---|---|---|
| 64, t = 0.05 | **98.68% wrong**, chip 5.5 / emulator 39.9 dB | 0%, 39.9 / 39.9 dB |
| 784, t = 0.2 | **98.34% wrong**, chip 5.3 / emulator 37.3 dB, batch ≠ frame-by-frame | 0%, 37.3 / 37.3 dB |
| 256, t = 0.2, K bias +3 | **98.19% wrong**, 5.4 / 22.0 dB | 0%, 22.0 / 22.0 dB |

Truncated models show `Q·Kᵀ` and the softmax bit-exact on the chip; only P·V is wrong. The error is deterministic, is not a permutation, and does not correlate with any operand. At 784 tokens the wrong output also depends on the batch.

The problem is not specific to softmax. In a plain Qᵀ·K probe (C = 32, 100 tokens, with the operand-signedness workaround from the separate matmul post applied; baseline 4.3% mismatch, 26.7 dB), forcing the same split with `matmul1_fs_0, matmul1_fs_1, matmul1_d0, matmul1_d1, matmul1_dc = defuse(matmul1, 2, defuse_type=INPUT_FEATURES)` makes the chip **96.1% wrong (3.8 dB)**, while the emulator stays at 33.4 dB. **An INPUT_FEATURES split of a dynamic-weights matmul is wrong on the chip whenever the compiler accepts it.**

The compiler validates this split for some matmuls. Forcing it on the attention Q·K is rejected with "Defuse features split 2 with unsupported alignment: matmul1", but for P·V it is accepted. At more than ~400 tokens the same split also causes the `_Map_base::at` crash when the compiler's own `auto.alls` is loaded back.

**Steps to reproduce**
1. Export the model:
```python
import torch, torch.nn as nn
class Attn(nn.Module):
    def __init__(s, c=32, t=0.2):
        super().__init__(); s.q, s.k, s.v, s.out = (nn.Conv2d(c, c, 1) for _ in range(4)); s.t = t
    def forward(s, x):
        b, c, h, w = x.shape
        q = s.q(x).reshape(b, c, h * w).transpose(1, 2)
        k = s.k(x).reshape(b, c, h * w)
        v = s.v(x).reshape(b, c, h * w).transpose(1, 2)
        p = torch.softmax((q @ k) * s.t, dim=-1)
        return s.out((p @ v).transpose(1, 2).reshape(b, c, h, w))
torch.manual_seed(0)
torch.onnx.export(Attn().eval(), torch.randn(1, 32, 28, 28), "m.onnx", input_names=["x"], opset_version=17, dynamo=False)
```
2. `python check.py m.onnx` (`check.py` is below). `auto.alls` contains `matmul2_fs_0, matmul2_fs_1, matmul2_d0, matmul2_d1, matmul2_dc = defuse(matmul2, 2, defuse_type=INPUT_FEATURES)`, and the output is:
```
chip != emulator: 98.34% of outputs, max 94 LSB
SNR vs float: chip 5.3 dB, emulator 37.3 dB
chip batch == chip frame-by-frame: False
```
3. Put this line in `pin.alls` and run `python check.py m.onnx pin.alls`:
```
compilation_param(matmul2, resources_allocation_strategy=manual_scs_selection, number_of_subclusters=8)
```
The output becomes `chip != emulator: 0.00%`, with SNR chip 37.3 dB and emulator 37.3 dB.

**Expected**: the chip equals `SDK_BIT_EXACT`, or the compiler rejects the split for P·V as it does for Q·K.
**Workaround**: pin every softmax×V matmul with the `compilation_param` line above (2 or 8 subclusters both work).

[details="check.py"]
(same `check.py` as in the ConvTranspose post; paste it here)
[/details]

=====

**[Hailo-8] Matmul with a ReLU (non-negative) second operand: the chip reads values above 127 as negative**

When the second operand of a dynamic matmul (K in Q·Kᵀ) comes from a ReLU, the Hailo-8 output is 88% wrong (9.6 dB vs float), while `SDK_BIT_EXACT` gives 27 dB. The chip writes the operand unsigned (0..255), and the matmul reads it as int8.

**Environment**
Hailo-8 PCIe · DFC 3.33.1 · HailoRT 4.23.0 · Linux x86-64 · torch 2.4 → ONNX opset 17 · default flow.

**What we measured** (`x → conv3×3 → ReLU → {conv1×1 → ReLU = Q, conv1×1 → ReLU = K} → Qᵀ·K (100×100) → conv1×1`, C = 32, 10×10)

| input amplitude | default: chip ≠ emulator, SNR chip / emulator | with `force_range_out=[0, 2·max]` on K |
|---|---|---|
| 1.0 (= calibration) | **88.06%**, max 56 LSB, **9.6** / 27.0 dB | 4.33%, 26.7 / 33.4 dB |
| 2.0 | **98.74%**, 1.3 / 8.9 dB | 87.0%, 6.1 / 7.8 dB |

**Mechanism.** We modelled it bit-exactly, matching 10,000 of 10,000 outputs on a capture.
- The quantizer marks the K producer **signed**: `base_hailo_layer.get_signed_output()` returns True for any 8-bit layer that feeds a matmul's weights input. The emulator therefore saturates K at 127.
- On the chip, that producer's activation and output stage runs **unsigned** (0..255) when its function is non-negative.
- The matmul then reads the stored byte as **int8**, so every K above 127 wraps negative.
- Either half alone leaves about 1,000 mismatches.

Linear (non-ReLU) K and V in our attention probes are bit-exact, chip equal to emulator.

**Steps to reproduce**
1. Export the model:
```python
import torch, torch.nn as nn, torch.nn.functional as F
class M(nn.Module):
    def __init__(s, c=32, h=10):
        super().__init__(); s.stem = nn.Conv2d(c, c, 3, 1, 1); s.q = nn.Conv2d(c, c, 1); s.k = nn.Conv2d(c, c, 1)
        s.out = nn.Conv2d(h * h, c, 1)
    def forward(s, x):
        y = F.relu(s.stem(x)); b, c, h, w = y.shape
        q = F.relu(s.q(y)).reshape(b, c, h * w).transpose(1, 2)
        k = F.relu(s.k(y)).reshape(b, c, h * w)
        return s.out((q @ k).transpose(1, 2).reshape(b, h * w, h, w))
torch.manual_seed(0)
torch.onnx.export(M().eval(), torch.randn(1, 32, 10, 10), "m.onnx", input_names=["x"], opset_version=17, dynamo=False)
```
2. `python check.py m.onnx` (`check.py` is below):
```
chip != emulator: 88.06% of outputs, max 56 LSB
SNR vs float: chip 9.6 dB, emulator 27.0 dB
```
3. Read the K producer's calibrated maximum from the HAR (here `conv2`, max 1.2565). Put this line in `fix.alls` and rerun with `python check.py m.onnx fix.alls`:
```
quantization_param(conv2, force_range_out=[0, 5.02614])
```
The chip then matches the emulator bit-exactly (0.00%, 32.2 dB) at amplitude 1 and 2. With 2·max (`[0, 2.51307]`) 4.3% stays wrong: the forced K scale is range/255, so values above half the forced range still wrap.

**Expected**: the quantizer encoding, the generated program and the emulator agree on the operand's signedness.
**Workaround**: the `force_range_out` line above on every non-negative producer of a matmul's second operand. Avoiding ReLU on K and V also works.

[details="check.py"]
(same `check.py` as in the ConvTranspose post; paste it here)
[/details]

=====

**[Hailo-8] Softmax denominator accumulator wraps around when inputs exceed the calibration range, which zeroes attention rows**

The quantizer gives the softmax denominator a 16-bit **wrapping** accumulator sized from the calibration maximum. Any attention row whose Σexp exceeds what calibration saw wraps around and gets a reciprocal of ≈0. At half the calibration amplitude a 784-token attention drops to −16 dB on both chip and emulator, so no DFC accuracy report shows it.

**Environment**
Hailo-8 PCIe · DFC 3.33.1 · HailoRT 4.23.0 · Linux x86-64 · torch 2.4 → ONNX opset 17 · default flow + the P·V pin from the separate softmax×V post, so that only this effect is measured.

**Where it is in the code**
`hailo_model_optimization/acceleras/hailo_layers/hailo_reduce_sum.py`:
```python
max_final_accumulator_by_channel = np.maximum(np.abs(pre_act_stats.min), np.abs(pre_act_stats.max))  # calibration stats
utilize_wraparound = self.act_op.is_negative_input()      # True for the softmax denominator (-sum of exp)
self.reduce_sum_op.create_hw_params(max_output_per_channel=max_final_accumulator_by_channel,
                                    utilize_wraparound=utilize_wraparound, ...)
if utilize_wraparound:  # fixes an edge case where resulted accumulator is exactly -2**15
```
`quantization_param(..., force_range_preact=...)` on the layer does not change the accumulator.

**What we measured** (784 tokens, C = 32; chip equals emulator in every case below)

| temperature | calibrated / worst-case denominator | input amplitude 0.25 / 0.5 / 1.0, default | same, accumulator sized N × max(exp) |
|---|---|---|---|
| t = 0.2 | 542.7 / 784 | 9.2 / 17.5 / 37.3 dB | 31.0 / 36.0 / 37.0 dB |
| t = 1.0 | 144.8 / 784 | **−14.6 / −16.3** / 17.8 dB | 4.4 / 9.2 / 13.8 dB |

From a 451,584-row map on a production model:

| row sum S | correct reciprocal | chip and emulator |
|---|---|---|
| 16,384–32,768 | 2–4 LSB | correct |
| 32,771–40,000 | 2 | 0–2 |
| 40,000–65,535 | 1–2 | **0** |

In a production YOLO26m (C2PSA attention, 784 tokens), 2.6% of attention rows on real images are above 32,770.

**Steps to reproduce**
1. Export the attention model from the softmax×V post with `t=1.0`.
2. Run `python check.py m.onnx pin.alls 0.5` (amplitude 0.5):
```
SNR vs float: chip -16.3 dB, emulator -16.3 dB
```
At amplitude 1.0 it is 17.8 dB. Smaller inputs give flatter softmax rows, and a larger row sum wraps.
3. Size the denominator for the worst case by running this before `runner.optimize()`:
```python
from hailo_model_optimization.acceleras.hailo_layers.hailo_reduce_sum import HailoReduceSum
orig = HailoReduceSum.create_hw_params
def patched(self, wc, target, hw_shifts=None):
    if "reduce_sum_softmax" in self.full_name:
        st = self.act_op.get_input_stats(0)
        worst = self.reduce_sum_op._reduce_size * float(np.max(np.abs(self.reduce_sum_op.get_input_limvals(0)))) \
                * float(np.max(np.abs(np.asarray(self.reduce_sum_op.kernel))))
        new = st._replace(min=np.minimum(st.min, -worst))
        self.act_op.get_input_stats = lambda i, _s=new, _g=self.act_op.get_input_stats: _s if i == 0 else _g(i)
    try: return orig(self, wc, target, hw_shifts=hw_shifts)
    finally: self.act_op.__dict__.pop("get_input_stats", None)
HailoReduceSum.create_hw_params = patched
```
The result at amplitude 0.5 is 9.2 dB, against −16.3 dB by default.

**Expected**: a saturating accumulator, or one sized from the reduce length (N × max exp) instead of the calibration set.
**Note**: a wider range spreads the 8-segment reciprocal table, which costs up to 4 dB in distribution (t = 1.0: 17.8 → 13.8 dB). A model QAT-trained on the wrapping behaviour should get the headroom before QAT.

[details="check.py"]
(same `check.py` as in the ConvTranspose post; paste it here)
[/details]

=====

**[Hailo-8] SDK_BIT_EXACT emulator clamps UINT16 network inputs at 32767, while the chip uses the full 0..65535 range**

For a network with a 16-bit (UINT16) input, the `SDK_BIT_EXACT` emulator behaves as if every raw input value were clamped to 32767. The Hailo-8 chip uses the full unsigned range, so emulator-based validation of 16-bit models is wrong for any input above 32767.

**Environment**
Hailo-8 PCIe · DFC 3.33.1 · HailoRT 4.23.0 · Linux x86-64 · torch 2.4 → ONNX opset 17 · all layers `a16_w16`.

**What we measured** (`conv1×1 → y + y + y → conv1×1`, C = 16, 24×24, input zero point 16,006; a frame of zero point everywhere except one element set to the raw value)

| raw input element | chip output (same element) | emulator output | output elements that differ |
|---|---|---|---|
| 30,000 | 25,066 | 25,066 | 0 |
| 32,000 | 26,012 | 26,012 | 0 |
| 32,767 | 26,375 | 26,375 | 0 |
| 33,000 | 26,485 | **26,375** | 16 |
| 35,000 | 27,432 | **26,375** | 16 |
| 40,000 | 29,798 | **26,375** | 16 |
| 50,000 / 65,535 | 32,767 (saturated) | **26,375** | 16 |

The emulator output is frozen at the value for 32,767 for every input from 32,767 up. With in-distribution N(0,1) inputs, chip and emulator agree except for 1-LSB differences on 0.69% of outputs.

Also, `InputVStreamParams.make(ng)` with the default `FormatType.AUTO` rejects uint16 arrays for this input ("numpy array item size is 2, not 1"). Passing `format_type=FormatType.UINT16` works.

**Steps to reproduce**
1. Export the model:
```python
import torch, torch.nn as nn
class M(nn.Module):
    def __init__(s, c=16):
        super().__init__(); s.a = nn.Conv2d(c, c, 1); s.b = nn.Conv2d(c, c, 1)
    def forward(s, x): y = s.a(x); return s.b(y + y + y)
torch.manual_seed(0)
torch.onnx.export(M().eval(), torch.randn(1, 16, 24, 24), "m.onnx", input_names=["x"], opset_version=17, dynamo=False)
```
2. Put this line in `a16.alls` (layer names as parsed by DFC 3.33.1):
```
quantization_param([input_layer1, conv1, mul_and_add1, ew_add1, conv2, output_layer1], precision_mode=a16_w16)
```
3. Build with `python check.py m.onnx a16.alls`. Then feed frames that are `qp_zp` everywhere except `X[k, 12, 12, 0] = v` for v = 30000 … 65535, to both the chip (with `FormatType.UINT16`) and `SDK_BIT_EXACT`, and compare element [12, 12, 0] of the outputs.

**Expected**: the emulator uses the full UINT16 range, as the chip does.

=====

**[Hailo-8] ConvTranspose 3×3, stride 2, padding 1 parses to a 48×48 output where ONNX gives 47×47**

`ConvTranspose2d(16, 16, 3, stride=2, padding=1)` on a 24×24 input has a 47×47 output in PyTorch/ONNX, but the DFC parses it to 48×48 without any warning. The HAR's float output therefore cannot match the model. Hailo's 48×48 output is the ONNX output plus one extra **leading** row and column: `hailo[:, 1:, 1:]` equals ONNX exactly (139 dB, float emulator). The parser seems to build every deconv as TF-SAME (output = input × stride, odd padding at the end) and drop the ONNX begin pad. Workaround: SAME pads `[0, 0, 1, 1]` + Slice `[1:48]` (chip exact, 35.0 dB).

**Environment**
DFC 3.33.1 · torch 2.4 → ONNX opset 17 · no `output_padding`.

**What we measured**
- onnxruntime output: (1, 16, **47, 47**), from (24 − 1)·2 − 2·1 + 3 = 47.
- `ClientRunner.translate_onnx_model()`, then `SDK_NATIVE` inference: (1, **48, 48**, 16).
- The parse succeeds, and quantization and compile complete.

This looks related to the known `output_padding` thread, but here there is no output_padding at all.

**Steps to reproduce**
```python
import torch, torch.nn as nn, numpy as np
from hailo_sdk_client import ClientRunner, InferenceContext
m = nn.Sequential(nn.ConvTranspose2d(16, 16, 3, 2, 1), nn.Conv2d(16, 16, 1)).eval()
torch.onnx.export(m, torch.randn(1, 16, 24, 24), "m.onnx", input_names=["x"], opset_version=17, dynamo=False)
r = ClientRunner(hw_arch="hailo8"); r.translate_onnx_model("m.onnx", "probe")
with r.infer_context(InferenceContext.SDK_NATIVE) as c:
    print(np.asarray(r.infer(c, np.zeros((1, 24, 24, 16), np.float32))).shape)   # (1, 48, 48, 16), ONNX: 47x47
```

**Expected**: the parsed output shape equals the ONNX shape, or the parser raises an error.
**Workaround**: use k4 s2 p1 (48×48, exact on the chip) and adjust the model, or crop explicitly.

=====
