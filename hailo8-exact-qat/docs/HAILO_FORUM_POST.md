**[Hailo-8 / DFC 3.33.1] 8 cases where the HEF compiles cleanly and passes SDK_BIT_EXACT, but the chip output is wrong (repros + workarounds)**

Several common layers compile with no warning and look correct in the emulator, but give wrong results on the Hailo-8. Each case below has a minimal PyTorch repro, measured numbers, the DFC rule we traced it to, and a workaround.

**Environment**
- Hailo-8 PCIe (two devices, both reproduce everything), Dataflow Compiler 3.33.1, HailoRT 4.23.0 with its bundled firmware, Linux x86-64.
- PyTorch 2.4 export to ONNX opset 17.
- Default flow: `optimization_level=0`, `compression_level=0`, 64 N(0,1) calibration frames, no other model-script lines unless stated.

**Method**
`check.py` (at the end of this post) builds the ONNX with that flow. It then runs the same 8 raw quantized frames on the chip, in `SDK_BIT_EXACT` and in onnxruntime. It reports:
- the percentage of output elements where the chip differs from the emulator;
- SNR vs float for both;
- whether batch output equals frame-by-frame output.

The chip is deterministic: identical across runs, reboots and both devices, except where noted. All numbers were measured on 2026-10-26 with exactly the steps below.

| # | Case | Chip ≠ emulator | SNR vs float, chip / emulator | Workaround |
|---|---|---|---|---|
| 1 | ConvTranspose with kernel = stride (2/2, 4/4) | 99.5% | −9.6 / 35.1 dB | force `SUPER_DECONV` |
| 2 | GroupNorm(1, C) | 49.3% | 3.1 / 36.0 dB | rewrite with plain ops |
| 3 | Dilated conv on a 9–11-px-wide piece | ~99% of that piece; changes with earlier frames | −2.6 / 37.9 dB | pad W to a multiple of 8, then crop |
| 4 | INPUT_FEATURES split of a dynamic matmul (attention P·V) | 98.3% | 5.3 / 37.3 dB | pin the matmul's subclusters |
| 5 | Matmul whose second operand comes from a ReLU | 88.1% | 9.6 / 27.0 dB | `force_range_out` on the producer |
| 6 | Softmax denominator wraps off-calibration | chip = emulator, both wrong | −16.3 / −16.3 dB | wider denominator range |
| 7 | Emulator clamps UINT16 inputs at 32767 | emulator wrong, chip right | — | — |
| 8 | ConvTranspose k3 s2 p1 parses to 48×48, not 47×47 | wrong shape | — | SAME pads + crop (exact) |

---

### 1. ConvTranspose with kernel = stride is wrong on the chip

`ConvTranspose2d(C, C, 2, stride=2)` and `(4, stride=4)` are 99.5% wrong on the chip, while the emulator is correct.

| ConvTranspose | DFC route (`auto.alls`) | Chip ≠ emulator | SNR chip / emulator |
|---|---|---|---|
| k2 s2, C16 24×24 | `defuse(deconv1, 1, defuse_type=DECONV)` | **99.50%** (max 212 LSB) | **−9.6** / 35.1 dB |
| k2 s2, C64 16×16 | DECONV | **99.54%** | **−10.5** / 34.8 dB |
| k4 s4, C16 24×24 | DECONV | **99.48%** | **−6.2** / 38.2 dB |
| k4 s4, C64 12×12 | DECONV | **99.48%** | **−9.1** / 36.2 dB |
| k4 s2 p1 | DECONV | 0% | 36.3 / 36.3 dB |
| k8 s4 p2 | DECONV | 0% | 38.6 / 38.6 dB |
| k16 s8 p4 | DECONV | 0% | 41.4 / 41.4 dB |
| k2 s2 with `SUPER_DECONV` forced, same HAR | SUPER_DECONV | 0% | 35.1 / 35.1 dB |

**Route.** `hailo_sdk_common/numeric_utils/numeric_utils.py::is_super_deconv()` sends exactly five kernel/stride pairs to the native DECONV (feature-interleave) path and everything else to SUPER_DECONV:
```python
return not ((k == 4 and s == 2) or (k == 4 and s == 4) or (k == 2 and s == 2)
            or (k == 8 and s == 4) or (k == 16 and s == 8))
```
- **kernel = 2 × stride** (k4s2, k8s4, k16s8): bit-exact.
- **kernel = stride** (k2s2, k4s4), the standard non-overlapping upsamplers: wrong, deterministically. The output is not a permutation of the correct one.

**Steps**
```python
import torch, torch.nn as nn
class M(nn.Module):
    def __init__(s):
        super().__init__(); s.up = nn.ConvTranspose2d(16, 16, 2, 2); s.out = nn.Conv2d(16, 16, 1)
    def forward(s, x): return s.out(s.up(x))
torch.manual_seed(0)
torch.onnx.export(M().eval(), torch.randn(1, 16, 24, 24), "m.onnx", input_names=["x"], opset_version=17, dynamo=False)
```
`python check.py m.onnx` prints:
```
chip != emulator: 99.50% of outputs, max 212 LSB
SNR vs float: chip -9.6 dB, emulator 35.1 dB
```
Put the line below in `super.alls` and run `python check.py m.onnx super.alls`. The result is `chip != emulator: 0.00%`, 35.1 / 35.1 dB.
```
deconv1_defuse_conv, deconv1_defuse_d2s = defuse(deconv1, 1, defuse_type=SUPER_DECONV)
```
**Workaround.** The `SUPER_DECONV` line above, or in ONNX a 1×1 conv (C → 4C) followed by `DepthToSpace(2, CRD)`. Both are bit-exact on the chip.

---

### 2. GroupNorm(1, C) is wrong on the chip

`nn.GroupNorm(1, C)` gives 49% wrong outputs on the chip. GroupNorm with 2, 4 or 8 groups is exact.

| Model (C = 64, 20×20) | Statistics tensor in the DFC's decomposition | Chip ≠ emulator | SNR chip / emulator |
|---|---|---|---|
| `GroupNorm(1, 64)` | 1 channel, 16-bit | **49.27%** (max 108 LSB) | **3.1** / 36.0 dB |
| `GroupNorm(2, 64)` | 2 channels | 0% | 35.6 / 35.6 dB |
| `GroupNorm(4, 64)` | 4 channels | 0% | 35.5 / 35.5 dB |
| `GroupNorm(8, 64)` | 8 channels | 0% | 35.2 / 35.2 dB |
| channel-only `LayerNorm(C)` (ConvNeXt style) | 1 channel, no spatial reduce | 0.01% (1 LSB) | 36.5 / 36.5 dB |
| same maths rewritten (below) | 1 channel, 8-bit, AVGPOOL | 0% | 35.9 / 35.9 dB |

A 35-shape grid (C 8–128 × H = W 8–40) is wrong at every shape: 37–52% of outputs, by up to 131 LSB. With two contexts, a frame's output also depends on the previous frame.

**Route.** `decompose_norm_layer_algo_v2` builds the **same layer graph** for GroupNorm(1) and GroupNorm(4):

channel `reduce_mean` → spatial `reduce_mean` → `resize` → `ew_sub` → `square` → `conv_var_inv` → spatial `reduce_mean` → `resize` → `ew_mult`, all a16_w16.

The only difference is the width of the statistics tensor: 1 channel vs 4. The failing piece is the **16-bit spatial reduce_mean/resize of a 1-channel map**:
- 2, 4 or 8 channels are exact.
- A 1-channel map without the spatial step (channel LayerNorm) is exact.

No `layer_norm_decomposition` option fixes it:
- equalization, nudging, `optimize_ew_mult` and `square_12_bit`: still 49% wrong;
- `split_precision`: 31% wrong;
- `a8_w8`: does not compile;
- `ppu`: not supported on Hailo-8.

**Steps**
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
`python check.py m.onnx` prints `chip != emulator: 49.32% ... SNR vs float: chip 3.1 dB, emulator 36.0 dB`. With `GroupNorm(4, 64)` the result is `0.00%`.

**Workaround.** The same function with plain ops; the chip then equals the emulator:
```python
m = x.mean((2, 3), keepdim=True).mean(1, keepdim=True)
d = x - m
v = (d * d).mean((2, 3), keepdim=True).mean(1, keepdim=True)
y = d * torch.reciprocal(torch.sqrt(v + eps)) * weight.view(1, -1, 1, 1) + bias.view(1, -1, 1, 1)
```

---

### 3. Dilated conv is wrong on 9–11-pixel-wide pieces, and depends on earlier frames

When the compiler gives a dilated 3×3 conv a row piece between 9 and 8 + dilation − 1 pixels wide, that piece is ~99% wrong on the chip. Its values also change with what ran before.

`Conv2d(24, 24, 3, padding=d, dilation=d)`, H = 30.

Layer compiled as one piece:

| W | d = 2 | d = 3 | d = 4 | d = 6 |
|---|---|---|---|---|
| 8 | | exact | | |
| 9 | **98.7% wrong**, −1.8 dB | **98.7% wrong**, −2.3 dB | **98.8% wrong**, −2.7 dB | |
| 10 | exact | **98.7% wrong**, −2.6 dB | **98.9% wrong**, −2.8 dB | |
| 11 | | (split) | **98.8% wrong**, −3.0 dB | |
| 13 | | | | exact |

Layer split by the compiler (`defuse_type=SPATIAL`):

| W | d = 2 | d = 3 | d = 4 |
|---|---|---|---|
| 11–13 | | exact (2 slices) | exact (W = 12) |
| 17 | **52.2% wrong** (one 9-wide slice) | | |
| 18 | exact (3 slices) | **54.9% wrong** (2 slices of 9) | **54.8% wrong** (2 slices of 9) |
| 50 | exact (7 slices) | **19.7% wrong**: slices 16+16+8+10, only columns 40–49 wrong | **19.7% wrong** |
| 50, padded to 56 then cropped | | 0%, 37.9 dB | |

**Rule (d = 2–4, every case measured).** A piece is wrong exactly when its width is 9 … 8 + d − 1, that is, just past an 8-column boundary by less than the dilation. At d = 6, width 13 is exact; widths 9–12 at d = 6 are untested.

- The emulator gives 37–38 dB in every case.
- In every wrong case, the chip's batch output ≠ its frame-by-frame output, and the first run after loading the HEF differs from later runs. That looks like the piece reading stale memory, probably right-edge padding that is never written.
- Changing `layer_output_buffers_format` or the defuse count does not help. Compiler string: "columns dilation > 2 is not supported for convmxns1_no_contexts".

**Steps**
```python
import torch, torch.nn as nn
torch.manual_seed(0)
m = nn.Conv2d(24, 24, 3, padding=3, dilation=3).eval()
torch.onnx.export(m, torch.randn(1, 24, 30, 50), "m.onnx", input_names=["x"], opset_version=17, dynamo=False)
```
`python check.py m.onnx` prints:
```
chip != emulator: 19.73% of outputs, max 149 LSB
SNR vs float: chip 4.8 dB, emulator 37.9 dB
chip batch == chip frame-by-frame: False
```
At W = 9 the whole output is ~99% wrong.

**Workaround.** Zero-pad W to a multiple of 8, run the conv, then crop. This is the same function, and it is exact on the chip:
```python
y = conv(F.pad(x, (0, 56 - 50)))[..., :50]   # export with static pads
```

---

### 4. INPUT_FEATURES split of a dynamic matmul is wrong on the chip (attention softmax×V)

In plain attention the compiler splits `softmax(QKᵀ)·V` with INPUT_FEATURES. That split is 98% wrong on the chip; keeping the matmul whole is bit-exact.

| Attention (C = 32) | Default compile, `defuse(matmul2, 2, defuse_type=INPUT_FEATURES)` | P·V pinned to 8 subclusters |
|---|---|---|
| 64 tokens, t = 0.05 | **98.68% wrong**, 5.5 / 39.9 dB | 0%, 39.9 / 39.9 dB |
| 784 tokens, t = 0.2 | **98.34% wrong**, 5.3 / 37.3 dB, batch ≠ frame-by-frame | 0%, 37.3 / 37.3 dB |
| 256 tokens, K bias +3 | **98.19% wrong**, 5.4 / 22.0 dB | 0%, 22.0 / 22.0 dB |

- Q·Kᵀ and the softmax alone are bit-exact on the chip; only P·V is wrong. The error is not a permutation and does not correlate with any operand.
- **It is not specific to softmax.** On the Qᵀ·K probe of case 5, with case 5's fix applied (4.3% baseline), forcing `matmul1_fs_0, matmul1_fs_1, matmul1_d0, matmul1_d1, matmul1_dc = defuse(matmul1, 2, defuse_type=INPUT_FEATURES)` gives **96.1% wrong (3.8 dB)**, while the emulator gives 33.4 dB.
- The compiler rejects this split for the attention Q·K ("Defuse features split 2 with unsupported alignment"), but accepts it for P·V.
- At more than ~400 tokens, the same split also causes `_Map_base::at` when the compiler's own `auto.alls` is loaded back.

**Steps**
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
`python check.py m.onnx` prints `chip != emulator: 98.34% ... chip 5.3 dB, emulator 37.3 dB`. With the line below in `pin.alls`, `python check.py m.onnx pin.alls` gives `0.00%`, 37.3 / 37.3 dB.
```
compilation_param(matmul2, resources_allocation_strategy=manual_scs_selection, number_of_subclusters=8)
```
**Workaround.** Pin every softmax×V matmul as above (2 or 8 subclusters both work).

---

### 5. Matmul with a ReLU second operand: the chip reads values above 127 as negative

When K in Q·Kᵀ comes from a ReLU, the chip output is 88% wrong. The chip writes K unsigned (0..255), and the matmul reads it as int8.

Probe: `x → conv3×3 → ReLU → {conv1×1 → ReLU = Q, conv1×1 → ReLU = K} → Qᵀ·K → conv1×1`, C = 32, 10×10.

| Input amplitude | Default | `force_range_out=[0, 2·max]` on K | `force_range_out=[0, 4·max]` on K |
|---|---|---|---|
| 1.0 (= calibration) | **88.06% wrong**, **9.6** / 27.0 dB | 4.33%, 26.7 / 33.4 dB | **0.00%**, 32.2 / 32.2 dB |
| 2.0 | **98.74%**, 1.3 / 8.9 dB | 87.0%, 6.1 / 7.8 dB | **0.00%**, 8.4 / 8.4 dB |

**Mechanism** (modelled bit-exactly, 10,000 of 10,000 outputs):
- `base_hailo_layer.get_signed_output()` marks any 8-bit producer of a matmul's weights input as **signed**, so the emulator saturates K at 127.
- On the chip, that producer's activation and output stage runs **unsigned** (0..255) when its function is non-negative.
- The matmul then reads the byte as **int8**, so K values above 127 wrap negative.
- A forced range does not change this: the forced K scale is range/255 (`qp_out` = 2.51307/255 for `[0, 2.51307]`), so values above half the forced range still wrap. A 2× range leaves the tail above the calibrated max wrapping (4.33%); 4× is bit-exact.

Linear (non-ReLU) K and V are exact.

**Steps**
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
`python check.py m.onnx` prints `88.06% ... chip 9.6 dB, emulator 27.0 dB`. The K producer here is `conv2`, with calibrated max 1.2565. With the line below in `fix.alls`, the chip matches the emulator bit-exactly (32.2 dB).
```
quantization_param(conv2, force_range_out=[0, 5.02614])
```

---

### 6. Softmax denominator accumulator wraps when inputs leave the calibration range

The quantizer gives the softmax denominator a 16-bit **wrapping** accumulator sized from calibration. Rows whose Σexp exceed it get a reciprocal of ≈0. Chip and emulator agree, so no DFC report shows it.

`hailo_model_optimization/acceleras/hailo_layers/hailo_reduce_sum.py`:
```python
max_final_accumulator_by_channel = np.maximum(np.abs(pre_act_stats.min), np.abs(pre_act_stats.max))  # calibration
utilize_wraparound = self.act_op.is_negative_input()      # True for the softmax denominator (-sum exp)
if utilize_wraparound:  # fixes an edge case where resulted accumulator is exactly -2**15
```
`force_range_preact` does not change the accumulator.

Measured on the case-4 model, 784 tokens, P·V pinned. Chip = emulator in every row:

| Temperature | Calibrated / worst-case denominator | Amplitude 0.25 / 0.5 / 1.0, default | Same, sized N × max(exp) |
|---|---|---|---|
| t = 0.2 | 542.7 / 784 | 9.2 / 17.5 / 37.3 dB | 31.0 / 36.0 / 37.0 dB |
| t = 1.0 | 144.8 / 784 | **−14.6 / −16.3** / 17.8 dB | 4.4 / 9.2 / 13.8 dB |

From a 451,584-row map on a production model:
- row sums 32,771–40,000 get a reciprocal of 0–2 instead of 2;
- row sums 40,000–65,535 get **0** instead of 1–2.

In production YOLO26m (C2PSA, 784 tokens), 2.6% of attention rows on real images are above 32,770.

**Steps.** Export the case-4 model with `t=1.0`, then run `python check.py m.onnx pin.alls 0.5`. It prints `chip -16.3 dB, emulator -16.3 dB`; at amplitude 1.0 the result is 17.8 dB.

**Fix used.** Run this before `optimize()`; at amplitude 0.5 it gives 9.2 dB:
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
It costs up to 4 dB in distribution (t = 1.0: 17.8 → 13.8 dB), because the 8-segment reciprocal table is spread over a wider range. A saturating accumulator would avoid the trade-off.

---

### 7. SDK_BIT_EXACT clamps UINT16 inputs at 32767; the chip uses 0..65535

For a 16-bit-input network, the emulator behaves as if raw inputs were clamped to 32767. The chip uses the full range.

Probe: `conv1×1 → y + y + y → conv1×1`, C = 16, 24×24, every layer a16_w16, input zero point 16,006. The frame is the zero point everywhere except one element.

| Raw input element | Chip output | Emulator output | Output elements that differ |
|---|---|---|---|
| 30,000 / 32,000 / 32,767 | 25,066 / 26,012 / 26,375 | same | 0 |
| 33,000 | 26,485 | **26,375** | 16 |
| 35,000 | 27,432 | **26,375** | 16 |
| 40,000 | 29,798 | **26,375** | 16 |
| 50,000 / 65,535 | 32,767 | **26,375** | 16 |

**Steps**
```python
import torch, torch.nn as nn
class M(nn.Module):
    def __init__(s, c=16):
        super().__init__(); s.a = nn.Conv2d(c, c, 1); s.b = nn.Conv2d(c, c, 1)
    def forward(s, x): y = s.a(x); return s.b(y + y + y)
torch.manual_seed(0)
torch.onnx.export(M().eval(), torch.randn(1, 16, 24, 24), "m.onnx", input_names=["x"], opset_version=17, dynamo=False)
```
1. Put this line in `a16.alls`:
```
quantization_param([input_layer1, conv1, mul_and_add1, ew_add1, conv2, output_layer1], precision_mode=a16_w16)
```
2. Run `python check.py m.onnx a16.alls`.
3. Feed frames that are `qp_zp` everywhere except `X[k, 12, 12, 0] = v` to the chip and to `SDK_BIT_EXACT`, and compare element [12, 12, 0].

Also, `FormatType.AUTO` rejects uint16 arrays for this input ("numpy array item size is 2, not 1"). Pass `FormatType.UINT16` explicitly.

---

### 8. ConvTranspose k3 s2 p1 parses to 48×48 where ONNX gives 47×47

With no `output_padding`, `ConvTranspose2d(16, 16, 3, 2, 1)` on a 24×24 input is 47×47 in ONNX: (24 − 1)·2 − 2 + 3 = 47. The DFC parses it to 48×48 without a warning. Hailo's 48×48 output is the ONNX output plus one extra **leading** row and column: `hailo[:, 1:, 1:]` equals ONNX exactly (139 dB, float emulator). The parser seems to build every deconv as TF-SAME (output = input × stride, odd padding at the end) and drop the ONNX begin pad. Workaround: SAME pads `[0, 0, 1, 1]` + Slice `[1:48]` (chip exact, 35.0 dB).

```python
import torch, torch.nn as nn, numpy as np
from hailo_sdk_client import ClientRunner, InferenceContext
m = nn.Sequential(nn.ConvTranspose2d(16, 16, 3, 2, 1), nn.Conv2d(16, 16, 1)).eval()
torch.onnx.export(m, torch.randn(1, 16, 24, 24), "m.onnx", input_names=["x"], opset_version=17, dynamo=False)
r = ClientRunner(hw_arch="hailo8"); r.translate_onnx_model("m.onnx", "probe")
with r.infer_context(InferenceContext.SDK_NATIVE) as c:
    print(np.asarray(r.infer(c, np.zeros((1, 24, 24, 16), np.float32))).shape)   # (1, 48, 48, 16)
```
This is related to the existing `output_padding` thread, but no output_padding is involved here.

---

### What would help

1. Confirmation of cases 1–5, and whether they are fixed in a newer DFC or firmware.
2. For cases 1 and 2: is the cause the generated program or the hardware unit? We cannot tell from outside.
3. A saturating softmax-denominator option (case 6), and emulator fixes for case 7, so that `SDK_BIT_EXACT` can be trusted as a pre-silicon check.
4. Compile-time errors instead of silently wrong HEFs for cases 1, 3 and 4, and a parse error for case 8.

Happy to share the ONNX files, HARs and HEFs.

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
