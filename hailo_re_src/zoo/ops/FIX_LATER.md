# Hailo-8 / DFC 3.33.1 operator issues to fix later

Source: `zoo/ops/probe.py` (hand-built ONNX, 169 cases) and `zoo/ops/torch_cases.py` (torch.onnx.export opset 17,
88 cases), each `x -> op -> identity 1x1 conv`, level-0 quantization, bit-exact HSim vs onnxruntime float.
Baseline (no op, 8-bit in/out) = 39.0 dB. Results per case: `zoo/ops/res/<case>/result.json`.

Status legend: **FAIL** = parse/compile error; **SILENT** = compiles but computes the wrong function (worst kind);
**LOSSY** = runs but far below the 39 dB floor; **EMU** = runs on chip but HSim cannot bit-exact emulate it.

## A. Silent wrong results (compile fine, numbers wrong)
| Op | float-emu SQNR | chip SQNR | What happens | Planned fix |
|---|---|---|---|---|
| Pad mode=reflect / edge (`nn.ReflectionPad2d`, `ReplicationPad2d`) | 8.3 dB | 8.3 dB | parsed as zero padding | rewrite: Slice + Concat (exact) |
| Floor (op alone) | 8.9 dB | 8.9 dB | node dropped by the parser | host, or PWL staircase over the calibrated range |
| ConvTranspose k3 s2 p1 output_padding=1 | ? | -3.0 dB | wrong output geometry | rewrite to an equivalent k4 s2 p1 deconv (zero-embedded kernel) |
| Pow(x, -0.5) | 147 dB | -10.7 dB | `pow` activation quantized wrongly | rewrite to Reciprocal(Sqrt) -> `inv_sqrt` (35.6 dB) |
| Max(a, b) two tensors | 352 dB | 10.7 (torch) / 27.5 dB | concat + reduce_max at one shared scale | rewrite a + relu(b - a); test |
| Gemm / GlobalAveragePool -> dense | 124 dB | 8-11 dB | precision collapse at the pooled 1x1 | investigate (16-bit pool output?) |

## B. Parse failures with an exact or near-exact rewrite
| Op | Error | Rewrite |
|---|---|---|
| ONNX attribute defaults omitted (BatchNorm `epsilon`, ...) | IndexError in parser | `fill_defaults()` (probe.py): write schema defaults explicitly. torch export already does |
| Gelu (opset 20 op), GELU erf pattern in non-torch order | IndexError / UnsupportedActivation | emit torch's order `Div(x,√2)->Erf->Add(1)->Mul(x)->Mul(0.5)` -> native `gelu` (34.6 dB) |
| GELU tanh form | compile `map::at` | rewrite to erf GELU (max diff 5e-4) or PWL |
| Pow(x, 3), Pow(x, 1.5), Pow(tensor, tensor) | IndexError / input_shapes | x*x*x; x*sqrt(x); exp(y*log(x)) for x>0 |
| ReduceMax / ReduceMin over W or H*W | IndexError | MaxPool kernel (1,W) / GlobalMaxPool (48 dB); min = -max(-x) |
| ReduceL1 | IndexError | ReduceSum(Abs) |
| ReduceLogSumExp | IndexError | max + log(sum(exp(x - max))) (= LogSoftmax parts, 43.7 dB) |
| ReduceProd | IndexError | exp(sum(log x)) for x > 0; else host |
| SpaceToDepth (ONNX op) | IndexError | torch PixelUnshuffle export already works (39 dB) |
| Flip (Slice step -1) | IndexError | transpose W<->C, channel Gather/permutation conv, transpose back |
| CumSum along W | IndexError | depthwise conv with ones kernel (1,W) left-padded, or log2(W) shift-and-add |
| Where(cond, a, b), Softshrink, Hardshrink | Where unsupported | b + mask*(a-b) with mask = `greater` activation; shrink = x - clip(x) |
| Split -> both halves into one node | `'bias_add1' is not in list` | use two Slices (Slice_C works) |
| Expand | UnexpectedNode | Tile or Resize nearest (Tile_W works) |
| Resize bicubic | IndexError | separable fixed-kernel ConvTranspose (4 taps per phase) + DepthToSpace |
| LRN | invalid kernel | x * pow(k + a/n * conv1x1_band(x^2), -b) via square + band conv + PWL |
| MatMul with a constant (last-axis dense on 4-D) | invalid kernel shape | transpose to channels + 1x1 conv (Linear_tokens works, 34.5 dB) |
| Einsum (generic) | UnsupportedEinsum | per-equation rewrite to MatMul / Mul + ReduceSum |
| Attention via manual reshape / SDPA | Reshape / Softmax layout | nn.MultiheadAttention token layout parses (ViT_block 32.7 dB); rewrite to that layout |
| LSTM / GRU / RNN | Reshape layout | find the layout the DFC expects; else unroll into conv + PWL gates |

## FIXED by `onnx_hailo_compat.py` (verified: onnxruntime equivalence + DFC parse/quantize/bit-exact/compile)
| Case | Before | After (chip, bit-exact) | Rule |
|---|---|---|---|
| Pad reflect / edge (hand + torch Reflection/ReplicationPad2d) | 8.3 dB SILENT | **39.0 dB** (= floor) | fold + pad_mode -> Slice+Concat |
| Pow(x,-0.5) | -10.7 dB SILENT | **35.6 dB** | pow -> Reciprocal(Sqrt) = inv_sqrt |
| Max(a,b) tensors (torch / hand) | 10.7 / 27.5 dB | **34.7 / 34.2 dB** | ew_minmax -> a + Relu(b-a) |
| Min(a,b) tensors | 33.1 dB | 34.0 dB | a - Relu(a-b) |
| Pow(x,3), Pow(x,1.5) | parse FAIL | 40.1 / 48.1 dB | pow |
| ReduceMax/Min over W or HW (incl. torch amax/amin) | parse FAIL | 43.7-48.0 dB | reduce_pool -> MaxPool / GlobalMaxPool |
| ReduceL1 | parse FAIL | 46.0-47.4 dB | Abs + ReduceSum |
| ArgMin | parse FAIL | 20.0 dB (= ArgMax 21.4) | ArgMax(Neg) |
| Split into one node | parse FAIL | 36.0 dB | split -> Slices |
| Expand | parse FAIL | 36.4 dB | Tile |
| Gelu op20 (erf / tanh), torch tanh-GELU, hand tanh-GELU | parse / compile FAIL | 34.6 dB native gelu | gelu |
| Shrink op / torch Softshrink (Where pattern) | parse FAIL | 30.6 / 31.1 dB | shrink; NB -Relu(-v) chain quantizes to 7.9 dB, use v - Relu(v) |
Still open from B: GELU erf in non-torch node orders (needs a pattern normalizer), ConvTranspose output_padding,
LRN, Einsum, CumSum, Flip, bicubic, attention layouts, LSTM/GRU.

## FIXED (batch 2: compat2.py, verified onnxruntime-equal + DFC bit-exact)
| Case | After (chip dB) | Rule |
|---|---|---|
| ReduceLogSumExp C / W / HW, torch logsumexp | 49.0 / 47.8 / 46.7 / 49.0 | max + log(sum(exp(x - max))) (+ 2nd pass to pool the new ReduceMax) |
| LRN | 39.0 | x / (bias + alpha/size * band 1x1 conv(x^2))^beta |
| Flip (Slice step -1) | 39.0 (exact) | reversed single slices + Concat |
| Pow(tensor, tensor) | 40.0 | exp(y log x), x > 0 |
| LpNormalization | 36.7 | x / ReduceL2 (or / sum|x|) |
| MeanVarianceNormalization | 36.1 | InstanceNorm(scale 1, bias 0) |
| LpPool | 35.3 | sqrt(k * AvgPool(x^2)) |
| MatMul with a constant over W | 34.8 | one depthwise (1, W) conv per output column + Concat (a channel->width Reshape is mistranslated: -3 dB) |
| GELU erf in any node order | 34.6 (native gelu) | numerically verified pattern -> torch order |
| LayerNorm over W | 34.4 | W<->C transposes around the channel LayerNorm pattern |
| torch Hardshrink (Greater/Less/Or/Where) | 33.6 | boolean graph -> 0/1 arithmetic |
| GroupNormalization (opset 18) | 33.3 | Reshape -> InstanceNorm -> affine, BEFORE constant folding (ORT inlines it) |
| Where(c, a, b) | 32.1 | b + m (a - b) |
| CumSum (H / W) | 30.1 | log-step shift-and-add |
| Mod, Equal | via Floor/Sign/Round PWL steps | x - m floor(x/m) (fmod: trunc = sign * floor|x|); window mask |
Lossy (on chip but poor): ReduceProd exp(sum log) 8.9 dB; Hardmax ramp 6-9 dB -> host (see HOST_VS_CHIP.md).
Host-only (no on-chip form): TopK, ScatterElements, GridSample, RoiAlign, NonZero, bicubic resize; LSTM > 16 steps
(compile runs out of resources at 24) and GRU (compiler crash) -> host or replace.

## C. No native op: implemented with PWL activations (`zoo/ops/pwl_inject.py`)
Hailo-8 APU = <= 9 linear pieces per channel with **non-negative slopes** (SAGE mantissas unsigned). Any unary f:
K-piece least-squares fit -> Jordan split f = sum G - sum H, each layer a bounded window of <= 7 pieces.
Sin, Cos, Tan, Atan, Asin, Acos, Sinh, Cosh, Asinh, Acosh, Atanh, Erf, Sign, Round, Floor, Ceil, Selu, Celu,
ThresholdedRelu, Shrink, Mod (periodic), TopK (host), ArgMin (= ArgMax(-x)).
- Sin over +-7.5 (2.4 periods), 9 pieces, G/H/sub in a16_w16: **29.2 dB** on bit-exact HSim, compiles.
- 8-bit G/H: 20-29 dB (cancellation); keep G/H + the add/sub tree in a16_w16.
- Open: precision-mode `a8_w8_a16` on PWL layers fails compile ("precision mode is not accurate").

## D. Emulator / tooling gaps
| Item | Effect | Fix |
|---|---|---|
| Resize bilinear (any mode) | compiles, HSim: "Not all layers support bit exact" | verify on chip; or rewrite as depthwise ConvTranspose 4x4 s2 (exactly emulatable) |
| Final softmax on rank-2 (classifier head, `HailoSoftmax`) | no bit-exact emulation -> zoo coverage capture fails (efficientnet, mobilenet, regnet, inception) | cut before softmax (host-side anyway) |
| Whisper zoo configs | no ONNX path (HAR-only) | fetch Hailo's HARs |
| DFT / STFT / GridSample / RoiAlign | unsupported | STFT = conv with fixed cos/sin kernels (needs the Split fix); GridSample/RoiAlign -> host |
| Our emulator (graph26/qatfast) | `postprocess` (host NMS) not emulated | host-side, out of scope |

## E. Low-accuracy natives worth a better fit
Exp 23.6 dB, Reciprocal / 1/x 25.9 dB, ew Mul 26.0 dB, HardSwish 30.7 dB, Mish 32.2 dB, SiLU 33.0 dB, Attention
18.0 dB, ArgMax 21.4 dB. Options: our own breakpoints (pwl_inject on the calibrated input), a16 outputs, or
decompositions (exp(x) = exp(x/2)^2).

## Open after batch 2 (ONNX_COVERAGE.md: 118/166 relevant ops run, 85 on chip at >= 25 dB, 43 untested)
- Mod: compat form x - m floor(x/m) fails parse ("ew mult ... expects 2 inputs but found 1"); Equal: needs the Round
  step from pwl_inject applied before parse (parser rejects Round).
- SpaceToDepth (ONNX op): rewrite to torch's PixelUnshuffle Reshape/Transpose (native 39 dB).
- Einsum general decomposer; DFT/STFT as fixed cos/sin conv.
- 43 untested ops (Sum, Mean, Swish, RMSNormalization, Attention op23, RotaryEmbedding, GatherElements/ND,
  ScatterND, OneHot, DeformConv, Q/DQ + QLinear* folding, ...): probe them.
- Hailo's built-in SiLU caps ~39 dB even in 16 bit: default to our PWL fit (pwl_inject SiLU, PWL_SILU=1).

## Batch 3 (compat3.py + cases3.py, 60 cases, all compile at 8 bit and 16-bit max precision) -> ONNX_COVERAGE.md
All 200 default-domain ops covered: 137 on the chip, 18 folded, 45 host (host_ops_check.py: onnx conformance tests
through onnxruntime / onnx reference; StagedRunner falls back to the reference evaluator for ops onnxruntime lacks).
Chip quirks found (rules in compat3.py):
- `greater` activation = thresholded ReLU: threshold must be >= 0 (negative: quantize error alone, garbage inside a
  graph) -> Greater(x, t<0) = Not(Less(x, t)). Less takes any t.
- compare activations are 8-bit only: a 0/1 mask is limited by its input's 8-bit grid (And window (-1,1): 13.2 dB =
  the 8-bit-input oracle 12.8). CMP_ZOOM=K adds Clip(K (x - t)) before the compare for 16-bit producers (Equal 37.2).
- constants on attention scores parse only tiled over heads (1, h, N, N); Hailo matmul rejects a signed left operand
  without transpose (LinearAttention: Relu split with -Q); MatMul of two tensors needs M, K, N divisible by the head
  axis, padded / per-head layouts do not parse -> broadcast Mul + ReduceSum per column (17.3 dB).
- a 1-input-channel conv is classed depthwise and rejected ("Depthwise doesn't support group param"): feed 2 channels.
- the parser merges identical conv columns (cos symmetric) -> concat input count mismatch: merge Add(MatMul, MatMul).
- rank-3 Reshape chains confuse the channel mapping: Reshape(Reshape) merged, rank-3 Conv1d parses natively.
- probe.py rebuilds any case named like one of its own: rewritten probe cases live under R_<name>.
- 16-bit MILP: Concat lists a8_w8_a16 but rejects it; ReduceSum 16x4 compile failure (PREC_EXCLUDE=HailoReduceSum);
  relu activation a16_w16_a8 unsupported on SAGE (PREC_EXCLUDE_LAYERS); matmul/softmax/ew_mult/ArgMax 8-bit only.
Still limited (on chip, < 25 dB at 16 bit): Attention 15.4 (8-bit matmul/softmax; QAT), ArgMax/ArgMin 21, OneHot
15.4, MaxUnpool 20.6 (window ties), Einsum tensor x tensor 17.3. Host: TopK, GridSample, RoiAlign, MaxRoiPool, GRU,
LSTM > 16 steps, NonZero, NMS, Det, Unique, AffineGrid (dynamic), Loop, Scan, dynamic Gather/Scatter indices,
DeformConv with learned offsets, LinearAttention gated/delta, and the non-tensor ops (strings, sequences, random...).
