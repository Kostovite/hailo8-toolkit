# Hailo-8 bug reports (ready to file)

Environment for every report:
- Hailo-8 PCIe x2 (both chips reproduce; results are deterministic across runs and reboots).
- Dataflow Compiler 3.33.1 (`hailo_sdk_client`, `hailo_model_optimization`).
- HailoRT 4.23.0 and the firmware shipped with it.
- Linux 6.1, x86-64.

Reproduction scripts referenced below are in `tools/bugrepro/`. The full catalogue, including limitations that are not bugs,
is in `docs/HAILO8_ISSUES.md`.

Ordered by impact. "Emulator" means `InferenceContext.SDK_BIT_EXACT` (HSim gives the same results).

"Likely layer" is our inference from black-box tests. No item is a proven silicon defect: the chip is deterministic
everywhere we looked, and for every chip-wrong case an equivalent rewrite of the same maths runs exactly.

---

## 1. Dynamic matmul: the second operand is read as signed int8 on silicon, as uint8 in the emulator (silent wrong)

**Summary.** In a matmul with `dynamic_weights=True` (attention Q·Kᵀ or P·V), the chip interprets the second
operand's 8-bit values as signed. The quantizer may give that operand an UNSIGNED encoding, for example a non-negative
tensor after ReLU: zp 0, range 0..255. Then every value above 127 is wrapped to a negative number on the chip. The
emulator uses the unsigned value, so all DFC accuracy checks pass while the HEF is wrong.

**Repro.**
1. Build `x -> conv3x3 -> relu -> {conv1x1 -> relu = Q, conv1x1 -> relu = K} -> Q·Kᵀ -> conv`: 10x10 tokens, C=32.
2. Quantize at level 0, compile with defaults.
3. Run the same raw frames on the chip and in SDK_BIT_EXACT (`chip_vs_emu.py`, `judge.py`).

**Observed.**
- Q and K are bit-exact chip == emulator, but the matmul output differs on 9.3% of elements, always lower on the chip
  (−1..−128 LSB).
- The wrong key tokens are exactly those with a quantized K value > 127.
- `Q · wrap_int8(K)` predicts the chip output to 0.17 LSB mean.
- SNR vs float: chip 10-14 dB, emulator 25-32 dB.

**Expected.** Either the quantizer encodes the weights operand as int8, or the hardware program reads it as uint8.
The emulator should match the silicon in either case.

**Workaround.** `quantization_param(<K producer>, force_range_out=[0, 4*max])`: chip == emulator bit-exact (the forced scale is range/255, so 2*max still wraps the tail: 4.33% wrong).
`dynmm_check.py Q.har` lists the affected operands.

**Exact mechanism** (bit-exact on 10,000/10,000 outputs, `tools/bugrepro/a1_exact/`). The HAR marks the K
producer's 8-bit output SIGNED (forced because it feeds the weights input), so the emulator saturates K at 127. The
silicon runs that producer's APU output stage (clip2 and output) UNSIGNED, 0..255, and the matmul reads the stored
bytes as int8: K values 128..255, which occur whenever inputs exceed the calibration range, wrap negative. Both parts
are needed to reproduce the chip; either alone leaves ~1,000 mismatches.

**Likely layer.** DFC: the quantizer's encoding (signed) and the generated hardware program (unsigned producer,
int8 reader) disagree.

## 2. GroupNorm(1) / whole-tensor normalisation is wrong on silicon (silent wrong)

**Summary.** `nn.GroupNorm(1, C)` parses to reduce_mean [H,W,1] -> [1,1,1] -> resize -> sub -> square ->
conv_var_inv -> inv_sqrt.
- On the chip, ~98% of output bytes differ from the emulator (correlation 0.66); SNR vs float is 1-3 dB on the chip
  and 17-18 dB in the emulator.
- When the HEF has two contexts (`resources_param` forced low), consecutive frames contaminate each other: the output
  of a frame depends on the previous frame.
- GroupNorm(4) at the same size is correct.
- It is wrong at every shape we tried: C 8-128 x H=W 8-40, 35 shapes, 37-52% of outputs differ by up to 131 LSB.

**Stock vs our compiler** (3 shapes, chip SNR vs float; emulator 36.1-36.7 dB):

| | stock | ours, 1 context | ours, 2 contexts | rewrite (stock and ours) |
|---|---|---|---|---|
| GroupNorm(1) | 3.1-3.6 dB | 3.3-3.5 dB | 44-49% of outputs wrong | 35.5-36.7 dB, == emulator |

**Bisection.** The centring half (channel mean -> spatial mean -> subtract) built from plain ops is exact on the chip.
No `pre_quantization_optimization(layer_norm_decomposition, ...)` option fixes the rest: equalization, nudging,
optimize_ew_mult or square_12_bit off leave 49% wrong; `bit_decomposition_mode=split_precision` 31%; a8_w8 does not
compile; `mode=ppu` is unsupported. The defect is in the variance path of the DFC's whole-tensor decomposition
(spatial REDUCE_MEAN of a 1-channel map, RESIZE broadcast of the scalar); the exact rewrite uses AVGPOOL instead.

**Likely layer.** That REDUCE_MEAN / RESIZE pair as programmed by the compiler (or the silicon unit behind it).
Deterministic, identical with both compilers. Not separable further without another DFC / firmware version.

**Repro.** torch `Conv -> GroupNorm(1, 64) -> Conv` at 20x20, exported to ONNX opset 17, default flow.

**Workaround.** Compute the spatial mean first, then the channel mean (`onnx_hailo_compat.py`, rule `gn1`): chip ==
emulator.

## 3. A dilated conv over a 9- or 10-wide row is wrong on silicon and depends on earlier frames (silent wrong)

**Summary.** `Conv2d(24, 24, 3, padding=3, dilation=3)` at 30x50: the compiler splits W into slices of 16, 16, 8 and 10.
On the chip, every output in the 10-wide slice (columns 40-49) is wrong (99%); all other columns are bit-exact.
The wrong values depend on what the chip processed before: batch output != frame-by-frame output (10% of
elements), and the first run after loading differs from later runs.

| W | compiled pieces | chip vs emulator |
|---|---|---|
| 7, 16 | undivided | exact |
| 9, 10 | undivided | 99% wrong |
| 11-15, 20, 22, 24, 40, 48, 52, 60, 64 | slices of 3-8 (and 16) | exact |
| 18, 26, 34, 50 | include a 10-wide slice | that slice 99% wrong |

SNR vs float at W=50: chip 4.8 dB, emulator 37.5 dB. Changing the slice's `layer_output_buffers_format`, the
defuse count (2, 5, 6) or the conv's buffer format does not help.

**Workaround.** Zero-pad W to a multiple of 8, conv, crop (same function: the conv's own padding is zeros):
chip == emulator, 38.5 dB. `dil_check.py M.hef Q.har` flags the bad pieces after a compile.

**Likely layer.** Compiler / hardware program: the right-edge padding of a 9-10 wide dilated row is never
written, so the reads see leftover buffer contents. Deterministic per history, not random.

## 4. The softmax x V matmul's INPUT_FEATURES split is silently wrong on silicon (silent wrong)

**Summary.** For single-head attention `softmax(Q K^T * t) V` (N = 64 or 784 tokens, C = 32), stock compiles P.V as
`defuse(matmul2, 2, defuse_type=INPUT_FEATURES)`. The chip result is 1-6 dB vs float; the emulator is 37-43 dB.
Truncated models show Q.K^T and the softmax bit-exact on the chip; only P.V is wrong (deterministic, not a
permutation; no correlation with any operand).

**Workaround.** `compilation_param(matmul2, resources_allocation_strategy=manual_scs_selection,
number_of_subclusters=2)` (or 8) keeps the matmul whole: chip == emulator, 40 dB. The same split is the
`map::at` replay crash at > 400 tokens (report on the allocator below).

**Likely layer.** Compiler: the INPUT_FEATURES defuse of a dynamic matmul.

## 5. Quantizer: the softmax denominator is sized from calibration and wraps (silent wrong off-calibration)

**Summary.** `HailoReduceSum.create_hw_params` sizes `reduce_sum_softmax`'s 16-bit accumulator from the calibration
maximum of -sum(exp) and allows wraparound. Rows whose exp-sum exceeds the calibrated maximum wrap and get a
reciprocal of ~0. At half the calibration amplitude a 784-token probe drops from 17.8 dB to -15.3 dB.
`force_range_preact` on the layer does not change the accumulator (the shift stays the same).

**Fix used** (`tools/fixes/smx_headroom.py`): during quantization, give the accumulator the worst-case range
N x max(exp). Chip results (amplitude 0.25 / 0.5 / 1.0):

| probe | calibrated / worst range | default | worst-case headroom |
|---|---|---|---|
| t = 0.05 | 705 / 784 | 37.9 / 40.7 / 42.6 dB | 42.8 / 42.9 / 42.5 dB |
| t = 0.2 | 518 / 784 | 10.0 / 18.0 / 37.8 | 34.1 / 37.9 / 36.7 |
| t = 1.0 | 130 / 784 | -13.2 / -15.3 / 17.8 | 4.3 / 9.7 / 12.8 |

The cost when the calibrated range is far below the worst case (t = 1.0: -5 dB in distribution) comes from the
8-segment reciprocal table being fitted over a wider range, so the headroom is chosen per model on validation data.

**Expected.** A saturating accumulator, or a range chosen for the reduce size rather than the calibration set.

## 6. Quantizer: the softmax denominator accumulator wraps at 2^15 (silent wrong, chip and emulator agree)

**Summary.** The decomposed softmax's `reduce_sum_softmax` (kernel −2, bias 1 × factor 2, shift 1) feeds the
`inv_pos` APU from a 16-bit WRAPPING accumulator: APU input = −S + 1 mod 2^16, S = sum of the 8-bit exp outputs of
a row. For S > 32,770 it wraps positive.

**Observed** (451,584 rows, chip; the emulator agrees exactly except report 10):

| row sum S | correct reciprocal (LSB) | chip / emulator |
|---|---|---|
| 16,384-32,768 | 2-4 | 2-4 (within 1) |
| 32,771-40,000 | 2 | 0-2 |
| 40,000-65,535 | 1-2 | 0 |
| 65,536-70,000 | 1 | 0-20 |

A production YOLO26m (C2PSA attention, 784 tokens) has 2.6% of its attention rows above 32,770 on real pages. Those
rows get softmax weights of ~0, and no DFC accuracy report shows it because the emulator wraps identically.

**Expected.** A saturating or wider accumulator, or an exp output scale chosen so that 784 × max(exp) fits.

**Likely layer.** DFC quantizer.

## 7. Quantizer: dynamic-matmul second operand with a non-zero zero point is never compensated (silent wrong)

**Summary.** When the second operand of a `dynamic_weights` matmul is unsigned with zp ≠ 0 (K or V with a bias),
the HAR's matmul has `zp_correction_type: zp_comp_none` and `zp_x_vals [0 0]`. The zero point is ignored, on the chip
and in the emulator (≈3 dB lost on the probe). `quantization_param(<producer>, signed_output=True)` and a symmetric
`force_range_out` do not change the encoding.

**Measured in softmax attention** (chip == emulator in all cases): K bias offset 3.0: 33.9 dB; with the exact rewrite
(K bias dropped, it is constant along the softmax axis; V bias moved after P.V, rows of P sum to 1;
`tools/fixes/attn_bias_rewrite.py`) 42.4 dB. V offset 42.1 -> 44.4 dB, both 41.8 -> 44.4 dB.

**Likely layer.** DFC quantizer.

## 8. Compiler accepts a context starting at a bilinear resize and emits a wrong, batch-dependent HEF (silent wrong)

**Summary.** A random 4-block network (`fz_1043`) partitioned so that context 2 starts at a bilinear resize compiles
without warnings. On the chip 878,131 of 884,736 output elements differ from the stock HEF of the same HAR, and the
output differs between batch 1 and batch 8. Moving the boundary one layer (off the resize) gives a HEF that is exact
at batch 1 and 8.

**Expected.** A compile error, as for other illegal boundaries.

**Likely layer.** DFC compiler / allocator.

## 9. ConvTranspose with kernel = stride = 2: the DECONV lowering is wrong on silicon (silent wrong)

**Summary.** `ConvTranspose(kernel_shape=[2,2], strides=[2,2])`, the standard non-overlapping 2x upsampler, gives
−17 dB vs float on the chip, while the emulator gives +32 dB for the same HEF and frames. Kernel 4 / stride 2 and
kernel 3 / stride 2 are bit-exact chip == emulator.

**Repro.** `x(16,24,24) -> ConvTranspose(16,16,k2,s2) -> conv1x1`, level 0, default compile.

**Workaround.** A 1x1 conv to 4·C channels followed by `DepthToSpace(blocksize=2, mode=CRD)` (`deconv_k2s2_rewrite.py`) computes the same
function (onnxruntime max diff 0) and is correct on the chip (35.7 dB, == emulator).

**Root cause.** For kernel == stride the compiler lowers the layer to `defuse(deconv, 1, defuse_type=DECONV)`: a
conv to 4C channels plus FEATURE_INTERLEAVE layers. That HEF is 99-100% wrong (-11 dB vs float, stock and our
compiler alike, 3 shapes). Forcing `defuse(deconv1, 1, defuse_type=SUPER_DECONV)` (the lowering stock picks for
k3s2) on the SAME quantized HAR is bit-exact on the chip. The chip output is not a permutation of the right one
(correlation 0.06).

**Likely layer.** Compiler: the DECONV / FEATURE_INTERLEAVE lowering. Workaround in a model script:
`deconv1_defuse_conv, deconv1_defuse_d2s = defuse(deconv1, 1, defuse_type=SUPER_DECONV)`.

## 10. Emulator: softmax reciprocal (`reduce_sum` + `inv_pos`) has a discontinuity at an APU input of −32,768 (emulator wrong, silicon right)

**Summary.** In a decomposed softmax, `reduce_sum_softmax` computes about −Σexp + 1 (kernel −2, bias 1 × factor 2,
shift 1) and applies the `inv_pos` PWL. Its x_points start at −32,767.
- When the row sum is 32,769 or 32,770 (APU input −32,768 / −32,769), the emulator returns 0.
- The chip returns 2, which is what both return for the neighbouring sums 32,765 and 32,774-32,775.
- All 1,329 rows with sums ≥ 32,771 agree.

**Found on.** A production YOLO26m (C2PSA attention, 784 tokens, 4 heads): 8 of 916 validation pages differ chip
vs emulator on every output head. Bisected with sub-HEFs of the attention block fed the exact intermediate
integers (`build.py`, `run_att.py`, `one.py`):
- matmul1 and the exp step are bit-exact on all pages.
- `reduce_sum_softmax1` differs in exactly one element per bad page, and none on good pages.
- Every row with sum 32,769/32,770 is wrong (8/8).

**Mechanism** (451,584-row chip map). S = 32,769 gives APU input −32,768, below the first x_point; S = 32,770 wraps to
+32,767, equal to the last x_point. The silicon evaluates the first / last PWL segment there. The emulator zeroes
every input outside [x_0, x_last). An APU model with edge-segment extension matches every chip row.

**Expected.** The emulator should extend the edge segments, as the silicon does.

## 11. Emulator: raw UINT16 network inputs are clamped at 32,767 (emulator wrong, silicon right)

**Summary.** For a UINT16 input layer (zp 16,006), SDK_BIT_EXACT behaves as if raw input values were clamped to
32,767. The chip uses the full unsigned range.

| raw input | chip output | emulator output |
|---|---|---|
| 30,000 / 32,000 / 32,767 | equal | equal |
| 33,000 | 30,026 | 29,828 |
| 40,000 | 32,767 (saturated) | 29,828 |

The emulator's output is 29,828 for every raw value ≥ 32,767.

**Repro.** `u16_boundary.py` (single element per raw value) and `ampsweep.py`: `Sum(x, 2x, x)` in a16_w16,
one input element set to each raw value.

**Impact.** Emulator-based validation of 16-bit models disagrees with silicon whenever inputs exceed the calibration
range.

## 12. Emulator vs silicon 1-2 LSB differences on 16-bit layers (minor)

16-bit layers (Sum, Mean, QLinearConv folded to conv) differ by 1-2 LSB on 0.15-0.23% of outputs at every input
amplitude. Several 8-bit gather/crop/DFT probes and InstanceNorm differ by 1 LSB on <0.1%. Deterministic. The
"bit-exact" emulation is therefore not bit-exact for these layers.

## 13. Parser: several ONNX ops silently compute a different function

All of these parse without error, compile, and run on the chip, but compute a different function:

| ONNX | What the DFC builds | SNR |
|---|---|---|
| `Pad` mode reflect / edge | zero padding | 8.3 dB |
| `ConvTranspose` k3 s2 p1 `output_padding=1` | wrong output geometry | −3 dB |
| `Pow(x, -0.5)` | a `pow` activation quantized wrongly | −10.7 dB |
| `Floor` | node dropped | |
| `Mul(x, x)` | Mul skipped | |
| any unsupported op | dropped; an `IndexError` appears later, in the next layer's shape calculation | |
| `Greater(x, t)` with t < 0 | thresholded ReLU semantics, garbage inside a graph | |

**Expected.** A parse error for anything the DFC cannot represent.

## 14. Compiler: produces an allocator solution it cannot replay (`map::at`)

**Summary.**
1. A single-context compile of an attention block at > 400 tokens may INPUT_FEATURES-defuse the softmax x V matmul
   (`matmulN_fs_0, ... = defuse(matmulN, 2, defuse_type=INPUT_FEATURES)`).
2. Loading that compile's own `auto.alls` back onto the same sub-network HAR fails with `_Map_base::at`.
3. Merely giving the full model a `context([...])` partition makes the compiler pick that split and crash the same way.

**Workaround.** `compilation_param(matmulN, resources_allocation_strategy=manual_scs_selection,
number_of_subclusters=8)`.

## 15. Compiler: core assert and internal errors at context boundaries

Each of these is a crash on a legal multi-context partition:

| Context boundary | Error |
|---|---|
| a context ends at a `feature_multiplier` (x²) | core assert `FeatureMultiplierKernel::set_default_power_table: m_out_amdc.size() ==` |
| a context starts at a `format_conversion` (attention fed from a context input) | `_M_range_check`, or "Memory units capacity exceeded (available: 128, required: 165)" |
| both outputs of a `feature_splitter` cross to the next context | `ValidationError: [...] has non-unique elements` (HN schema) |
| a decomposed LayerNorm is split | `UnsupportedModelError: Shapes at dim #2 aren't equal at ew mult layer ..._mult_reduce_sd0` |
| an attention-internal (1, N, C) tensor crosses | `can't spatial reshape from (1, 400, 800) to (50, 8, 800) without a successor format_conversion` |

Partition rules that avoid them: `docs/COMPILER_RULES.md` sections 18-21.

## 16. Compiler: a second `compilation_param` line for the same layer resets the first one's parameters

`compilation_param(L, resources_allocation_strategy=manual_scs_selection, number_of_subclusters=2)` followed by
`compilation_param(L, mixed_mem=disabled)` drops the SC pin, and the compile then fails ("input buffers should be
divisible by 2"). Expected: parameters merge, or the second line is rejected.

## 17. Compiler: parallel compiles in the same working directory corrupt each other

Two `ClientRunner.compile()` processes in one cwd overwrite each other's `errors.pb` and rotating SDK logs. The
result is spurious `DecodeError: Unexpected end-group tag`, empty "Compilation failed:" messages and
`rotating_file_sink` errors. Workaround: a private cwd per process.

## 18. Compiler: partition search is single-threaded

YOLO26m takes 96 min at default level. 5,105 s of it is the Slotter hill-climb, with ONE busy thread out of 220.
`allocator_param(compilation_num_threads=48, num_of_workers=32)` changes nothing. An external per-context
partitioner (`turbo/hcc.py`) builds a bit-exact, faster HEF in 9.5 min.

## 19. Profiler rejects a graph the compiler accepted

After a successful single-context compile of a Llama decoder layer (RoPE: one conv with two outputs feeding
rotate_half and the cos multiply), `runner.profile()` raises `UnsupportedModelError: Shapes at dim #2 aren't equal at
ew mult layer ew_mult3, shapes=[[-1, 8, 16, 192], [-1, 8, 8, 192]]`. The compiled HN has [8,8,192] on both inputs.

## 20. HSim: stream-0 launch race; sm_61-only kernels

- HSim.so launches its CUDA kernels on stream 0 while TensorFlow in the same process uses a non-blocking stream. The
  outputs have random garbage in their tail; this reproduces on any GPU.
- The sm_52 image in HSim.so is an empty stub; only sm_61 cubins and PTX are present.

Fix used: an `LD_PRELOAD` shim that fences HSim's launches and swaps in the PTX retargeted to sm_52 (`hsim_sm52/`).

## 21. HailoRT: `FormatType.AUTO` rejects uint16 arrays for UINT16 inputs

`InputVStreamParams.make(network_group)` with the default `FormatType.AUTO`, then `InferVStreams.infer({name:
uint16_array})`, raises `HailoRTException: ... numpy array item size is 2, not 1`. Passing
`format_type=FormatType.UINT16` works.
