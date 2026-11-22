# Hailo-8 / DFC 3.33.1 / HailoRT 4.23: every defect, mismatch and limitation we measured

Scope: two local Hailo-8 (PCIe, `/dev/hailo0`, `/dev/hailo1`), Dataflow Compiler 3.33.1, HailoRT 4.23.0,
firmware shipped with HailoRT 4.23. Everything below was **measured on the real chips** or in the DFC itself.
Nothing is inferred from documentation. Reproduction scripts are in `tools/bugrepro/`. The compiler rulebook these
findings come from is `docs/COMPILER_RULES.md`, and the operator tables are in `tools/ops/`.

The items we believe are genuine Hailo bugs, written up for reporting, are in `docs/HAILO_BUG_REPORTS.md`.

Severity:
- **S1 silent wrong:** compiles and runs, numbers are wrong, and nothing warns.
- **S2 emulator wrong:** silicon is right, but Hailo's bit-exact emulator (and anything validated against it) is wrong.
- **S3 crash/refusal:** a hard error, so the user at least knows.
- **S4 performance/tooling.**

## 0. What was scanned (coverage) and what was not

| Scan | Size | Method | Result |
|---|---|---|---|
| ONNX operators (default domain) | 200 ops, 453 compiled probes | `x -> op -> 1x1 conv`, HSim bit-exact vs onnxruntime | 137 on chip, 18 folded, 45 host-only; parser and quantizer defects in sections 3-4 |
| Chip vs SDK_BIT_EXACT emulator | 438 probe HEFs x 8 frames | `chip_vs_emu.py`: same raw frames on the chip and in the emulator | 399 bit-exact, 11 at 1 LSB, 28 different. Every different case was re-judged with in-distribution inputs |
| Layer types at context boundaries | all 31 HN layer types | 2 convs -> OP -> 2 convs, cut before and after, multi-context | bit-exact except the cases in section 5 |
| Random multi-context networks | 120 nets, 424 forced partitions | `probe/fuzz.py`: 32 block types, chains of 4-7 | 423/424 compile bit-exact with the rules in `COMPILER_RULES.md` section 21 |
| Production model (YOLO26m, 896x896) | 916 val pages, 6 heads | chip vs exact executor; both chips; 3 days apart across a reboot | deterministic (all 916 pages bit-identical across runs and chips); 908/916 bit-exact vs the emulator; the 8 others are an EMULATOR bug (A3) |
| 16-bit input range | 3 ops x 8 amplitudes + single-element boundary test | `ampsweep.py`, raw values 30,000-65,535 | A5, A6 |
| Firmware | APP + CORE images | Ghidra decompile; `measure-fw-actions` timelines | no functional bug found; timing behaviour in section 7 |
| Every torch op case at 6 shapes | 90 cases x C/H/W = 8/8/8, 32/40/40, 64/16/16, 128/32/32, 256/12/12, 24/30/50 | `probe/scan/shapes.sh`: compile, chip vs emulator on both chips | 55-58 chip-run per shape (tables: `docs/SCAN_TABLES.md`). New: dilated conv (A9) at 24x30x50. Otherwise only the known A2/A5 items. Two compiles never finished (D17) |
| GroupNorm(1) shape grid | C 8-128 x H=W 8-40, 35 shapes | `probe/scan/gn_grid.sh` | wrong at EVERY shape (A2) |
| Frame / network contamination | 553 HEFs (probes + fuzz partitions) | `contam.py`: repeated vs varied frames, different orders, neighbours, batch 1/8/16 | 552/553 identical; the one exception is an invalid partition the compiler accepted (D16) |
| HailoRT scheduler | 8 random triples of HEFs | ROUND_ROBIN, 3 network groups inferred concurrently from threads, 6 runs each | all bit-identical to solo runs |
| Multi-process sharing | the same 8 triples | `hailort_service`, `multi_process_service=True`, 3 PROCESSES on one chip at once, 6 runs each | 24/24 HEFs bit-identical to solo runs |
| 4-bit weights | 13 probes | `quantization_param(..., precision_mode=a8_w4)` | 13/13 chip == emulator, no FPS change. `model_optimization_config(auto_4bit_weights_ratio=...)` is silently ignored |
| Dilated conv width sweep | dilation 3, C24 H30, W 7-64 (22 widths) + 6 compile variants + padding fix | chip vs emulator, per-column error map, repeat/batch runs | A9 |
| Stock vs our compiler, GN1 and ConvT k2s2 | 3 shapes each + rewrites, stock / ours 1 ctx / ours 2 ctx (34 HEFs) | chip vs emulator, SNR vs float | both compilers equally wrong; rewrites exact with both (A2, A4) |
| Attention probes (single head, N 64-784, temperatures 0.05-1, asymmetric K/V) | 20 HEFs | chip vs emulator, amplitude sweep 0.25-4 | D18, softmax headroom, zero-point rewrite |
| Softmax reciprocal map | 451,584 rows, row sums 254-106,976 | YOLO26m attention sub-HEF, crafted inputs | the APU extends its first/last segment past the table (A3); denominator accumulator wraps (C) |

**Production check after the emulator fixes:** YOLO26m validation (916 pages) with the fixed exact executor: mAP50
0.886052 (chip 0.88605; before the APU-edge fix 0.886034). A saturating softmax denominator (the overflow fix applied
to the already QAT-trained model) LOWERS it to 0.8841 / mAP50-95 0.8162 (from 0.8176): QAT learned the wrapping
arithmetic, so the denominator headroom must be set before QAT, not patched afterwards.

**Not scanned:**
- Power modes and clock settings (we never touch I2C or power).
- Ethernet and M.2 variants.
- Other firmware / DFC versions (needed to tell silicon from firmware from compiler code generation for A2 and A4).

**Which layer is at fault.** All tests are black-box, so the layer is inferred, not proven. A3, A5, A6 are SDK
emulator software (silicon is right). A1, C and D are DFC software (quantizer, code generator, allocator). A2 and A4
are deterministic and the same maths written differently runs exactly, which points at the compiler's hardware
program rather than broken arithmetic; the frame contamination in A2 involves context switching, which firmware
drives. No item is a proven silicon defect.

## A. Silicon vs Hailo's own emulator

| # | Sev | Finding | Evidence | Workaround |
|---|---|---|---|---|
| A1 | S1 | **Dynamic matmul reads its 2nd operand as signed int8 on the chip; the emulator reads it as uint8.** When that operand (K in Q·Kᵀ, V in P·V) is non-negative, e.g. post-ReLU, the quantizer makes it unsigned 0..255. Every value > 127 then wraps negative on silicon. | Probe: 2 convs -> ReLU -> Q·Kᵀ. Both operands are bit-exact chip == emulator, but the matmul output differs on 9.3% of elements. It is always lower on the chip, by 1..128 LSB. The wrong keys are exactly those with K > 127, and `Q·wrap_int8(K)` predicts the chip to 0.17 LSB. SNR vs float: chip 10-14 dB, emulator 25-32 dB. | `quantization_param(<K producer>, force_range_out=[0, 4*max])`: chip == emulator bit-exact (forced scale = range/255, so 2*max still wraps the tail: 4.33% wrong). Detector: `tools/bugrepro/dynmm_check.py Q.har`. **Exact mechanism (bit-exact, 10,000/10,000, `tools/bugrepro/a1_exact/`):** the HAR marks the K producer SIGNED (forced by `get_signed_output` on the weights edge) and the emulator saturates it at 127; the chip runs that producer's APU output stage (clip2 + output) UNSIGNED, 0..255, when its function is non-negative, and the matmul reads the bytes as int8. Modelled in `hailoqat/graph26.py` (`silicon_weights_operands`) + `intops` (`x1_bits`); `HAILOQAT_SDK_MATMUL=1` restores SDK behaviour. |
| A2 | S1 | **GroupNorm(1)** (stock AND our compiler, 1 or 2 contexts: chip 3.1-3.6 dB vs emulator 36.1-36.7 at C16/64/128) (whole-tensor mean/variance: reduce_mean [H,W,1] -> [1,1,1] -> resize -> sub -> square -> inv_sqrt) is wrong on the chip at every shape tested. | Grid C 8-128 x H=W 8-40 (35 shapes): 37-52% of probe outputs differ, up to 131 LSB. C=64 20x20: ~98% of output bytes differ from SDK_BIT_EXACT (corr 0.66). SNR vs float: chip 1-3 dB, emulator 17-18 dB. In 2 contexts, consecutive frames contaminate each other: a repeated frame gives a different output from varied input. GroupNorm(4) is fine. | Statistics spatial-first, then the channel mean (`onnx_hailo_compat.py` rule `gn1`): chip == emulator, 35.5-36.7 dB, with both compilers. Bisection: the centring half (channel mean -> spatial mean -> subtract) is exact on the chip; no layer_norm_decomposition knob fixes the rest (equalization / nudging / ew_mult / square_12_bit off: unchanged 49%; split_precision: 31%; a8: no compile). The defect is in the variance path the DFC's own decomposition uses (spatial REDUCE_MEAN of a 1-channel map + RESIZE broadcast); the rewrite uses AVGPOOL instead. |
| A3 | S2 | **The emulator's softmax reciprocal (`reduce_sum` + `inv_pos` APU) is discontinuous at 2^15.** A softmax row whose exp-sum lands on 32,769 or 32,770 (APU input −32,768 / −32,769) gets 0 from SDK_BIT_EXACT/HSim. The silicon gives 2, the same value as for the neighbouring sums 32,765 and 32,774. | Shipped YOLO26m: 8/916 val pages differed chip vs emulator on all 6 heads. Bisected with sub-HEFs of the attention block fed the exact intermediate integers. matmul1 and the exp step are exact on all pages. `reduce_sum_softmax1` differs in exactly ONE element per bad page, and none on good pages. Every row with sum 32,769/32,770 is wrong (8/8); all 1,329 rows with sums ≥ 32,771 agree. Mechanism, from a 451,584-row chip map: the APU input is −S+1 in a 16-bit wrapping accumulator. S=32,769 gives −32,768 (below the first x_point) and S=32,770 wraps to +32,767 (= the last x_point). The silicon evaluates the first / last segment there; the emulator zeroes everything outside [x_0, x_last). An APU model with edge-segment extension matches all 451,584 chip rows. | None needed on silicon. Fixed in this repo's executor (`hailoqat/intops.py` apu, `hq_ext/hq.cu`; `HAILOQAT_SDK_EDGES=1` restores the SDK rule). Accuracy-neutral here: mAP 0.88605 chip vs 0.88603 emulator. |
| A4 | S1 | **ConvTranspose with kernel = stride = 2** (the non-overlapping 2x upsampler) is wrong on the chip. | C=16, 24x24 -> 48x48: chip −17 dB vs float, emulator +32 dB, same HEF and frames. k4 s2 and k3 s2 deconvs are bit-exact. The op survey's "native 33.6 dB" was the emulator's number. | 1x1 conv to 4C channels + DepthToSpace(2, CRD): identical function (onnxruntime max diff 0), chip == emulator 35.7-38.6 dB. **Root cause:** for kernel == stride the compiler lowers the deconv to `DECONV` -> FEATURE_INTERLEAVE (99-100% wrong, stock and ours, -11 dB); forcing `defuse(deconv, 1, defuse_type=SUPER_DECONV)` (the k3s2 lowering) on the same HAR is bit-exact. Our compiler forces it (rule R23). |
| A5 | S2 | **16-bit path, small LSB differences.** 16-bit layers differ from SDK_BIT_EXACT by 1-2 LSB on 0.15-0.23% of outputs (Sum3, QLinearConv, Mean2) at every input amplitude, including tiny ones. Several 8-bit gather/crop/DFT probes and InstanceNorm differ by 1 LSB on <0.1%. Deterministic. | `ampsweep.py`; the 11 cases at ≤1 LSB in the 438-case scan. | None (numerically negligible, but "bit-exact" emulation is not bit-exact). Bisected on the chip (`tools/bugrepro/a5/`): our executor == SDK_BIT_EXACT on every layer; the 1-LSB flips (17-18 of 9,216) come from a 16-bit conv / 16-bit add stage, not from the APU. Excluded: APU first-shift rounding precision (6-40 bits), half-up rounding, accumulator truncation before the APU multiply (1-13 bits), byte-plane (8x8) MAC rounding, per-product truncation in the 16-bit add. The flips are not rounding ties (chip up to 1.17 LSB from the exact value). Open. |
| A6 | S2 | **The emulator clamps raw UINT16 network inputs at 32,767; the silicon uses the full 0..65,535 range.** The quantizer's own data stays below ~32,012 (zp 16,006), so only out-of-calibration inputs hit it. With them, 59-91% of outputs differ from the emulator. | Sum3, QLinearConv, Mean2 (16-bit). Amplitude sweep: ≤0.2% 1-LSB differences up to raw 32,390, then 74-81% from raw 40,581. One raw element at 30,000 / 32,000 / 32,767: chip == emulator. At 33,000 the chip gives 30,026 vs the emulator's 29,828 (frozen for every value ≥ 32,767); at 40,000 the chip saturates at 32,767. Only the element's 3x3 neighbourhood changes (no frame-wide effect). | Clip inputs to the calibrated range on the host. Do not validate 16-bit models with synthetic full-range inputs. |
| A7 | S2 | **Placement changes arithmetic for global reductions.** Flipping only `mixed_mem` (or the resource caps) on GroupNorm(1) changes outputs by ≤4 LSB, deterministically. | `COMPILER_RULES.md` §18. | Judge exactness against a fixed reference HEF, not across compile settings. |
| A9 | S1 | **A dilated conv computed over a 9- or 10-wide row reads stale memory on the chip.** Any compiled piece of a dilation-3 conv (the whole layer, or one W slice of a spatial defuse) with output width 9 or 10 is ~99% wrong, and the values depend on earlier frames and on what ran before (batch != 1-by-1; chip 0's first run differs from its later runs). Pieces 3-8 and 16 wide are exact. | Found by the shape sweep at C24 30x50: the compiler cut W into 16+16+8+10 and columns 40-49 are 99% wrong, all others exact; chip 4.8 dB vs float, emulator 37.5. Width sweep W 7-64: wrong exactly at W 9, 10 (undivided) and whenever a 10-wide slice appears (18, 26, 34, 50). Forcing another defuse count or buffer format does not help. | Zero-pad W to a safe width (multiple of 8, e.g. 50->56, 10->16), conv, crop: chip == emulator, 38.5 dB. Guard: `tools/bugrepro/dil_check.py M.hef Q.har`. |
| A8 | S1 (DFC) | **Attention_pattern probe**: chip and emulator are both far from float (−8..−13 dB) and differ from each other on 93% of elements. | 438-case scan + in-distribution judge. | Use the `nn.MultiheadAttention` token layout (32.7 dB). |

## B. Parser silent-wrong (DFC `translate_onnx_model`)

| Op | Effect | Fix |
|---|---|---|
| Pad mode reflect / edge | parsed as zero padding (8.3 dB) | Slice + Concat (exact) |
| ConvTranspose k3 s2 p1 output_padding=1 | wrong output geometry (−3 dB) | equivalent k4 s2 p1 deconv |
| Pow(x, −0.5) | `pow` activation quantized wrongly (−10.7 dB) | Reciprocal(Sqrt) -> `inv_sqrt` |
| Floor alone | node dropped | PWL staircase or host |
| Max(a, b) of two tensors | shared-scale concat + reduce_max (10.7 dB) | a + Relu(b − a) |
| Any unknown op | silently DROPPED; surfaces as `IndexError` in the next layer's shape calculation | rewrite before parsing |
| Attributes left at their ONNX default | not filled in (BatchNorm `epsilon` -> IndexError) | write defaults explicitly |
| `x*x` (Mul of a tensor with itself) | the Mul is skipped | `Pow(x, 2)` |
| Identical conv columns (e.g. symmetric cos/sin) | merged by the parser, so the concat input count no longer matches | merge as Add(MatMul, MatMul) |
| Weights shared via ONNX `Identity` (torch dedup) | `get_kernel` IndexError | give each Identity its own initializer |
| Token strip 1 x N layout | parses to a zero-height tensor | 2-D token grid (8x8 for N=64) |
| Greater(x, t < 0) | `greater` activation = thresholded ReLU: garbage inside a graph | Not(Less(x, t)) |
| channel -> width Reshape | mistranslated (−3 dB) | per-column depthwise conv + Concat |

## C. Quantizer

| Finding | Effect |
|---|---|
| Dynamic-matmul second operand may be quantized unsigned | triggers A1 on silicon. The emulator does not show it, so every DFC accuracy report misses it |
| **S1:** dynamic-matmul second operand unsigned with zp ≠ 0 (e.g. K with a bias, range not starting at 0). In softmax attention it costs accuracy rather than breaking: K offset 33.9 dB -> 42.4 with the exact rewrite (`tools/fixes/attn_bias_rewrite.py`, compat rule `attn_bias`: K bias dropped, V bias moved after P.V; perturbation-based, works on fused nn.MultiheadAttention in_proj); V offset 42.1 -> 44.4; both 41.8 -> 44.4 | the HAR records `zp_correction_type: zp_comp_none`, `zp_x_vals [0 0]`: the zero point is never compensated, so chip AND emulator are wrong (≈3 dB lost). `quantization_param(..., signed_output=True)` and symmetric `force_range` do not change it. Signedness comes from `get_signed_output()`: signed only for 8-bit outputs on a `LAYER_OUT_WEIGHTS` edge. Proposed rewrite (not yet chip-verified): drop K's bias before softmax (row-constant), move V's bias after P·V (rows of P sum to 1) |
| **S1:** the softmax denominator (`reduce_sum_softmax`) uses a 16-bit WRAPPING accumulator sized from the CALIBRATION maximum (`HailoReduceSum.create_hw_params`: `utilize_wraparound`) | row sums of the 8-bit exp outputs above 32,770 wrap. Sums 32,771-65,535 give a reciprocal of 0-1 instead of 1-2, so the attention row is nearly zeroed; sums ≥ 65,536 give arbitrary values (up to 254 LSB off). Shipped YOLO26m: 2.6% of C2PSA attention rows (1,329/50,176 on 16 val pages). Chip and emulator agree, so no DFC report shows it. Off-calibration inputs collapse attention: probe N=784, t=1.0 at half amplitude -15 dB. `force_range_preact` does NOT change the accumulator. Fix (`tools/fixes/smx_headroom.py`, a quantizer hook): size the accumulator for the worst case N x max(exp). Chip: t=0.2 at amplitudes 0.25/0.5/1.0: 10.0/18.0/37.8 -> 34.1/37.9/36.7 dB; t=0.05: 37.9/40.7/42.6 -> 42.8/42.9/42.5; t=1.0 (calibrated range 6x below worst): -13.2/-15.3/17.8 -> 4.3/9.7/12.8 (the 8-segment reciprocal table spread over a wider range costs 5 dB in distribution). Choose per model on validation data (modes none / x2 / full) |
| `SDK_QUANTIZED` (what AdaRound / QFT optimize) != chip | skips per-product rounding and the 16-bit accumulator wrap; score diffs up to 0.13 vs the chip on YOLOv8m |
| First export after editing kernels shifts biases by a per-layer constant on `bias_factor` 2/4 layers | write-back must be done twice to land exactly |
| `a8_w8_a16` on PWL activation layers | compile refusal ("precision mode is not accurate") |
| Concat advertises `a8_w8_a16` but rejects it; ReduceSum 16x4 fails; relu `a16_w16_a8` unsupported | 16-bit MILP has to exclude them |
| GlobalAvgPool on zero-mean features | 11 dB (51 dB on positive features) |
| Neg -> Relu -> Neg chains | 7.9 dB; use v − Relu(v) |

## D. Compiler / allocator

| # | Sev | Finding | Rule / fix (`COMPILER_RULES.md`) |
|---|---|---|---|
| D1 | S3 | The compiler produces an allocator solution it cannot read back: an INPUT_FEATURES split of the softmax x V matmul at > 400 tokens -> `map::at` when replayed, even on its own sub-network. Stock itself crashes when merely given `context([...])` of such a model. | R12: pin the matmul to 8 SC |
| D2 | S3 | `FeatureMultiplierKernel::set_default_power_table: m_out_amdc.size()` core assert when a context ends at x² | R13 |
| D3 | S3 | A context starting at a format_conversion (attention fed straight from a context input): `_M_range_check`, or "Memory units capacity exceeded (128 vs 165)" | R10 |
| D4 | S3 | `ValidationError: non-unique elements` when both outputs of a feature_splitter cross a context boundary | R20 |
| D5 | S3 | `Shapes at dim #2 aren't equal at ew mult` when a decomposed LayerNorm is split across contexts | R19 |
| D6 | S3 | `can't spatial reshape (1,400,800) to (50,8,800) without a successor format_conversion` when an attention-internal tensor crosses a boundary | R18 |
| D7 | S3 | "Splitter failed to find a possible solution" for many legal-looking partitions (norm-block entry, bilinear-resize entry, SE squeeze, conv_slice families) | R14, R16, R21, R22 |
| D8 | S3 | `microcode exceeded size for dw1`, `Latency simulation reached timeout`, `Resolver ... Watchdog expired after 1h` on small contexts | re-partition |
| D9 | S3 | A second `compilation_param(L, ...)` line on a layer resets the parameters of the first (an SC pin disappears -> "input buffers should be divisible by 2") | merge into one line |
| D10 | S3 | GRU crashes the allocator; LSTM > 16 steps refused | host / unroll |
| D11 | S4 | Blind spots, deterministic across seeds: dw5/dw7 never defused (1 SC, 25x slower), stride ≥ 4 inserts a 1-SC EXTERNAL_PAD bottleneck, c=72 gets 28 SC, C2f n=1 8x slower. Level max fixes some. | `COMPILER_RULES.md` §10, §12, §14 |
| D12 | S4 | The partition search runs on ONE thread (CP-SAT sequential, time-boxed); `compilation_num_threads` / `num_of_workers` change nothing; 96 min for YOLO26m | own partitioner: 9.5 min |
| D13 | S4 | Parallel compiles in one working directory clobber each other's `errors.pb` and rotating logs: fake `DecodeError`, empty "Compilation failed:" | private cwd per compile |
| D14 | S4 | mixed_mem: stock disables it per layer but allows it on its helper nodes. A blanket `mixed_mem=disabled` breaks allocation ("l4_portal ... not found in toposort recipe"). Disabling it before allocation trades batch-1 latency against batch-4 throughput. | `COMPILER_RULES.md` §20 |
| D15 | S3 | Row width w*c > 128 KiB does not compile; matmul "memory units" hard cap 128 per cluster | shape changes |
| D16 | S1 | **The compiler accepts a partition whose context starts at a bilinear resize and the HEF is wrong.** fz_1043 `cut_4_7`: 878,131 of 884,736 output elements differ from the stock HEF at batch 1, and the result changes with batch size (b1 != b8). The rebuild with the context moved off the resize is exact at b1 and b8. Stock never picks that cut | R16: never start a context at a resize |
| D18 | S1 | **The softmax x V matmul's INPUT_FEATURES defuse is silently wrong on the chip.** Single-head attention (N = 64 and 784 tokens, 4 probes): stock splits P.V with `defuse(matmul, 2, defuse_type=INPUT_FEATURES)` and the chip gives 1-6 dB vs the emulator's 37-43 (deterministic; Q.K and the softmax alone are exact). The same split at > 400 tokens is the D1 replay crash. | R12 now pins every softmax x V matmul (manual_scs_selection, 2 or 8 SC): chip == emulator, 40 dB. Stock users: add the same compilation_param line. |
| D17 | S4 | Allocator runs > 60 min ("Trying parallel splits" loop, 4 threads busy) on `Std(x, dims=(H,W))` at C=256 12x12 and on a ViT block at C=24 30x50; every other case of the sweep compiles in ~3 min | killed by watchdog; reduce C or split the op |

## E. Profiler and CLI tools

| Finding | Effect |
|---|---|
| `runner.profile()` rejects a compiled graph the compiler accepted: a layer with two outputs feeding a RoPE multiply is reported as shape [8,16,192] vs [8,8,192] | per-context FPS unavailable; we measure the sub-HEF on the chip instead |
| `runner.profile()` gives no FPS for multi-context HEFs without runtime data | use `hailortcli benchmark` / `run2` |
| `hailo profiler` CLI writes shell autocompletion into the environment's `bin` and fails | use the Python API |

## F. Emulators (SDK_BIT_EXACT / HSim.so)

| Finding | Effect |
|---|---|
| A3 reciprocal discontinuity, A5/A6 16-bit differences | emulator != silicon |
| HSim.so ships only sm_61 kernels + PTX (the sm_52 image is an empty stub) | no HSim on Maxwell GPUs; `hsim_sm52/` retargets the PTX |
| HSim launches its kernels on stream 0 while TensorFlow uses a non-blocking stream | race: random garbage in the tail of outputs, on any GPU; `hsim_sm52/` shim fences it |
| HSim reduce_mean rounds in float32 for |x| > 2^24 | inexact on Swin/DaViT |
| Bilinear resize (several modes) and rank-2 softmax have no bit-exact emulation | "Not all layers support bit exact" |

## G. HailoRT (4.23)

| Finding | Effect | Workaround |
|---|---|---|
| `InputVStreamParams` with `FormatType.AUTO` on a UINT16 input rejects uint16 numpy arrays ("numpy array item size is 2, not 1") | 16-bit-input HEFs cannot be fed with the default params | pass `format_type=FormatType.UINT16` explicitly |
| Batch ≥ 6 on multi-context HEFs hits the scatter-gather descriptor limit (64K descriptors, 512 B pages): "Desc page size value ... not optimal", FPS drops | batch 4-5 is the best | larger descriptor pages |
| A VDevice left open by an exception holds the device: every other process gets HAILO_DEVICE_IN_USE | 437 spurious failures in our sweep | release in `finally` |
| ROUND_ROBIN scheduler, 3 network groups from 3 threads; frame order / batch 1-16 | no contamination: all bit-identical to solo runs (8 triples, 553 HEFs) | none needed |
| Inter-context tensors always go through host RAM over PCIe (no DRAM on Hailo-8; CCB not supported for inter-context on PCIe) | design limit, documented here for sizing | minimise bytes crossing cuts |

## H. Firmware (APP + CORE images, ContextSwitchTask)

No functional bug found. Observed behaviour, useful for performance work:
- Each context ends with module_config_done + trigger_sequencer (one per cluster) -> sequencer_done_interrupt.
- The sequencer phase at the end of context k is context k+1's configuration load. It costs 0.28-0.98 ms with
  `mixed_mem=disabled` and a flat 1.75-1.85 ms with it enabled. On a 5-context model this is the whole batch-1 gap
  (5.4 ms).
- `measure-fw-actions` timestamps are 200 MHz; the NN core runs at 400 MHz.

## I. Third-party / environment (affects the DFC)

| Finding | Workaround |
|---|---|
| NVIDIA DALI's VMM allocator fails on Maxwell (CUDA_ERROR_INVALID_VALUE in MakeContiguous) -> DFC levels 3/4 (AdaRound) die | `DALI_USE_VMM=0`, or `nodali/` |
| TF in the DFC wheel loads the pip cuDNN 9.1, older than the cuDNN 9.4 it was built with ("No DNN in stream executor") | LD_PRELOAD the system cuDNN 9.5 |
| `pkill -f <pattern>` / `pgrep -f` in helper scripts match their own shell | kill by PID |
