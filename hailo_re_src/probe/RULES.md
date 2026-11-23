# Hailo-8 / DFC 3.33 compiler rules, measured by probing

Black-box method: tiny parametric ONNX nets (`sweep.py`, 661-case grid in `runs/grid`, plus the 453 compiled op
probes of `zoo/ops`) -> compile -> read the compiler's own record in the HEF (`mapping` = integrated_hw_graph
proto: per-node resources, cycles, latency profile, rule formulas) -> firmware timings on the chip (`rtprof.py`).

## 1. Where the numbers come from

| HEF field | meaning |
|---|---|
| `resources_list[i].frame_total_cycles` | cycles per frame of node i at 400 MHz |
| `bottleneck_fps` | exactly 400e6 / max(frame_total_cycles) |
| `profiler_data` | per context, per layer start/end cycle = pipeline fill (latency) |
| `se_formula` | per node: AND of clauses `"<feature used> AND !(<supported bit>)"` = capability rules |

## 2. Chip

* NN core 400 MHz; 8 clusters; the CP-SAT splitter allows <= 16 nodes and <= 64 subclusters per cluster.
* Conv peak ~59-64 MAC per subcluster per cycle (8-bit) -> 8 x 64 x 64 x 2 x 400 MHz = 26.2 TOPS (the datasheet).
* About 350 subclusters are usable in one context (16-layer chain overflows at 43 per layer).
* Cluster input stream ~8 B/cycle (1x1 s2 conv is input-bound); a concat of spatial slices streams ~3-4 B/cycle and
  is often THE bottleneck node.
* No DRAM: inter-context tensors and DDR portals go to host RAM over PCIe.

## 3. Capability matrix (from se_formula, 596 HEFs, `db/rules_v1.json`, `capmatrix.py`)

1 = supported, 0 = never, 0/1 = depends on the node's configuration.

| node type | 8->16 | 16x4 SC | dilation | 16-bit bias | 8-bit bias | no_contexts |
|---|---|---|---|---|---|---|
| CONV, DW | 0/1 | 0/1 | 0/1 | 1 | 1 | 0/1 |
| MATMUL | 0/1 | 0 | 0 | 1 | 1 | **0** (and max 1 output) |
| ACTIVATION, NORMALIZATION | 0/1 | 0 | 0 | 1 | 1 | 1 |
| STANDALONE_EW_ADD / SUB | 0/1 | 0 | 0 | 1 | **0** | 1 |
| EW_MULT | 0 | 0 | 0 | 1 | 0/1 | 1 |
| AVGPOOL | 0 | 0/1 | 0/1 | 0/1 | 1 | 1 |
| MAXPOOL, CONCAT, splitters, SLICE, SHORTCUT, IO | 0 | 0 | 0 | 0 | 1 | 1 |
| RESIZE, REDUCE_MEAN | 0 | 0 | 0 | 0/1 | 1 | 1 |
| FEATURE_MULTIPLIER | 0 | 0 | 0 | 1 | **0** | 1 |

A16_W16 is allowed on every type; 16->8 transitions everywhere; 4-bit weights everywhere.

## 4. Cost (grid medians; MAC per subcluster per cycle)

| op | 8-bit | 16-bit |
|---|---|---|
| conv 1x1 | 48 | 7.4 |
| conv 3x3 | 50 | **2.8** |
| conv 5x5 / 7x7 | 55 / 56 | |
| conv 3x3 stride 2 | 29 | |
| depthwise 3x3 / 5x5 / 7x7 | 1.9 / 2.7 / 2.9 | 1.7 |

* 16-bit costs 7-18x the compute of 8-bit (not 4x); 16-bit add 2.5x, mul 7x, concat 2x, activation 2.5x slower.
* Activations are free (relu = silu = sigmoid = tanh = leaky; identical cycles).
* Channels < 32 waste the MAC array (8 ch: 37 MAC/SC/cycle); >= 256 output channels force an output-feature split
  x16 at ~40.

## 5. Decisions the compiler makes

* Spatial defuse of a conv = ceil(W / 8) width slices (W16->2 ... W80->10), capped ~11-16 by per-cluster limits.
* Depthwise is defused by INPUT_FEATURES (x15-17).
* Big output channels -> output-feature defuse x16 (`conv1_d0..d15`).
* Pipelined chains run at the slowest layer: 2-8 equal convs = same FPS as one.

## 6. Blind spots / failures (rewrite before compiling)

| case | symptom | rewrite |
|---|---|---|
| depthwise 5x5 | never defused: 1 subcluster, 405 FPS vs 15k for 3x3 | split, or 3x3 cascade |
| depthwise 7x7 at >= 80x80 | "microcode exceeded size" | force a defuse |
| C2f block n=1, 128 ch | 1175 FPS vs 7318 for n=2 | (open) |
| elementwise Max / Min | 7x / 3.6x slower than Add | max(a,b) = a + relu(b - a) |
| spatial ReduceSum | 7x slower than spatial ReduceMean | ReduceMean x H*W |
| nearest resize x4 | 8x slower than x2 | two x2 resizes (exact) |
| MaxPool 5x5 s1 | 2x slower than 3x3 | maxpool3 o maxpool3 (exact) |
| spatial ReduceMax | parser: features axis + keepdim only | global MaxPool |
| SpaceToDepth after conv | parser: UnsupportedShuffleLayer | |

## 7. Multi-context rules (the external partition pipeline, `turbo/`)

* R1 a producer and its slice family stay in one context.
* R2 a sub-network input is host IO (tf_rgb conversion needs features % 8 == 0) -> sub-compile with reshapes off.
* R3 a context-entry layer must not be SPATIALLY defused.
* R4 sub-compile SPATIAL_RESHAPE pieces of format conversions -> remove_node in the full script.
* R5 an entry layer CAN be FEATURE-defused (stock: dw4 INPUT_FEATURES x4); the sub-compile does not try it.
* R6 single-context compiles keep weights in L2 (`mixed_mem=enabled`); multi-context reloads them through the
  sequencer at every switch (1.7 ms per context vs 0.25) -> `mixed_mem=disabled`, except nodes that become
  infeasible.
* Objective: a context's batch-1 time is its pipeline FILL (latency), not its period; the single-context default
  stops at a throughput target -> `performance_param(optimize_for_batch=1)` (ctx2 of nano: 1010 -> 4034 FPS).
* A sub-compile's choice can be infeasible inside the full model ("Validator failed on node X: Agent infeasible")
  -> fall back to that context's next-best variant (`hcc.py`).
* A transpose_matmul format conversion of an operand fed across a cut must be declared on the REAL producer.

## 8. Parser / quantizer failures by kind (195 native op-probe failures, `db/ops.jsonl`)

All 200 ONNX ops are covered after rewrites (`zoo/ops/OPS_SUPPORT.md`, `ONNX_COVERAGE.md`); natively:

| kind | n | meaning / fix |
|---|---|---|
| `IndexError: list index out of range` | 74 | unknown op silently DROPPED by the parser; surfaces in the next layer's shape calc -> PWL (`pwl_inject.py`) or host |
| `ParsingWithRecommendationException` | 18 | ArgMin, Equal, DFT, Einsum... -> compat rewrites |
| `Not all layers support bit exact` | 6 | bilinear down/aligned-corners resize: no bit-exact emulation |
| `float() argument ... list` | 6 | Pow with non-scalar exponent constant -> rewrite to Mul chains |
| `UnsupportedModelError` | 5 | Mod, MatMul with const W / two tensors, Pow of tensors |
| LSTM/GRU unroll limits | 5 | > 16 steps (8 bidirectional) refused; GRU crashes the allocator |
| `Greater` with negative theta | 2 | greater = thresholded ReLU, theta >= 0 only -> Not(Less) |
| LayerNorm over all dims | 1 | beta over all dimensions unsupported |
| GELU tanh form | 1 | allocator `map::at` -> rewrite to erf GELU |

## 9. yolo26m attention contexts (open)

Partition res70_seed1 (4 ctx). Contexts holding the attention block (format_conversion -> conv_feature_splitter
-> matmul -> softmax pieces -> matmul) do not compile as stand-alone sub-networks:
* default / lat: `matmul1: Memory units capacity exceeded (available 128, required 165)` (per-cluster memory units)
* relaxed resources_param 1.0: same (a hard per-cluster limit, not a cap)
* defuse(matmul1, 2, DYNAMIC_WEIGHTS) as stock does for matmul3: `Defuse insufficient rows (needed 2, actual 1)`
* stock's per-node compilation_param for matmul only: next node (`reduce_max_softmax1`) overflows memory
* stock's settings for the whole attention block: the sub-network no longer fits ONE context ("Initial
  utilization overflow")
* the transposed matmul operand needs its explicit `matmulN_reshape = format_conversion(..., transpose_matmul)`
* R1 as implemented is too strict here: stock keeps conv_feature_splitter11_1 and _2/_3 in different contexts
Next: partitions whose attention context carries fewer other layers (cut right before format_conversion1 and after
matmul2), searched with part_search.py.

## 10. Hardware model (decoded from the HEF's own config actions, `hwlayout.py`, `hwjoin.py`, `hwtab.py`)

The preliminary/context `enable_sequencer` action carries per-cluster bitmaps of the units switched on, and
`enable_lcu` names each layer controller. Summed over clusters they equal the compiler's per-node resources
exactly (sum subclusters = SC bits, sum apus = APU bits = LCUs, sum input_aligners = IA bits), so they ARE the units.

| unit per cluster (8 clusters) | count | whole chip | usable at the stock 0.75 caps |
|---|---|---|---|
| subcluster (SC, MAC array ~50 MAC/cycle) | 64 | 512 | 384 (seen exactly) |
| L2 memory bank | 64 | 512 | 384 (seen exactly) |
| LCU (layer controller; 1 per placed layer piece) = APU | 16 / 20 APU | 128 | 96 (seen exactly) |
| input aligner | 32 | 256 | 192 |
| L4 memory cuts per model | | 28 | ("29 L4 memory cuts max (28)") |
| L3 memory block ("memory units"; enable_sequencer initial_l3 index 0..127) | 128 | 1024 | hard, cap-independent |

Max seen per context over 3,595 HEFs (all probe/zoo/portfolio compiles): SC 505/512, L2 469/512, LCU 127/128,
APU 144/160, IA 171/256 -> the 0.75 caps are a default, not hardware (level max raises them to 0.9-1.175).

Laws that follow (each observed rule is one of these):
* **One placed layer piece lives in ONE cluster**: <= 64 SC (48 at 0.75). Per-piece ceiling = 400 MHz x 48 x ~50 MAC.
  Anything faster MUST be defused into pieces on several clusters -> this is why the compiler defuses big
  layers, and why single convs at 40x40 report `cl 1`.
* **16 LCUs per cluster** -> at most 16 layer pieces per cluster (the CP-SAT "<= 16 nodes" limit); a x16 defuse
  costs one whole cluster's controllers; a context holds <= 96 pieces at stock caps.
* **Stream port = 8 B/cycle** (3.2 GB/s): host INPUT is exactly 8.00 B/cycle max, feature splitters 7.99; any
  tensor wider than that per frame-time must be split into several streams (defuse/splitters).
* Pool/dw units are slow: MaxPool and DW bottleneck nodes run at 0.2-0.4 B/cycle -> pooling/dw throughput needs
  INPUT_FEATURES defuse (stock does x15-17).
* Row width: w*c > 128 KiB per row does not compile (c64 w4096, c256 w512 fail; c64 w2048, c16 w4096 pass).
* Attention: cycles/frame ~= heads * N^2 / 7.5 (the softmax/elementwise over the score maps), independent of c.
  e0 attention (qkv fed straight from a network/context input) crashes (`_M_range_check`) or overflows the 128
  memory units at N >= 784; the SAME block with a compute layer in front (e2) compiles up to N = 1600 -> R10.
* Large kernels get no extra SCs (k3..k9: 43 SC) -> FPS ~ 1/k^2 (k3 12552, k9 1567, k15 615).

Blind spots from this sweep (compiler under-allocates; candidates for forced defuse / seeds):
| case | FPS | vs neighbour |
|---|---|---|
| residual over 4 convs (c64 80x80) | 396, 18 SC | n=8: 3717, 363 SC |
| conv stride >= 4 | 2098, 6 SC | stride 3: 12159 |
| conv c=72 at 40x40 | 6653, 28 SC | c=64: 12552, 43 SC |
| dw5 / dw7 at 20x20 | 1 SC | dw9: 52 SC |

## 11. R10 (multi-context)
A context must not start at a format_conversion (the attention reshape): the producer moves forward into that
context (`part_fix.py`), so the attention is e>=1. yolo26m ctx3 failed exactly this way (input_layer -> reshape ->
qkv -> matmul3: "Memory units 128 vs 165/302").

## 12. Blind spots are deterministic; level max fixes most
Same FPS with allocator seed 1, seed 2 and all resources_param at 1.0 -> a fixed level-1 heuristic, not luck.
`performance_param(compiler_optimization_level=max)` (compile <= 90 s on these nets):
| case | level 1 | level max |
|---|---|---|
| C2f block n=1 c128 80x80 | 294 | 2412 |
| dw7 20x20 | 1012 | 25203 |
| dw5 80x80 | 204 | 5402 |
| conv c72 / stride 4 / k9 | 6653 / 2098 / 1567 | unchanged |
skip n=4: explicit defuse decisions (conv x5 spatial) 396 -> 1335.
hcc variants accept a `_max` flag (e.g. lat_max, lat_fs_max).

## 13. R11 LCU-path layers across a cut (assemble.py)
A bilinear resize runs on the LCU/PPU path (`hailo_rgb_to_lcu` input conversion). When its output crosses a cut,
declare `X_output_reshape = format_conversion(X, consumer, ppu_to_hailo_rgb)` (stock does; a sub-compile never
emits it because its consumer is a host output). Also keep format_conversion lines whose endpoints are pre-defuse
layer names. With this, all 34 boundary op probes (`bnd.py`, cut before and after each op) are bit-exact.

## 14. Conv tiling: SC = 8 output channels x 8 columns (defuse algebra)
* Spatial defuse count is FIXED at ceil(W/8) (8-column slices): any other count -> "Unsupported defuse".
* SCs per spatial piece = output features / 8 (c64 -> 8 SC, o32 -> 4 SC, same cycles) -> max useful SCs of a conv =
  ceil(o/8) x ceil(W/8); k9 c64 W40 already uses all 40 tiles = hardware-bound (FPS ~ 1/k^2 beyond that).
* For o > 64 the level-1 heuristic packs 16 channels per SC (features_per_mac 2, padded to x16) and halves the
  tile count. Nested defuse fills it: `L_d0, L_d1, L_dc = defuse(L, 2)` then each `L_dX` SPATIAL x ceil(W/8):
  | conv 40x40 k3 | stock | nested 2 x 5 | feature x2 only | feature x4 only |
  |---|---|---|---|---|
  | c72 | 6653 | 13107 | 2668 | 3689 |
  | c128 | 3832 | 7598 | 1535 | 2136 |
  | c64 | 12552 | 14719 | 4026 | 4188 |
  Feature defuse alone REPLACES the spatial split (slower). Level max does not find the nesting.
* Stride >= 4 with padding: the compiler inserts EXTERNAL_PAD on 1 SC (0.35 B/cycle) = the bottleneck (2098 FPS,
  stride 3 absorbs the pad into the SUPER_CONV reshape: 12159).
* `turbo/relieve.py` turns a sub-compile's slowest convs into nested-defuse hints; `hcc.py --relieve` runs them as
  round-2 `<best>_nd` variants and keeps them only if they rank better.
* Measured on yolo26n (winner partition c56-127): level max = same per-context latency (175.0/408.4, 10x compile
  time); relief hints overflow the already-full contexts (sub-network no longer fits one context) or are slower
  (ctx2 2.02 -> 2.87 ms). Both pay only where a context has slack (isolated blocks, sparse contexts).

## 15. R12 the map::at crash = an unreplayable matmul split (found with pseudo-models, mapat.py)
Pseudo-models `attn_ctx_gen.py` (pre convs -> PSA -> post convs, 1 or 2 PSAs) in 3 cut layouts x 4 variants:
* map::at only at N = 784 tokens (28x28), never at N <= 400 (14x14, 20x20), for c = 128 and 256, 1 or 2 PSAs.
* It ALWAYS coincides with an attention sub-compile that INPUT_FEATURES-defused the softmax x V matmul
  (`matmulN_fs_0, matmulN_fs_1, ... = defuse(matmulN, 2, defuse_type=INPUT_FEATURES)`: a 2-operand split, one
  operand behind the `transpose_matmul` conversion). Not the DYNAMIC_WEIGHTS split of Q x K (it replays fine).
* Proof: that sub-compile's own auto.alls does not replay on its own sub-HAR (map::at) -> the compiler can make this
  solution but cannot read it back; every assembled script inherits the crash. Even stock crashes when merely given
  `context([...])` of such a model, because it then also picks that split.
* Fix (R12, part_fix -> c{i}.extra.alls): `compilation_param(<softmax x V matmul>, resources_allocation_strategy=
  manual_scs_selection, number_of_subclusters=8)` keeps it whole: 1123.8 vs 1135.7 FPS, and the result replays.
  4 SC -> 200 FPS, 16 SC -> no partition. Stock's own yolo26m script has matmul4 whole with 8 SC: same rule.
* Infrastructure bug found on the way: parallel compiles in ONE cwd clobber each other's errors.pb / rotating SDK
  logs -> spurious `DecodeError: Unexpected end-group tag`, empty "Compilation failed:", "unexpected crash",
  `rotating_file_sink` errors. compile_alls.py / compile_sub.py / sweep.py now chdir into a private dir.

## 16. (was open) yolo26m full compile map::at -> R12
Every 4/5-context assembly (our R10 partition AND stock's own context partition) crashes with `map::at` at
allocation start, while each context compiles alone (ctx3 attention 1135.7 FPS). Bisect so far: not the portals,
not compilation_param/place; `remove_node(format_conversion4)` and the ctx3 defuse decisions of conv88/conv93/
conv97/dw6/matmul4 change the error -> several interacting decisions (`turbo/undefuse.py` reverts one layer's
defuse consistently for this bisect). The ctx3 variant without matmul3 DYNAMIC_WEIGHTS (lat_fs_sr) passes this
stage and fails later in the Splitter.

## 17. First yolo26m from the own pipeline (2026-10-25) + R6 is not universal
* Partition m_part_r12 (R1 forward, R10, R12 pins), combo default+default_sr+default+default_sr, full script = the
  assembled one WITHOUT the R6 blanket `mixed_mem=disabled`: compiled in 253 s (+ 308 s parallel sub-compiles,
  vs stock ~96 min), BIT-EXACT vs cprof/base_default.hef, one chip: b1 24.58 (stock 26.28), b4 39.57 (stock 38.81).
* With R6 every combo failed "Splitter failed" in ctx0 (core log: "Failed recording best-effort buffers:
  l4_portal_from_matmul1_d0_to_matmul1_dc was not found in toposort recipe" on the pseudo-model; with all
  mixed_mem enabled instead: "Parallel activation required but not assigned at: matmul1"). -> hcc keeps R6 and
  falls back to the sub-compiles' own mixed_mem choice on those errors (marker file r6_off).
* Pseudo-model check of R12: ac_c256_h28 (never assembled before) compiles bit-exact with the attention in the last
  context (233.9 FPS b1); attention followed by a cut needs the R6 fallback.

## 18. Op coverage of the own pipeline (2026-10-25, bnd.py: 2 convs -> OP -> 2 convs, cut before AND after OP)
Layer types the parser produces from the 259 ONNX op probes: 31 (pre-quantization HN). In the quantized HAR the
pipeline actually reads, layer_normalization / softmax / feature_splitter are already decomposed (reduce_mean,
ew_sub, feature_multiplier, reduce_max/sum ...), at C=16 and C=64 alike, so they never reach hcc as such. Every
remaining type is boundary-tested through hcc, chip bit-exact vs stock except where noted (bnd5/bnd6 at the
survey's own C=16 24x24: LayerNorm_C, InstanceNorm2d, RMSNorm_C, GroupNorm(4), ViT_block, Softmax_C/W, Argmax_C,
AvgPool3s1 +nopadcount all EXACT both cuts). Earlier count: 28 (runs bnd2 34 cases, bnd3 sweep families, bnd4 torch-exported `probe/op_ctx_gen.py` `@DIR` cases):
conv dw gconv maxpool avgpool(+count_include_pad=0 const map) ew_add/mult/sub max/min concat split slice resize
(nearest, bilinear) reduce_sum/max/mean softmax(C and W) matmul d2s s2d(PixelUnshuffle) deconv external_pad
feature_shuffle C2f-block activation layer_normalization(LayerNorm/InstanceNorm/GroupNorm(4)) batch_norm
normalization(div, L2-norm) shortcut(Tile) feature_multiplier ViT block (LN+MHA+GELU MLP, feature_splitter).
Not reachable: dense (`No format for fc1 -> conv3`, stock fails), const_input-only, argmax/mean_C cut_after
(op folded into the conv: probe artefact, empty first context).
* R13 (part_fix): a context never ENDS at a feature_multiplier (x^2): core assert
  `FeatureMultiplierKernel::set_default_power_table: m_out_amdc.size() ==` -> move it forward to its consumers.
  op_pow2 cut_after: 0/6 -> BIT-EXACT.
* R14 (part_fix, shares R10's loop): a context never STARTS at a mean-subtract fan-out (a tensor feeding a
  reduce_mean AND an ew_sub = a norm block): Splitter failed in that context (T_GroupNorm C=16 cut_before 0/4);
  moving the producer conv forward -> BIT-EXACT. (At C=64 it happened to pass.)
* bnd.py cut locator counts dw too (1-channel convs after argmax become dw; the old locator cut at conv1).
* STOCK SILENT-WRONG on the chip: GroupNorm(1) (= whole-tensor mean/var: reduce_mean [20,20,1] -> [1,1,1] ->
  resize -> sub -> square -> conv_var_inv [.,1] -> inv_sqrt) at C=64 is ~98% of bytes wrong vs SDK_BIT_EXACT
  (corr 0.66) even single-context stock; and in 2 contexts (stock forced with resources_param 0.10, or ours) the
  frames MIX (repeated-frame input differs from varied input; 255 errors). The op survey's 36.4 dB was HSim at
  C=16 where it fuses into one layer_normalization. GroupNorm(4) matches the emulator. -> rewrite whole-tensor
  norms (GroupNorm(1)) as per-channel-group statistics or run them on the host; never trust HSim alone for
  global reductions.
* mixed_mem is NOT numerically neutral for global reductions: stock script with only mixed_mem flipped to
  disabled differs by <=4 LSB (deterministic). Placement (resource caps) also changes it by <=4.
* Chip vs SDK_BIT_EXACT is not byte-exact in general (op_unshuffle 1,222/204,800 bytes at <=2 LSB, frame 0 of
  GroupNorm(4) 479 bytes): exactness of the own pipeline is judged against the stock HEF, which ours match.

## 19. yolo26m on STOCK's own 5-context membership (2026-10-25, turbo/t/m_part_stock15) + attention recipes
* R1 is stricter than the hardware: stock keeps conv22 in ctx0 and its conv_feature_splitter3_* in ctx1.
* Attention contexts need stock's per-layer recipe (c{i}.extra.alls), otherwise "Memory units capacity exceeded
  (available 128, required 165)" for the Q x K matmul in the sub-compile and Splitter failed in the assembly:
  ctx1 (matmul1/2, 400 tokens): compilation_param(matmul1|matmul2, manual_scs_selection, number_of_subclusters=2,
  mixed_mem=disabled), conv_feature_splitter6_* 1 SC.  ctx3 (matmul3/4, 784 tokens): remove_node(
  format_conversion4/5/6) [keeps 28x28 rows; without it "Defuse insufficient rows in layer: matmul3 Needed 2
  Actual 1"], matmul3 = defuse(2, DYNAMIC_WEIGHTS) at 4 SC each, matmul4 8 SC, conv_feature_splitter11_* 2/2/3 SC.
* Result: 4/64 combos assemble (all with ctx3 = lat_fs, 558 FPS sub; the 1124-FPS default variants fail the full
  Splitter / "Validator failed on node: concat21"). BIT-EXACT, one chip b1 23.44 / b4 39.15 (stock 26.28 / 38.81,
  own m_part_r12 24.58 / 39.57). b4 beats stock, b1 does not.

## 20. Why stock wins batch 1 on yolo26m: the per-context SEQUENCER phase = mixed_mem (fw actions, 2026-10-25)
`hailortcli run2 --mode raw_async measure-fw-actions set-net X.hef --batch-size 1` (turbo/t/fwact/) gives every
firmware action with a 200 MHz timestamp. Each dynamic context ends: output_channel_transfer_done ->
module_config_done + trigger_sequencer (one per active cluster) -> sequencer_done_interrupt -> deactivate.
* Same partition (stock15), same per-context bottlenecks (ctx0 107.7 FPS conv22_d0 in BOTH; ours >= stock in every
  context), yet b1 stock 38.0 ms vs ours 42.1 ms: the sequencer phase is 0.28-0.98 ms per context in stock and a
  flat 1.75-1.85 ms in ours (5 contexts: ~5.4 ms) -> the whole b1 gap.
* Cause: stock's script has mixed_mem=disabled on 319 layers and `allowed` on 30, and the 30 are exactly the
  compiler HELPER nodes (sh_from_, ws_from_, l4_portal_from_, ddr_portal_from_, auto_reshape_from_, concat_w_from_).
  Ours (R6 fallback after the blanket R6 failed) had 296 enabled. Blanket R6 disabled the helpers too, which is
  what broke yolo26m ("l4_portal_... not found in toposort recipe").
* hcc R6 is now r6_stock(): disabled on model layers + their defuse pieces, allowed on helpers.
* conv22 itself is NOT the difference: our ctx0 sub-compile chose stock's exact defuse(conv22, 16) x 4 SC.
* Partition search (turbo/part_refine.py, scored by T(b) = sum lat + (b-1) sum 1/fps + K*2.6 ms, which reproduces
  measured b1/b4 within 1%/3%): no boundary move around ctx0/1 beats stock's membership.
* r6_stock applied AFTER the sub-compiles fails (ctx0: "Splitter failed"): mixed_mem must be decided BEFORE
  allocation. hcc variant flag `_md` appends `compilation_param(L, mixed_mem=disabled)` to each sub-compile for every
  layer of the context (defuse pieces, not removed nodes, not layers the recipe already pins: a 2nd
  compilation_param line on a pinned layer resets its SC pin -> "input buffers should be divisible by 2").
* RESULT (m_w_md): BIT-EXACT, b1 27.61 (stock 26.28, ours-before 23.44); sequencer 0.35-0.59 ms/context
  (<= stock). b4 fell to 37.35. CORRECTION of a first guess (ctx1 variant): pinning ctx1 default_sr_md, plain
  default_sr in ctx1, ctx2 lat_fs_sr_md all give 27.6/37.3; every stock15 (mixed_mem enabled) combo gives
  23.4/39.1. Disabling mixed_mem is a b1-vs-b4 TRADE independent of variants. The sequencer phase at the END of
  context k belongs to context k+1's config load: plain ctx1 makes ctx0's sequencer 1.80 ms (hyb build).
  `--pin I=VARIANT[,...]` fixes contexts for such experiments.
* Per-context sweep (plain = stock15 variants, b1/b4; all plain 23.4/39.1, all _md 27.6/37.3):
  _md only in c0 24.0/35.3 | only c1 23.6/38.4 | only c4 24.2/39.1 | only c2 or c3: Splitter fails |
  c1-c4 _md, c0 plain lat 26.6/37.5. NOT additive: the b4 cost needs _md in the middle contexts, the b1 gain
  needs it everywhere. Pareto front (all BIT-EXACT) vs stock 26.28/38.81:
    m_w_md2  full_lat_md_default_sr_md_default_md_lat_md_lat_fs_md   27.61 / 37.35  (b1 +5.1%, b4 -3.8%)
    m_w_sw4  full_default_default_sr_lat_fs_sr_default_lat_fs_md     24.24 / 39.14  (b1 -7.8%, b4 +0.8%)
  No single HEF beats stock at both yet; pick per deployment batch. Open lever: what mixed_mem buys at b4
  (likely batch buffering in L3/L4 vs DDR) - next probe is ddr/l4 portal placement on the _md build.

## 21. Fuzz verdict + the retry-tier rules R17-R22 (2026-10-25, 424 forced cuts over 120 random nets)
* 393/424 bit-exact first pass (32 "void" cuts were infra: VDevice leak; recheck_infra all EXACT). 0 compile-but-wrong
  after rerun (fz_1043 was pre-R16). The 31 failures, rerun with R13-R16 (+14 exact), then the new rules below:
  ALL 424 bit-exact except fz_1011 cut_3_8 (SE block, Splitter fails even after R21) = 423/424.
* Rules (consumer pulled back into the producer's context). Measured on the fuzz: applying them UP FRONT changes
  127 cuts that already compile and costs FPS (fz_1068 -18%, fz_1111 -14%), so they are RETRY tiers: hcc compiles
  the partition as given; only a failed context / full compile re-runs part_fix with PART_FIX_STRICT=1, then 2
  (turbo/hcc.py strict_retry; tested: fz_1090 fixed at tier 1, fz_1075 escalates to tier 2, both BIT-EXACT).
  tier 1  R17 format_conversion -> X crosses (attention exit; map::at, "can't spatial reshape"): fail 12 / ok 4
          R18 token-strip tensor [., 1, N, C] crosses (inside attention; "(1,400,800) to (50,8,800) without a
              successor format_conversion")
          R20 feature_splitter output crosses ("non-unique elements" ValidationError)          fail 2 / ok 2
  tier 2  R19 pieces of one decomposed layer_normalizationN / softmaxN split ("Shapes at dim #2 aren't equal")
              fail 2 / ok 39 (+ R10 yields to R19: the norm group's start comes back instead of pulling forward)
          R21 dense (SE squeeze) output crosses fail 1/ok 7; R22 conv_slice piece away from producer 1/7;
          R16b any resize starts a context (nearest 8x8 failed once, fz_1055) 2/18
* In-distribution judge (DIST=normal) of the 14 uniform-input sweep flags (DeformConv_zero, DynQuant_ConvInteger,
  QDQ/QLinearConv, Scatter*, Sum3, Mean2, Einsum, ReduceLogSum_C, R_Mod, InstanceNorm, Softmax_W): chip == emulator
  in ALL 14 (61-75 dB vs float for the conv-like ones; ReduceLogSum_C 13 dB / InstanceNorm 20 dB = quantization,
  same on emulator). The uniform-input flags were saturation, not silent-wrong. GroupNorm(1) remains the only
  chip-wrong op (gn1 rewrite fixes it).

## 22. SILENT-WRONG: a dynamic matmul's 2nd operand is read as SIGNED int8 on the chip (2026-10-25)
* The in-distribution judge's open gap (bnd2 matmul probe: chip 9-14 dB vs emulator 24-29 dB, chip batch ==
  chip 1-by-1) is NOT frame mixing and NOT format_conversion removal (params-only compile: same). Truncating at
  the matmul (runs/mm_fc/out/mm_att) keeps it: both operands (conv3/conv4) are bit-exact chip == emulator, the
  matmul output is not: 9.3% of elements, ALWAYS chip < emulator (-1..-128 LSB), growing with the value.
* The wrong KEY tokens are exactly those where the K operand (conv4, post-ReLU, quantized UNSIGNED 0..255,
  zp 0) has a value > 127 (border pixels of the 10x10 map have smaller activations -> exact). Predicting the
  chip as Q . wrap_int8(K) matches it to 0.17 LSB mean (closer than my linear model fits the emulator itself);
  clip-to-127 and Q-side hypotheses are 20x worse. The emulator (SDK_BIT_EXACT) models K as uint8: STOCK HEF
  silently wrong.
* Workaround: quantization_param(<K producer>, force_range_out=[0, 2*max]) -> K data <= 127 LSB: chip ==
  emulator, 29-36 dB vs float (was 10-14 chip / 25-32 emu). Guard: probe/dynmm_check.py Q.har lists every
  dynamic-weights operand outside int8 and prints the fix line.
* yolo26m (all 4 matmuls) and SmolLM2-135M: K/V operands are already signed (zp 0, symmetric) -> unaffected.
  Affected nets: attention whose K (Q.K^T) or V (P.V) comes straight from a ReLU/non-negative layer.

## 23. Emulator-vs-silicon audit, round 2 (2026-10-26)
* YOLO26m 8/916 pages (chip != emulator on every head) = EMULATOR bug, silicon right: reduce_sum_softmax1 (acc =
  -2*sum + 2, shift 1, inv_pos PWL from x = -32767) returns 0 in SDK_BIT_EXACT/HSim for row sums 32769/32770 (APU
  input -32768/-32769); the chip returns 2, as both do for sums 32765 and 32774+. Bisected with attention sub-HEFs
  fed exact intermediates (runs/det/att: matmul1 + exp exact, one reduce_sum element per bad page). Chip output
  deterministic across both chips and 3 days.
* ConvTranspose k2 s2 is SILENT-WRONG on silicon (-17 dB vs emulator +32 dB); k4s2/k3s2 exact. Exact rewrite: conv1x1
  C->4C + DepthToSpace(2, CRD): chip == emulator 35.7 dB.
* SDK_BIT_EXACT clamps raw UINT16 inputs at 32767; silicon uses 0..65535 (raw 33000: chip 30026 vs emu 29828).
  16-bit layers differ by 1-2 LSB on ~0.2% of outputs at any amplitude.
* dynmm_check.py fixed: qp = [zp, scale]; multi-output layers store scale 0 there (use layer_params/output_scales);
  a signed_output operand is clipped to int8 by the quantizer itself (yolo26m K: signed, safe).
* Full write-up: hailo8-exact-qat/docs/HAILO8_ISSUES.md and HAILO_BUG_REPORTS.md.

## 24. Emulator fixes from the chip scans (2026-10-26)
* APU edges: silicon extends the first/last PWL segment past the table ends; SDK_BIT_EXACT zeroes them. Chip map of
  451,584 softmax rows (runs/det/att/recip_chip.py, recip_fit.py): edge model 0 mismatches, SDK rule 6 (S=32769/70).
  hailoqat intops.apu + hq_ext/hq.cu now extend edges; HAILOQAT_SDK_EDGES=1 (python) / -DHQ_SDK_EDGES (cuda) = SDK.
* Softmax denominator: reduce_sum_softmax is a 16-bit WRAPPING accumulator (APU in = -S+1 mod 2^16). Row sums
  > 32770 wrap: 32771-65535 -> reciprocal 0-1 (should be 1-2), >= 65536 arbitrary. 2.6% of yolo26m C2PSA rows.
  Chip == emulator, so this is a quantizer accuracy loss, not a mismatch. Candidate fix: smaller exp output scale.
* Dynamic matmul (section 22) exact rule: producer marked signed (weights edge) but non-negative -> chip runs its
  APU clip2+output UNSIGNED 0..255, matmul reads int8. graph26.silicon_weights_operands + intops x1_bits:
  10,000/10,000 vs chip (runs/a1_exact); HAILOQAT_SDK_MATMUL=1 = SDK.
* graph26 input clamp is 0..2^bits-1 (UINT16 inputs use the full range on the chip).
* graph26 run(cont=True) substitutes the HSim reference for every mismatching layer: hypothesis tests downstream of
  a changed layer need cont=False.
* Shape sweep (6 shapes x 90 cases): no new chip-vs-emulator item beyond GroupNorm(1) (wrong at all 35 grid shapes)
  and the 1-3 LSB A5 family (ViT block, norms). Std over H,W at C256 12x12: allocator > 60 min (killed).

## 25. SILENT-WRONG: dilated conv piece 9-10 wide (2026-10-26)
* Any compiled piece of a dilation-3 conv (undivided layer or a W slice of a SPATIAL defuse) with output width 9 or
  10 is ~99% wrong on the chip and history-dependent (batch != 1-by-1). Pieces 3-8 and 16 wide are exact.
  W sweep 7-64 (runs/dil3/w): bad at 9, 10, 18 [8,10], 26 [16,10], 34 [16,8,10], 50 [16,16,8,10].
* Defuse count / buffer format changes do not move the 10-wide slice. FIX: zero-pad W to a multiple of 8, conv,
  crop (runs/dil3/pad: chip == emulator, 38.5 dB). Guard after every compile: probe/dil_check.py M.hef Q.har.
* yolo26m has no dilated convs (unaffected). Other dilations (2, 4+) not yet swept.

## 26. Attention, softmax, GN1 and k2s2 fixes for ALL models (2026-10-26)
* D18 SILENT-WRONG: stock INPUT_FEATURES-defuses the softmax x V matmul at ANY token count and the chip result is
  1-6 dB (single-head N=64/784, runs/smx). R12 now pins every softmax x V matmul (part_fix; was >= 400 tokens only).
* Softmax denominator: the 16-bit wrapping accumulator is sized from the CALIBRATION max (utilize_wraparound);
  off-calibration rows wrap (N784 t1.0 at half amplitude -15 dB). force_range_preact does NOT move it.
  turbo/smx_headroom.py (SMX_HEADROOM=full|<factor>, hooked in sweep.py) sizes it for N x max(exp): big wins
  off-calibration, up to -5 dB in distribution when the calibrated range is far below the worst case (8-segment
  reciprocal table spread wider) -> choose per model on VALIDATION data. Bug: widening the range's near end to 0
  fits the reciprocal to 1/0 (row sums are >= 1) -> extend only the far end.
* Zero point of K / V: turbo/attn_bias_rewrite.py (compat rule attn_bias): K bias dropped, V bias moved after P.V,
  per element by perturbation (handles fused MHA in_proj); exact in float; chip +8.5 dB (K offset), +2.3 (V).
* k2s2 ConvTranspose: the DECONV -> FEATURE_INTERLEAVE lowering is the defect (stock and ours 99-100% wrong);
  SUPER_DECONV on the same HAR is exact. R23 (part_fix) forces it; compat rule convt_k2s2 rewrites in ONNX.
* GN1: wrong with stock and ours, 1 or 2 contexts; centring half exact on chip; no layer_norm_decomposition knob
  fixes it (split_precision 49% -> 31%). Keep compat rule gn1 (exact, both compilers).
* Front end for any model: zoo/ops/onnx_hailo_compat.py --rules ...,gn1,convt_k2s2,dil_pad,attn_bias (compat4.py
  model passes, float-checked). graph26/intops emulate the silicon (APU edges, int8 weights operand).

## 27. Guard + production check (2026-10-26)
* turbo/hailo8_guard.py M.hef C.har: G1 dyn-matmul weights operand (unsigned > 127 OR signed-but-non-negative),
  G2 dilated 9-10 piece, G3 P.V INPUT_FEATURES, G4 k=s DECONV lowering, G5 GN1 decomposition, G6 softmax
  denominator wrap risk (warning). hcc runs it on every winner (hcc.json guard / guard_ok). dynmm_check.py is
  superseded (it trusted signed operands).
* yolo26m val with the edge-fixed executor: mAP50 0.886052 == chip. Saturating the softmax denominators of the
  QAT-trained HAR: 0.8841 / 0.8162 (worse): the model learned the wrap -> headroom only before QAT.
* A5 (16-bit 1-LSB flips) open: not APU rounding / acc truncation / byte-plane MAC / product truncation.

## 28. Every workaround closed and chip-verified (2026-10-26, runs/forum: cx_*, dynmm_r4, own_*)
* Dilated conv rule (chip, d = 2..4): a compiled piece is wrong iff 9 <= width <= 8 + d - 1. The compiler cuts a row
  into 8-aligned pieces + ONE remainder piece (W50 -> 16+16+8+10, W18 -> 8+10, W17 -> 8+9, W11 d3 -> 8+3), so an
  output width that is a multiple of 8 never makes a bad piece. dil_pad now pads the OUTPUT width (stride-aware).
  d >= 5 takes other routes (d5: width_feature_reshape; d6 W13 undivided exact): guard warns only.
  cx_dil_* (W9 d2, W10 d3, W11 d4, W17 d2, W18 d3, W50 d3/d4) all chip == emulator, 37.7-38.3 dB.
* ConvTranspose shape: the parser builds EVERY deconv as TF-SAME (out = in * stride, odd pad at the END) and drops the
  ONNX begin pad: k3s2p1 on 24 -> 48 x 48 = onnx 47 x 47 plus one leading row/column (hailo[1:,1:] == onnx, 139 dB;
  in the emulator too, so not chip-only). compat4 convt_same: SAME-padded ConvTranspose + Slice -> chip 35.0 dB, exact.
* kernel == stride deconv: compat4 convt_ks (was convt_k2s2) does any k = s: k4s4 -> Conv1x1 16C + DepthToSpace(4):
  chip exact 38.2 dB (was -6.2). R23 covers it in our compiler for HARs that skip the front end.
* G1 margin: the forced weights-operand scale is range/255 (qp_out 2.513/255) while the matmul reads int8, so values
  above HALF the forced range wrap. force_range_out=[0, 2*max] left 4.33% wrong (chip 26.7 vs emu 33.4);
  [0, 4*max] is bit-exact at amp 1 AND 2 (32.2 dB). Symmetric [-R, R] is -10 dB on chip AND emulator (zp defect).
  Guard / run.py --fix-dynmm now print 4*max.
* compile_sub.py self-repair: after every sub-compile the compiler's own auto script is read; an INPUT_FEATURES
  defuse of ANY dynamic matmul -> SC pin, a DECONV lowering of a k = s deconv -> SUPER_DECONV; recompiled once,
  dropped if it survives (res["repair"]). hcc guards every benchmarked candidate and ships the fastest guard-clean one.
* Our compiler on the probes (runs/forum/own_*): k4s4 via R23 and via compile_sub repair alone (R23 removed) both exact
  38.2 dB; plain dynamic matmul: OUR sub-compile chose INPUT_FEATURES itself -> repair pinned it, the full compile then
  hit "Can't set BufferFormat: DIVIDE_BY_ROWS of layer conv2 to matmul1" (assemble removes the format_conversion the
  sub-compile buffered for) -> hcc.repair sets that layer's buffer format automatic -> exact, BIT-EXACT vs stock HEF.
  k3s2p1 (convt_same), dil W50 d4 (dil_pad), gn1rw: exact. Guard G1 reads the quantized HAR's script (a compiled
  HAR's script is the compiler script: forced ranges were invisible -> false S1).
* yolo26m regression (turbo/t/m_w_regress_20261026, same pins as c0plain_b): same full.alls, BIT-EXACT, guard ok;
  re-benched side by side today: old 24.5/33.4, new 24.5/33.4, stock 24.0/35.3 (b1/b4) -> no change from the patches.

## 29. 16-bit compile stalls: super-defuse, token rows, partitioner start, solve server (2026-11-22, PP-DocLayoutV3 mix16)

Model: convert/runs/layout_g3 (r5_ord5, mix16, upsample blocks 16-bit). Tools: probe/layer_cost.py, turbo/compile_sub.py,
cprof/phases.py (named stack profile), turbo/turbo2.cpp hooks.

**A. "Assignment of the node needs super-defuse" = 16-bit tensors on a 784-wide token row.**
* Localised by sub-HAR bisect: only the AIFI encoder context fails. Alone, the 16-bit positional-embedding const_input
  (1x784x256, 392 KiB/row) is "Agent infeasible", and the 8->16 precision_change after the token reshape fails
  "more than 1 subclusters are needed for 16bit L2 biases". Manual SUPER_CONV / feature defuse do not help.
* Cut rules for bisecting: start a sub-HAR BEFORE a format_conversion and include const_input producers (else
  "A16_W16_A16 is not supported in tf_rgb_to_hailo_rgb..."); a cut ending at a model output gives map::at.
* FIX (float-exact): convert/recipes/aifi_spatial.py keeps every per-token op on 28x28 (1x1 conv WITH folded bias, LN
  over axis 1, pos-emb as a 28x28 const, P.V bias folded into o_proj, qkv identity layers dropped). The parser still
  puts the conv producing a matmul's 2nd operand (K, V) behind the 8-bit token format_conversion -> those two must be
  a8_w8_a8 (convert/recipes/kv8_alls.py), or quantization re-adds an 8->16 precision_change on the token row.
  Encoder alone: 1 context, 339 FPS. Box-AP gate PASS (max drop .0022).
* A bias-free Conv + separate Add is parsed as a `mul_and_add` normalization layer, not folded: emit Conv with bias.

**B. The stall is the Slotter's starting point, not its speed.**
* Stack profile of the running compile: 19/20 samples in allocator code (Slotter::solve_and_validate_partition ->
  "Adjust Edges" -> Shiftman/Racehorse::Automri), 1/20 in CP-SAT. So the CP-SAT cache/server cannot help here.
* hailo_sdk.core.log of each candidate: ~1,200 iterations in 4 h, ~85% "Automri finished with too many resources on
  context_N", 0 valid partitions; the context count crept 2 -> 10 at ~1 per hour.
* Decoded (fn 0x15f5960, called once before find_initial_partition): per-context budget
  B = max(1, floor(min(graph_estimate, ceil(hw_units * x0)) * x1)), x0 = resources_param cap: 128 * 0.6 = 77.
  find_initial_partition fills each context greedily to B estimated units. Raising the cap (u0.7..0.9) packs fuller
  contexts = worse; context_switch_param max_*_utilization (0.3 -> 0.15/0.10) does NOT change the start.
* Hook: HAILO_INIT_SCALE=s (B*s) / HAILO_INIT_UNITS=n / HAILO_INIT_LOG=1. RESULT: s=0.12 (B=9) found the FIRST valid
  partition of this net (28 contexts, Unikorn est. 1.97 FPS) 2 h 40 min after the search started; stock (B=77) and
  s=0.5/0.3/0.2 had none after 3-7 h. A performance_param(fps=5/10/20) target does NOT change the search (same ~270
  failed iterations at context 2); logger_param(default, file_verbosity=trace, ...) adds no detail (needs a logger name;
  console_verbosity on `default` is refused: no console sink).
* Compile-verified contexts (turbo/ctx_grow.py): grow each context by real single-context compiles of candidate spans
  (context_switch_param(mode=disabled)), 8 ends in parallel. Layout: ctx0 = 29 layers (stem), ctx1 = 15. A TIMEOUT is
  not "does not fit": infeasible spans fail fast (4-6 min), tight feasible ones search long (> 7 min).
* Budget-hook results on layout_g3: B=9 (s 0.12) -> 28 contexts after 3 h 9 min, but final mapping of context 5 hit
  the dynamic overall timeout (~4 h 15 min, "Resolver didn't find possible solution / Splitter shutdown timeout"):
  the Slotter's "valid" is weaker than the final per-context mapping. B=6 (s 0.08) -> valid 41 contexts after 53 min.
  Raise the overall limit with allocator_param(timeout=SECONDS) (mapping_timeout default is model-size dependent).
* ctx_grow on layout_g3: 10 compile-verified contexts for layers 0-134 (29,15,1,13,4,12,4,15,18,24). 16-bit weights
  make single layers fill a context (conv30 1664->512 1x1 = 1.7 MB; neck 3x3 256->256 ~1.2 MB). Stopped at 135:
  conv67 alone = map::at. CORRECTED (F below): not a subhar bug, the compiler crashes in its utilization raise.

**C. A layer's footprint follows the THROUGHPUT TARGET, not its width.** (probe/layer_cost.py: a layer of the real
q.har compiled alone at its real shape/precision, resources read from the HEF.)
* Compiled alone at the default level, the compiler spreads a layer over the idle chip: 112-wide 16-bit conv4 = 15
  HW nodes / 113 subclusters (124 FPS), 16-bit 256->1024 1x1 conv65 = 15 nodes. These are UPPER bounds (an earlier
  note here claiming "fixed ceil(W/8) pieces" was wrong).
* With `performance_param(fps=20)`: conv4 = 4 nodes / 10 SC (21 FPS), conv65 = 1 node / 7 SC (85 FPS). With a 0.15 cap:
  9 / 65 and 8 / 57. Measure footprints under a realistic FPS target, never at the default.
* Consequence for B: contexts overflow because Automri sizes layers for an FPS goal the 16-bit net cannot reach;
  test = full compile with performance_param(fps=N) at a realistic N.

**D. Modern-solver server only for big objective models.** All 236 models PP-DocLayoutV3 sent were < 500 vars:
server 15.6 s per INFEASIBLE, 62 s per UNKNOWN; the layer-option phase ran 40+ min instead of 4. turbo2.cpp now sends
only models with an objective and >= HAILO_CPSERVER_MINVARS (2000) variables (wire scan, exact on 423 requests).

**E. Process traps.** Killing compile_hef.py leaves its `compiler` child alive (ppid 1) with the OLD env/hook: kill
compilers by /proc/PID/cwd. Never pkill/pgrep -f a pattern that appears in your own command line (exit 144).

**F. map::at in "Trying to apply higher utilization solution" (2026-11-23).** conv67 (3x3 256->256, 2 inputs, 56x56)
compiled alone: "Model fits in single context" -> converged 71/384 SC -> "Trying to apply higher utilization solution"
-> map::at. It FITS; the crash is in the spread-over-the-idle-chip step. performance_param(fps=20) = OK in 12 s,
fps=100 = map::at, fps=400 = "Failed to reach fps" (max 136.5). ctx_grow.py now retries map::at / fps-miss spans down
CTXG_FPS_LADDER (default 400,200,100,50,20) and records fps_target. Relative OUT paths break its worker (it chdirs).
* Full-model mapping failures (s=0.12, 28 ctx, splitter_timeout=3600 cluster_timeout=1800): context 5 again, 22
  cluster-mapping reverts on the 16 conv30 pieces fed by shortcuts from the 1664-ch 16-bit concat4; ctx_grow found
  conv30 fits only ALONE in a context. Longer timeouts do not fix a Slotter context that cannot be mapped.


**G. The Slotter start budget is a NODE COUNT, not a cost (2026-11-23, static RE of compiler.dis).** 0x15f5960:
`graph_est` = call 0xe1c670(*self) = number of graph nodes passing 0xe14980 (three exclusion predicates 0xe12960 /
0xe12930 / vtable+0x48 and node type != 0x15: compute nodes only); `hw_units` = hw->0xb4 * hw2->0x1fc = 128 (layer
controllers). B = max(1, floor(min(n_nodes, ceil(128 * 0.6)) * x1)) = at most 77 NODES per starting context, every
node counted 1 whatever its size. Check: B=9 -> 28 contexts, B=6 -> 41 contexts, both ~250 nodes / B. A 1.7 MB
16-bit conv30 piece weighs the same as a tiny ew_add, so the even start overloads 16-bit nets and the hill-climb
never digs out. Proper fix (open): seed find_initial_partition (0x1608e70) with a size-weighted / measured split
(ctx_grow / part_pack output) instead of the count-even one.
* Confirmed in find_initial_partition (allocator::Slotter, slotter.cpp, 0x1608e70, 0x9d0 bytes; objdump, Ghidra not
  needed): n_ctx = ceil(n_compute_nodes / B) (divsd + ceil at 0x1608f14..f51), then nodes in topological order, a
  context closes when its compute-node count reaches the quota (e14980 filter + cmp at 0x16091cc..dd). Pseudo-model
  check (probe/runs/slotcount, conv chain n=24, 16-bit 256 ch at 56x56): hook log "stock 25" = 24 convs + 1 node = min(n, 77).
* rea (Ghidra 12.1.4) on the 50 MB compiler: import exceeds the 330 s startup deadline; with analyzers disabled it
  fails in 17 s. Use cprof/compiler.dis + fnmap_lib.strings_of (__PRETTY_FUNCTION__ / assert strings) instead.

**H. Compiler FPS estimate vs live chip, conv-chain pseudo-models (2026-11-23, probe/runs/slotcount, n=24 3x3 at
56x56, hailortcli run -t 5 on 0000:02:00.0).** est / chip FPS, compile time:
small 64ch 8-bit 1 ctx 1907 / 1679 (64 s); 64ch 16-bit 1 ctx 265 / 263 (65 s); 256ch 8-bit 1 ctx 144 / 143 (121 s);
256ch 16-bit 2 ctx (stock B 25, 73 Slotter iterations, 3 m 41 s) 23.1 / 7.99, batch 8 = 7.99, batch 16 = 6.99.
Single-context estimates are right (<= 12%); the multi-context estimate is ~2.9x optimistic for 16-bit weights
(context switch reloads them every frame). 16-bit costs 6.3x (small) and 18x (big, 2 ctx) of 8-bit on the chip.
Expect the layout HEF well below its 1.97 FPS estimate.
* n=100 chains (same probe, all stock B=77 -> start = ceil(100/77) = 2 contexts): small 8-bit 2 ctx 76 iterations
  7 m 20 s; small 16-bit 4 ctx 243 it 15 m 26 s; big 8-bit 6 ctx 283 it 29 m 12 s; big 16-bit 9 ctx 440 it 1 h 1 m.
  Search effort grows with the distance between the count-even start and the real context count = section G.

## 30. HEF back end: config writes, address map, weight layout, chip-verified edit (2026-11-23, hefre/)
Full notes: `hefre/HEF_BACKEND_20261123.md`. Tools: `hefre/ccwdec.py`, `hefpatch.py`, `chiprun.py`, `wexp/`.
* A context's config = CCW stream: `u32 nwords-1, u32 address, data, pad 8` (HailoRT parse_ccw_buffer).
* Cluster k = address window `0x1100_0000 + k * 0x0200_0000` (1,814 / 1,814 grid HEF-cluster pairs).
  `+0x00_8000..` subcluster L2 (32 KB slots); `+0x40_0000` = same memory through the decompressor.
* Default `hef_param(params_load_time_compression=allowed)` compresses the images (62-entry byte dictionary at
  `0x1180_0000` / `0x1980_0000` + nibble codes). Decode with `params_load_time_compression=disabled,
  should_compress_sequencer_data=False`. `compress_l3_weights` is unrelated (L3 only).
* 1x1 conv 8->8 int8: plain int8, 16 B per input channel (cin descending), words `[w(c,co), w(c,co+2), 0, 0]`
  duplicated per subcluster; cout split across clusters by defuse. Bias/zero-point words per channel at ~+7360.
* CHIP: patching 2 weight bytes + XXH3 rehash -> HailoRT loads it; only that output channel changes on hailo0.
