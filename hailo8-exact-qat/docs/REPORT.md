# Technical report: exact-arithmetic QAT and deployment on Hailo-8

Toolchain: Hailo DFC 3.33.1, HailoRT 4.23.0, 2× Hailo-8 (PCIe Gen3 x4 each), 4× Tesla M40 (sm_52) for training.
Models:
- YOLO26m document-layout detector, 14 classes, 896×896;
- YOLO26n, same data at 640, for controlled head-to-head comparisons.

All mAP numbers in this report were computed from the chip's integer arithmetic, through an executor that is
verified bit-exact against Hailo's emulator and against the silicon.

## 1. Evaluation protocol

| Set | Pages | Authors | Use |
|---|---|---|---|
| train | 18.4k | — | QAT and calibration |
| val | 916 | shared with train | Monitoring only |
| bench_sel | 1,385 | 18 unseen | Checkpoint / model selection |
| bench_test | 1,385 | 21 unseen | **Reported numbers** |

- Bench pages were never used for training or calibration.
- Val was dropped as a selection set. It shares authors with train and rewarded overfitting: nano all-16-bit QAT gained
  +0.035 mAP50-95 on val and +0.000 on bench.
- **Paired image bootstrap.** Differences are judged with `boot_ci.py`, 1,000 resamples, a 95% CI of the difference.
- **Tie-breaking.** Int8 confidences tie massively, and the reference `ap_per_class` uses an unstable sort, so mAP
  moved with the number of eval shards (0.8547 vs 0.8521). Ties are now broken by a fixed hash of (image, rank).

## 2. The Hailo-8 datapath, as reverse-engineered

The executor (`intops.py`, `graph26.py`; trainable twin `qatfast.py` + `hq.cu`) reproduces these operations:

| Operation | Behaviour |
|---|---|
| int8 conv / dw | Each 8×8 product is shifted right by the layer's pre-accumulator shift (1-4) with **banker's rounding per product**, then summed in a **16-bit wrapping accumulator**. Integer bias. Zero-point compensation is folded into the bias |
| a16_w16 conv | 16-bit inputs × 16-bit weights (stored unsigned + `zp_kernel`), shift 0, **32-bit** accumulator. Bias = `f32(f32(F·i)·R)`, with `i` a 16-bit integer and F, R per-layer constants; both products are float32-rounded |
| APU (activations) | Per-channel-group piecewise-linear tables: thresholds, 10-bit mantissa / 4-bit exponent slopes, offsets. SiLU uses 11 pieces. Applied to the accumulator (int64 needed for 32-bit accumulators) |
| ew_add / ew_sub / ew_mult | Separate input rescales with conv-add factors (`round(desired/feed)`), repeats |
| softmax | Decomposed by the DFC into matmul → reduce_max → ew_sub (exp APU) → reduce_sum (1/x APU) → ew_mult, all 8-bit |
| matmul zero-point correction | When it cannot be fused, materialized as linear + reduce_sum + concat. The concat is **group-interleaved** when it feeds a grouped matmul |
| concat / maxpool / resize | Pure integer data movement under a shared scale |

Verification:
- **YOLO26m int8 + 16-bit heads:** 155/155 layers exact from raw pixels.
- **YOLO26n:** top-8 16-bit 154/154, all-16-bit 206/206, int8 143/143.
- **Silicon** (YOLO26m, 916 val pages): 908 pages bit-identical between chip and executor. On the other 8, the chip
  also disagrees with Hailo's own emulator (HSim) on about 8% of one layer's elements, while executor == HSim. Chip
  mAP = emulator mAP to 4 decimals (0.88605/0.81760 vs 0.88603/0.81759).

Traps found on the way:
- Summing then rounding instead of rounding per product drifts.
- A float32 conv is not exact at 512 taps. Fixed with nibble split / digit planes.
- The float conv-add factor read before `enforce_encoding` is the pre-quantization one.
- `a8_w8_a16` on a non-output layer gives its consumer a 16-bit *wrapping* accumulator (mAP 0.02).
- The 16-bit bias is decomposed, so training the accumulator value directly produces unstorable biases.

## 3. Why exact QAT beats the DFC's own optimizers

DFC levels:
- **Level 0:** equalization only.
- **Level 1:** adds iterative bias correction.
- **Level 2:** adds QFT, 4 epochs on 1,024 images, weights only, L2 loss on the last layers.
- **Levels 3/4:** add AdaRound.

All of them optimize `SDK_QUANTIZED`, a fake-quant model that skips per-product rounding and accumulator wrap. On
YOLOv8m we measured `SDK_QUANTIZED` score errors of up to 0.13 against the chip, while `SDK_BIT_EXACT` matched the
chip.

Exact QAT optimizes the integers the chip will actually run:
- **Forward:** the bit-exact pipeline.
- **Backward:** straight-through.
- **Loss:** one2one detection loss plus KD from the FP model.
- **Bounded latent integers:** kernel ±4, bias ±256 around level 0. Adam moves every parameter by about the LR on
  every step, including those whose gradient is pure noise. Unbounded runs drifted below level 0.

### 3.1 YOLO26m results (bench_test, 1,385 pages)

| Model | mAP50 | mAP50-95 | Cost |
|---|---|---|---|
| FP (ceiling) | 0.9161 | 0.8553 | |
| DFC level 0 + 16-bit heads | 0.9012 | 0.7811 | 17 min CPU |
| DFC level 2 (default) + 16-bit heads | 0.9113 | 0.8222 | 89 min, 1 GPU |
| **Exact QAT from level 0, 1 epoch (q3a) — fast mode** | **0.9197** | **0.8426** | ~1 h, 3 GPUs |
| Exact QAT from level 2, 3 epochs | 0.9128 | 0.8394 | ~2.5 h, 3 GPUs |
| Exact QAT + 8 × a16_w16 — accurate mode | *see 3.3* | | |

- q3a vs DFC level 2: +0.0084 mAP50 [+0.003, +0.014], **+0.0204 mAP50-95** [+0.015, +0.025].
- Starting QAT from level 2 helped on val but not on unseen authors.
- Without training, the 16-bit head outputs alone raise level 0 from 0.731 to 0.762 mAP50-95 (val). The int8 head
  outputs cost 0.029 mAP50-95, mostly through box precision.

### 3.2 YOLO26n head-to-head, every DFC level

The nano model makes level 4 affordable (6.3 h), so every DFC option is compared with every QAT mode on the same data.

| Model (YOLO26n @ 640) | bench_sel | **bench_test** | Time |
|---|---|---|---|
| FP | 0.862 / 0.750 | 0.842 / 0.720 | |
| DFC level 2 + 16-bit heads | 0.851 / 0.695 | 0.833 / 0.674 | 26 min |
| DFC level 4 (AdaRound) + 16-bit heads | 0.848 / 0.710 | 0.832 / 0.680 | 6.3 h |
| Exact QAT, int8, 3 ep | 0.851 / 0.711 | 0.832 / 0.687 | 35 min, 1 GPU |
| DFC level 0 + all layers a16_w16, no training | 0.852 / 0.724 | 0.836 / 0.697 | 5 min |
| **Exact QAT, top-8 a16_w16 (lattice bias), 3 ep, picked on bench_sel** | **0.856 / 0.725** | **0.837 / 0.698** | ~1 h, 1 GPU |
| Exact QAT, all layers a16_w16, 3 ep | 0.860 / 0.737 | 0.826 / 0.698 | ~2 h, 1 GPU |

Cells are mAP50 / mAP50-95. Level 0 int8 alone is 0.512 / 0.367 on val: nano does not survive naive PTQ.

Bootstrap on the full bench, 2,770 pages, using the first top-8 run (same layers, pre-lattice bias, ep 2: 0.845 / 0.707):
- Top-8 16-bit QAT vs int8 QAT: +0.012 mAP50-95 [+0.007, +0.016].
- Top-8 16-bit QAT vs DFC level 4: +0.008 mAP50 / +0.016 mAP50-95, both significant.
- Int8 QAT vs DFC level 4: mAP50 tie, +0.005 mAP50-95 (borderline), in 1/10 of the time.

What this shows:
- **Exact QAT beats AdaRound** at a tenth of its cost.
- **Top-8 16-bit is the sweet spot.** All-16-bit removes almost all rounding noise but gains nothing more on unseen
  authors and costs 3.7× the chip time.
- **The remaining ~0.02 mAP50-95 gap to FP is not rounding.** All-16-bit has almost none and still shows it. Candidates
  are the APU's piecewise-linear activation approximation, the remaining 8-bit ops (ew_add, softmax, concat) and
  calibration clipping.

### 3.3 YOLO26m accurate mode

Sensitivity scan (`sens.py`, 256 pages, level 0):
- The int8 rounding-noise cost is extremely concentrated. One layer, `conv70`, the conv feeding a classification
  head, carries 79% of it, and the top 4 carry 94%.
- The 8 picks are all head-branch convs: `conv70 conv86 conv100 conv97 conv69 conv_feature_splitter11_2 conv67
  conv60`. Together they carry 96% of the score.
- Layers whose outputs cross a context cut in the baseline HEF were excluded, so their tensors don't double on the
  link.

Bias lattices: `bias_lattice.py` verified `f32(f32(F·i)·R)` on every channel of all 8 layers. Exactness: 167/167
layers from raw pixels.

Training: 2 epochs, q3a recipe, 2× M40, about 1.6 h. A checkpoint was saved every 1,150 steps, and every checkpoint
was scored on the whole bench.

| Model | bench_sel | **bench_test** | full bench |
|---|---|---|---|
| FP | 0.9233 / 0.8679 | 0.9161 / 0.8553 | 0.9143 / 0.8559 |
| q3a (fast mode) | 0.9221 / 0.8476 | **0.9197 / 0.8426** | 0.9176 / 0.8401 |
| top-8 a16_w16, level 0, no QAT | 0.9036 / 0.8189 | 0.8975 / 0.8020 | — |
| top-8 a16_w16 QAT, step 5750 (best on bench_sel) | 0.9329 / 0.8670 | 0.9140 / 0.8408 | 0.9155 / 0.8461 |
| top-8 a16_w16 QAT, final step 9200 (no selection) | 0.9313 / 0.8663 | 0.9147 / 0.8411 | 0.9154 / 0.8464 |
| **step 5750 after write-back (round-tripped HAR)** | 0.9314 / 0.8660 | **0.9145 / 0.8415** | **0.9156 / 0.8462** |

Paired bootstrap on the full bench against q3a:
- step 5750: mAP50-95 +0.0060 [+0.0019, +0.0113], mAP50 −0.002 (tie);
- final step, which involves no selection on bench: mAP50-95 +0.0063 [+0.0021, +0.0116].

Reading:
- The 16-bit model is **significantly better on the full unseen-author bench**, by about +0.006 mAP50-95.
- The gain is concentrated in the bench_sel authors, where it nearly reaches FP (0.8660 vs 0.8679).
- On bench_test it ties q3a.
- The write-back round trip is exact on 167/167 layers. The DFC re-decomposed two 16-bit biases (2 and 6 steps of
  `i`), at no measurable cost.
- The same pattern appeared on nano: 16-bit removes rounding noise, and the remaining gap is author-dependent rather
  than precision-limited.

Chip cost: PENDING (HEF compile in progress).

## 4. Running a model bigger than the chip: context switching

Hailo-8 has no DRAM. When a network does not fit its on-chip memory, the compiler splits it into **contexts**:
- the host DMAs each context's configuration (weights and microcode) in turn;
- every activation that crosses a cut, and every "DDR portal" buffer, travels to host RAM over PCIe and back.

### 4.1 Tools

| Tool | What it gives |
|---|---|
| `hefre/hef_dump.py` | Parses the HEF: a big-endian header with magic `0x01484546`, then HailoRT's `ProtoHEFHef` protobuf, then the CCW blob. Per context: config bytes, action mix, every edge tensor with its bytes per frame |
| `hailortcli run … collect-runtime-data` | Firmware per-action timestamps at 200 MHz. Per-context time per batch, and the action the firmware waits on |
| `mopt/part/hwgraph_json.py` | Dumps the HEF's `mapping` field (the compiler's integrated HW graph): every node's subclusters, APUs, memory and **`frame_total_cycles`**. The slowest node of a context predicts its period |
| `best.auto.alls` inside a compiled HAR | The compiler's full allocation solution as a model script: context lists, layer splits, cluster placement, buffers, portals. **Replaying it compiles in ~14 min instead of ~96**, with an identical HEF and FPS |
| `mopt/part/ilpcut.py` | Exact MILP (scipy/HiGHS) for the minimum-bytes context assignment, with precedence and per-context resource caps. Its cost model matches the HEF: 25.65 vs 26.0 MiB |
| `fwre/fw_split.py` + Ghidra | Firmware split into APP CPU (103 KB at 0x60000, signed) and CORE CPU (60 KB at 0xC0000), decompiled as Cortex-M Thumb |

Firmware structure:
- FreeRTOS tasks.
- `ContextSwitchTask` (0xC63FC) walks the context list once per batch and executes packed action records (type
  byte, 4-byte timestamp, payload).
- The firmware does no tensor math; it sequences DMA and core configuration.

### 4.2 Where the time goes

Per-frame traffic of YOLO26m q3a (5 contexts), from `hef_dump`:

| Per frame | MiB |
|---|---|
| Input | 2.3 |
| Outputs | 0.6 |
| Inter-context activations | 13.0 out + 13.0 in |
| DDR portals (attention `dw1`/`dw6`, `conv31`) | 5.5 out + 5.5 in |
| **Total** | **39.8** |

Configuration is 17.1 MiB per batch.

**First hypothesis, PCIe bandwidth: refuted.**
- The token-tail rewrite (4.4) removed both attention DDR portals and cut traffic to **32.4 MiB/frame (−19%)**.
  FPS did not change: 36.0 at batch 4.
- `measure-nnc-performance` measures the hardware only, with no host software in the loop. It gives the same FPS as
  full inference, so host software and the driver are not the bottleneck either.

**Measured bottleneck: the per-context pipeline period.**

Firmware timings, per batch of 4 frames:

| Context | ms per batch of 4 |
|---|---|
| ctx0 | **39.5** |
| ctx1 | 26.4 |
| ctx2 | 22.7 |
| ctx3 | 8.7 |
| ctx4 | 4.2 |

- Each context's time is almost entirely one wait for the compute units to finish (`lcu_interrupt`). That wait is
  the on-chip dataflow pipeline running at the pace of its slowest stage.
- The compiler's own per-node cycle counts predict the measurement exactly: ctx0 = 3.71 M cycles at 400 MHz =
  **9.29 ms/frame**.
- ctx0's slowest stages are the cluster output streams carrying **`conv21`**, the P3 feature map
  (112×112×512 = 6.1 MiB). They move about 1.7 bytes per cycle per stream.
- ctx2's slowest stage is the same tensor coming back in (5.07 ms).
- ctx1 is compute-bound (`conv35`/`maxpool2`, 4.36 ms).
- **One tensor sets about 14 of the 25 ms per frame.**

| Effective compute | Value |
|---|---|
| Work per frame | 133 GOP |
| Achieved throughput | 3.5–5.2 of 26 TOPS |
| Limiting resource | On-chip streaming of high-resolution wide tensors, not MACs and not the link |

Host and driver knobs gave nothing or −5%: `HAILO_DDR_ACTION_LIST`, aligned-CCW toggles, HEF copy modes. Firmware
patching would not help: the image is signed (LCS 3), and it does not run the pipeline.

### 4.3 Throughput (`hailortcli run`, 8 s, FPS)

| HEF | Contexts | 1 chip b1 | 1 chip b4 | 2 chips b1 | 2 chips b4 |
|---|---|---|---|---|---|
| YOLO26m q3a, default compile (0.6 caps) | 5 | 26.0 | 36.0 | 52.0 | 71.9 |
| q3a, max compile (0.7, ~30 h) | 4 | 27.0 | 40.0 | 53.0 | **79.9** |
| q3a, caps 0.75 (2.3 h) | 4 | — | 40.0 | — | 75.9 |
| q3a, caps 0.8 (2.7 h) | 4 | — | 40.0 | — | 79.9 |
| q3a, caps 0.9 (5.3 h) | 3 | 25.0 | 40.0 | 50.0 | 80.0 |
| token-tail attention (mTT, 32.4 MiB/frame) | 5 | 25.0 | 36.0 | 50.0 | 72.0 |
| token-tail + split pushdown (mTP) | 4 | 21.0 | 32.0 | 42.0 | 63.9 |
| token-tail + split pushdown + channel split (mTPC) | PENDING | | | | |
| YOLO26n int8 + 16-bit heads | 3 | 161.8 | 327.7 | 323.7 | 655.6 |
| YOLO26n top-8 a16_w16 | 4 | 113.9 | 207.8 | 227.8 | 415.6 |

Observations:
- **Batch.** Batch 4–5 is optimal. At batch ≥ 6 HailoRT hits its 64K scatter-gather descriptor limit (512 B default
  page) and FPS drops.
- **Chips.** Two chips scale almost linearly. Deploy on every chip.
- **Contexts.** Going from 5 to 3 contexts gives at most +11%. Contexts are not the limit; the ctx0 period is (3
  contexts: ctx0 still takes 41.2 ms per batch of 4).

### 4.4 Graph rewrites (FlashAttention-style: same math, different dataflow)

All of these are exact. Each script asserts head-conv agreement with the original in onnxruntime (≤ 2e-6 relative).
The end-to-end top-k output is not a usable check, because it reorders near-tied detections on float rounding.

| Rewrite | Script | What it changes | Measured |
|---|---|---|---|
| pe reads spatial V | `onnx_pe_spatial.py` | Feeds `pe` from a V-only 1×1 conv | No change: wrong transpose (below) |
| **Token tail** | `mopt/psa_tt.py` | `pe(v)` computed spatially and converted *into* tokens; proj, residual and ffn run on (B, C, 1, N); one token→spatial reshape at the block output | **Both attention DDR portals gone**, −19% PCIe bytes, FPS unchanged (not the bottleneck) |
| Concat pushdown | `onnx_concat_pushdown.py` | `Conv1x1(Concat(Resize(u), s))` → `Resize(Conv_u(u)) + Conv_s(s)` | No change: the DFC fused `Conv_s` with the add, which pulled it behind the cut |
| Split pushdown | `onnx_concat_pushdown.py --split` | `Conv_s` as two output halves + Concat, so `ew_add_fusing` cannot fuse it | `conv21` no longer crosses at 512 channels, but ctx0 slowed to 11.4 ms: the compiler rebuilds the full 512-channel tensor in one concat stream for its in-context consumers |
| **Channel split** | `onnx_channel_split.py` | Wide Conv+SiLU → two independent halves; every consumer conv → sum of two partial convs over the halves | PENDING |

Why the first rewrite missed:
- The DFC fuses `pe` and the attention residual add into one 2-input depthwise layer, and splits it as `SUPER_DW`.
- The DDR portal sits inside that split, on the attention-output input.
- The token tail avoids the 2-input depthwise entirely. A micro-model of one block (`mopt/micro/`) compiles in
  minutes and confirmed it: 1.97 MiB/frame of DDR in each direction before, none after.

Literature these follow:
- Fused-Layer CNN accelerators (Alwani et al., MICRO 2016);
- DeFiNES (Mei et al., HPCA 2023);
- LoopTree;
- Stream (Symons et al.);
- min-cut pipeline partitioning.

The common point: on a dataflow accelerator, schedule and layout decide speed, not op count.

### 4.5 Partitioning

The partition was solved exactly with `ilpcut.py`:
- The compiler's 5-context partition is **provably optimal** at its own 0.6 resource caps. The MILP optimum equals
  the dual bound, 25.65 MiB.
- With caps up to 1.6× and only the attention chain kept together, the minimum drops to 19–21 MiB (−19 to −25%).

Two ways of forcing a partition were tried, and both failed:
- **Manual `context([...])` with HN layer names** fails with `_Map_base::at`. The compiler needs every node it
  creates when splitting layers to be listed, and those names exist only after it has split.
- **Editing the membership of a replayed `auto.alls`** keeps splits that were chosen for the old contexts, and the
  cluster splitter then rejects every changed context ("Splitter failed").

So partitions must come from the compiler's own search. Raising `resources_param` is the working knob (table
above).

The partition search is **single-threaded**: one core at 100% whatever `compilation_num_threads` /
`num_of_workers` are set to. On a many-core host, run several searches in parallel, with different caps,
`partitioner=sloth` or `slotter_chances`, and keep the best HEF.

## 5. Training-speed engineering (Maxwell)

Exact QAT is integer arithmetic emulated in float on GPUs without int8 tensor cores. Speed came from:

| Step | Effect |
|---|---|
| Fused MAC tail kernel (accumulate, conv-add, wrap, banker's shift, bias, APU, grad scale in one pass) | 3.2× |
| int32 datapath, 2-D grid (64-bit div/mod is software on sm_52) | ✓ |
| Single fused conv `[x, A(x)] × [k, −B(k)]` when the fp32 sum provably stays below 2²⁴; runtime fraction flag falls back otherwise | Avoids nibble splitting on all 126 layers |
| Custom backward (no surrogate forward conv), shuffle reductions | ✓ |
| 16-bit layers: fp32 digit-plane convs (2 × 8-bit or 4 × 4-bit planes), mod-2³² integer combine, int64 APU | Bit-exact; 0.495 vs 0.463 s/step int8 (nano); float64 path was 20× slower |

Results:
- **Throughput:** 13.7 img/s exact QAT on 4× M40 for YOLO26m at 896. The plain-PyTorch reference ran 1.31 img/s.
- **Eval:** a full val eval takes 2.5 min (was 23 min).
- **HSim on Maxwell:** Hailo's HSim.so ships sm_61 code only. `hsim_sm52/` retargets its PTX to sm_52 and fences
  its stream-0 launches; that race corrupts results on any GPU.

## 6. Recommendations

| Goal | Recipe |
|---|---|
| Best accuracy per FPS (default) | Fast mode: int8 + 16-bit heads, exact QAT 1 epoch from level 0, max compile, 2 chips × batch 4 |
| Maximum accuracy | Accurate mode: add a16_w16 on the top-8 sensitivity layers; train 2 epochs, pick on bench_sel; report the round-tripped HAR |
| More FPS on a multi-context model | Use every chip (2× measured), batch 4–5, `resources_param` 0.8–0.9 or the max compile (+11%). Find the limiting stage first: firmware `collect-runtime-data` + `hwgraph_json.py` frame_total_cycles (4.2). On YOLO26m@896 it is the 112×112×512 P3 stream, not PCIe; DDR-portal removal and partition changes do not move it |
| Do not bother | DFC level 3/4 (AdaRound: slower and less accurate than exact QAT); trainable APU tables; firmware or driver patching for speed; hand-written context lists |

Open problems:
- Speed: a narrower or lower-resolution P3 (for example 768–832 input, or a narrower model.4) would cut the limiting stream directly but needs FP retraining.
- The ~0.02 mAP50-95 residual at 16-bit (APU approximation suspected).
- Pinning re-quantized 16-bit kernels exactly at export. Currently ±15/32767 on a few layers, about ±0.003 mAP.
