# Where each conversion step runs (CPU vs GPU)

Status 2026-11-22. `convert_gpu.sh` is the GPU pipeline; `convert.sh` is the older all-DFC pipeline (kept for reference).

## Runs on the GPU

| Step | Tool | Notes |
|---|---|---|
| Torch reference outputs (`data` stage) | recipe `*_torch.py` | fp32 torch on one M40 |
| Quantized emulator gate (`emu` stage) | `lib/gpu_emu.sh` | Exact Hailo-8 integer executor from `hailo8-exact-qat` (`qatfast.QATNet`, or batched float64 `graph26` when QATNet cannot build a layer, e.g. a16_w16 conv with a fused add). Router 296 vectors ~1 min; flat3d ~0.5 s/img at batch 8. Replaces the DFC `SDK_QUANTIZED` emulator, which runs 16-bit emulation on ONE CPU core (flat3d 800 frames ~4 h single process). |
| Task gates (`gate_q`, `gate_chip`) | recipe `*_gate.py` | torch heads / AP / table decode; light host post-processing |

## Stays on the CPU (no GPU option)

| Step | Time per model | Why |
|---|---|---|
| Hailo compile / allocation (`compile`) | 15-25 min (layout fails after ~15 min, see below) | Hailo's compiler is CPU-only: a placement search, not number crunching. `convert_gpu.sh` serializes compiles with `runs/.compile.lock`. Our partitioner (`hailo_re/turbo/`) can split it into per-context compiles, still CPU. |
| HSim capture (inside `gpu_emu.sh`) | 0.5-5 min | Hailo's bit-exact simulator on ONE eval input records the integers, scales and APU tables the GPU executor needs |
| Calibration (`quant`) | 1.5 min (orient) to 16 min (layout) | DFC statistics; partly CPU. Could be replaced by our own calibration on the GPU executor (not done). |
| ONNX parse, `hailo8_guard` | seconds | DFC parser / HEF checker |
| FP parse check (`gate_fp`) | first 32 rows only | DFC `SDK_FP_OPTIMIZED`; only catches parse/rewrite errors |
| Chip run (`chip`) | seconds-minutes | The Hailo-8 itself; host side is light |

## Exactness of the GPU executor (measured)

| Model | One-input coverage vs HSim | vs the real chip |
|---|---|---|
| RegionRouter (all16 + 16-bit input) | 6/6 layers exact | 99.3-99.8 % of output values identical, 100 % decisions. The rest differ by 1 LSB on 23/296 rows; on such a row our executor == HSim 6/6, so it is silicon vs HSim on 16-bit dense layers. |
| flat3d (all16) | 237/240 exact; conv4/11/20 (a16 5x5 conv + fused add) <= 1 LSB on 0.002-0.025 % of elements | pending |

## Executor fixes 2026-11-22 (all in `hailo8-exact-qat/hailoqat`, backups `*.bak_*`)

| Bug | Symptom | Fix |
|---|---|---|
| cuDNN float32 convs may use FFT/Winograd (choice depends on batch size) | flat3d batch 8 != batch 1 (up to 18,598 LSB); qatfast 20k/22k outputs wrong at batch 8, no flag | `intops._conv2d`: `HQ_CONV=gemm` (default) = float32 with cuDNN OFF (im2col + GEMM, exact for integer partial sums < 2**24 in any order); all exactness convs in intops + qatfast use it. `HQ_CONV=f64` also exact (4-6 s/img on M40). |
| `atomic.gather` flattened the whole batch into one vector | slice/pad/reshape layers returned batch 1 | maps recorded on one sample are applied per sample |
| a16 conv + fused add summed acc + residual exactly | 1 LSB on 0.002-0.12 % of elements (orient3, flat3d) | `HQ_CONVADD_SUM=f32` (default) = HSim's f32(f32(acc) + f32(term)); `exact` kept for a chip A/B |
| qatfast refused a16 conv + fused add | whole model fell back to float64 graph26 (4-5x slower) | those layers run the exact frozen graph26 conv-add inside QATNet |
| graph26 kept every activation | GridNet OOM at batch 4 | inference drops a tensor after its last consumer (`G26_KEEP_ALL=1` keeps all) |
| input quantization in float64 | router: 3/1024 input ints off | float32 like HSim / HailoRT; UINT16 inputs clamp to 2**bits-1 |
| missing `letterbox.py` | capture.py import error | added |
| CUDA float32 division is not correctly rounded | 16-bit input layer 1 LSB off on ~1 / 9,216 values (27 one-op probes) | `intops.f32_quantize`: float64 quotient rounded to float32 (= IEEE f32 division, 53 >= 2*24+2 bits), same for + zp; graph26 + qatfast |
| `concat_axis` (spatial_h / spatial_w) ignored | spatial concats joined on features: (384,1,24) instead of (16,24,24); Einsum, MatMulInteger, QLinearMatMul, DFT, RotaryEmbedding probes wrong or crashing | `torch.cat` on H / W when the HN says so; graph26 + qatfast |
| no `argmax` layer | ArgMax / ArgMin / Hardmax / OneHot probes crashed | channel argmax, first index on ties (HSim checked on all ties), `reverse_order` = last; graph26 + qatfast |
| free-after-use dropped tensors feeding a host postprocess | yolov5s / yolov6n / efficientdet KeyError | inputs of `postprocess` layers are kept (they are the chip outputs); graph26 + qatfast |
| qatfast `_refresh` on a net with no 8-bit MAC layer | all-16-bit nets: "stack expects a non-empty TensorList", 38 cases fell back to graph26 silently | nothing to refresh -> return |
| qatfast float32 datapath holds <= 24-bit lossy elements | 16-bit reduce_sum + log (32-bit element): QATNet refused the net | that layer alone runs the exact float64 graph26 handler inside QATNet (frozen), logged |

Prebuilt kernels: `hailoqat/hq_ext/build_prebuilt.sh` -> `hq_ext/prebuilt/hq_ext.so` (-O3, -Xptxas -O3, sm_52 + sm_86 +
compute_86 PTX, no fast-math); qatfast loads it first (`HQ_EXT_JIT=1` forces the JIT build).

Speed on flat3d (800 frames 712x488, one M40): DFC SDK_QUANTIZED ~4 h single process; graph26 batch 16 0.69 s/img;
qatfast (exact) batch 16 0.38 s/img.

## Test suite

`hailo8-exact-qat/tools/ops/gpu_suite.py` (+ `gpu_suite_all.sh`, `capture_probes.py`): per case T1 per-layer exact vs
HSim, T2 chained exact, T3 batch invariance (2/3/5/8/16), T4 qatfast == graph26; results in `results/gpu_suite/`.
Corpus: 13 zoo nets, the conversion models, 128 one-op probes (8-bit `c3/` + 16-bit `c3_16/`), 144 cases.
FINAL 2026-11-22 (all 144 cases rerun on the final code, 4 GPUs): 144 / 144 PASS, qatfast builds on 144 / 144
(1 case uses the graph26 fallback for one layer). Coverage: 31 layer types, 115 type|precision|activation combos,
precisions a8_w8_a8 / a8_w8_a16 / a16_w16_a8 / a16_w16_a16, 21 activations (relu, relu6, leaky, sigmoid, silu, swish,
hardswish, gelu, tanh, exp, log, sqrt, inv_sqrt, inv_pos, clip, threshold, greater, less, pwl, linear, none).
PASS now also requires T4: qatfast must BUILD and equal graph26 (before, a qatfast that failed to build passed
silently). Pre-fix run (94 pass, 29 fail, 21 crashed; 41 without qatfast) archived in `run_*_prefinal/`.

## HEF compile level

`convert_gpu.sh` compiles with `performance_param(compiler_optimization_level=max)` by default (fastest chip binary;
`COMPILE_OPT=default` for a quick stock compile).

Older fixes: `letterbox.py` added (capture.py imported it); input quantization in
`graph26.py` / `qatfast.py` now float32 like HSim/HailoRT (float64 missed 3/1024 rounding ties); `qatfast` clamps
UINT16 inputs to 0..2^bits-1 (the input layer of the router is 15-bit).

## Rules

* One HSim capture per HAR; the executor refuses to run if the capture-row coverage is worse than "exact or <= 1 LSB
  on < 0.1 % of elements" per layer.
* Hailo compiles: one at a time (CPU and RAM). GPU emulator jobs: one per GPU, each under a systemd memory cap.

## Compile acceleration (2026-11-22)

| Route | Tool | Status |
|---|---|---|
| Replay cache | `lib/compile_fast.sh` (+ `arch_sig.py`, `extract_auto_alls.py`); cache `compile_cache/RECIPE_SIG.auto.alls` | IN the pipeline (`convert_gpu.sh` compile stage). The signature covers layer names/types/shapes/precision, not weights: a retrain of the same model replays the previous allocator solution and skips the placement search. Router: full 37 s -> replay 16 s; chip outputs of the replayed HEF identical on all 296 inputs, same FPS. Earlier measurement (yolo26m): 96 -> 14 min. A failed replay falls back to a full compile. |
| Own partitioner + parallel per-context compiles | `hailo_re/turbo/` part_gen.py -> part_fix.py -> hcc.py (compile_sub.py per context, assemble.py) | Running on PP-DocLayoutV3 (k = 5, 8, 10 contexts), `HCC_PAR=4` (default 64 overloads the VM). Bit-exact vs stock and +8.8 % FPS on yolo-nano; multi-context assembly on yolo26m was still open. |
| Portfolio of stock searches | parallel stock compiles with different seeds / search knobs | Next. Measured trade-off: fast knobs can halve FPS (nano 82-84 vs 161), so every candidate is chip-benchmarked. |
