# hailo8-exact-qat

Quantization-aware training for the **Hailo-8** NPU in which the training forward pass *is* the chip's integer
arithmetic: per-product banker's-rounding MACs, 16-bit wrapping accumulators, integer biases, and the piecewise-linear
activation unit (APU). It runs on GPUs (fused sm_52 CUDA kernels for Tesla M40) with multi-GPU DDP. The trained
integers are written back into a Hailo HAR and compiled to a HEF with the standard Dataflow Compiler (DFC 3.33.1).

Model in this repo: YOLO26m document-layout detector (896x896, 14 classes).

## Results

Held-out test pages: 1,385 pages from 21 authors that were never used for training or model selection
(`configs/bench_split.json`). All Hailo numbers are exact chip arithmetic. Brackets are paired-bootstrap 95% CIs of
the difference to the recommended model (`hailoqat/boot_ci.py`).

| Model | mAP50 | mAP50-95 | Produced in |
|---|---|---|---|
| FP model (ceiling) | 0.9161 | 0.8553 | |
| DFC level 0 ("fast" conversion) + 16-bit heads | 0.9012 | 0.7811 | 17 min CPU |
| DFC level 2 (QFT, DFC default) + 16-bit heads | 0.9113 | 0.8222 | 89 min, 1 GPU |
| **This repo: exact QAT from level 0 (q3a, recommended)** | **0.9197** | **0.8426** | ~1 h, 3 GPUs |
| This repo: exact QAT from level 2, 3 epochs (final1) | 0.9128 | 0.8394 | ~2.5 h, 3 GPUs |

- Recommended model vs DFC level 2: +0.0084 mAP50 [+0.003, +0.014], **+0.0204 mAP50-95** [+0.015, +0.025].
- Recommended model vs FP: within 0.013 mAP50-95; mAP50 is above FP.
- DFC level 4 (AdaRound) was not included in the comparison.

### On the chip

Measured on the HEF of the recommended model, default compile, 5 contexts:

| | |
|---|---|
| Throughput (`hailortcli benchmark`) | 26.2 FPS per Hailo-8 |
| Latency | 36.9 ms |
| Raw head integers matching the training executor | 908 / 916 val pages |
| Val mAP from chip outputs vs emulator | 0.88605 / 0.81760 vs 0.88603 / 0.81759 |

On the other 8 pages the chip also differs from Hailo's own bit-exact emulator (HSim), while this executor agrees with
HSim layer by layer (155/155). Root cause: an emulator defect in the softmax reciprocal (the silicon is right); see
`docs/HAILO_BUG_REPORTS.md` report 4.

`results/viz/` shows ground truth, the FP model and the Hailo-8 chip output side by side on six random test pages.

### What mattered

1. **16-bit head outputs** (`configs/alls_a16heads.txt`, `a8_w8_a16` on the six one2one head convs).
   - Level-0 mAP50-95 went from 0.731 to 0.762 with no training.
   - The int8 head outputs alone cost 0.029 mAP50-95, mostly through box precision.
2. **Exact forward, straight-through backward.** QAT optimizes what the chip computes. DFC QFT and AdaRound optimize
   `SDK_QUANTIZED`, which skips per-product rounding and accumulator wrap.
3. **Bounded latent integers** (`--k_radius 2..4`, `--b_radius 128..256`).
   - Adam moves every weight by about the learning rate on every step, including weights whose gradient is pure noise.
   - Without a bound, the integers drift and accuracy falls below level 0.
4. **Fair metric.**
   - Int8 confidences tie massively, and `ap_per_class` sorts with an unstable sort. mAP therefore depended on the
     number of GPU shards (0.8547 vs 0.8521).
   - Ties are now broken by a fixed hash of (image, rank).

Negative results:
- **Trainable APU tables:** `--apu_train`; destructive at high learning rates, neutral at small ones.
- **Overflow penalty:** neutral.
- **KD-only loss:** worse than KD + ground-truth loss.
- **Starting QAT from DFC level 2:** helped on val (same authors as training) but not on unseen authors.

## Hailo-8 defects found

`docs/HAILO8_ISSUES.md` catalogues every chip, emulator, parser, quantizer, compiler, profiler and HailoRT defect we
measured, with coverage and workarounds. `docs/HAILO_BUG_REPORTS.md` has the 15 we consider genuine Hailo bugs,
written as issue reports. Reproduction scripts are in `tools/bugrepro/`. The compiler rulebook is `docs/COMPILER_RULES.md`.

Silent wrong results on silicon (the emulator says the model is fine):
- dynamic matmul with a non-negative second operand;
- GroupNorm(1);
- ConvTranspose with kernel 2, stride 2.

The emulator is wrong and the silicon is right:
- the softmax reciprocal at an exp-sum of 32,769-32,770. This is the cause of the 8/916 YOLO26m pages above.
- raw UINT16 inputs above 32,767.

## Layout

| Path | Contents |
|---|---|
| `hailoqat/qatfast.py` | Trainable exact network (fused kernels, custom autograd); `qatnet.py` is the plain-PyTorch reference |
| `hailoqat/hq_ext/hq.cu` | sm_52 kernels: input expansion, fused MAC tail, fused backward, APU-table gradients |
| `hailoqat/intops.py`, `graph26.py` | float64 reference executor, verified against the SDK's bit-exact emulator |
| `hailoqat/train_qat.py` | DDP training (teacher KD + one2one detection loss) |
| `hailoqat/evaluate.py` | Exact-arithmetic mAP |
| `hailoqat/capture.py` | Hooks the SDK's `call_bit_exact` to extract per-layer integer parameters and ground truth |
| `hailoqat/writeback.py`, `ints2npz.py` | Trained integers back into a HAR, verified against the translated params |
| `hailoqat/teacher_cache.py` | Precomputes FP teacher outputs |
| `hailoqat/bench_qat.py`, `bench_tail.py` | Exactness, gradient, timing and profiler harnesses |
| `hailoqat/boot_ci.py`, `split_map.py` | Paired bootstrap; per-author-split scoring |
| `hailoqat/chip_compare.py`, `chip_map.py`, `viz_pages.py` | Chip-vs-executor comparison, chip mAP, renders |
| `tools/` | Quantization (`b2_levelN.py`), compile (`compile_hef.py`), chip runs (`chip_dump.py`); experiment runners (`exp.sh`, `exp2.sh`, `evalgpu.sh`, `har_pipeline.sh`); environment (`env.sh`) |
| `hsim_sm52/` | Runs the SDK's HSim.so CUDA kernels on Maxwell (details below) |
| `nodali/` | Hides NVIDIA DALI from the DFC when needed (details below) |
| `configs/` | DFC model scripts, `.hn` graphs, dataset names, bench author split |
| `results/` | `eval.jsonl` (every evaluation), quantize/compile logs, renders |
| `runs/` | Per-experiment args and rank-0 training logs, queue scripts |
| `artifacts/` | **Not in git:** HARs, HEF, trained-integer checkpoints (~2 GB) |

### `hsim_sm52/`

HSim.so ships only sm_61 kernels plus PTX. `build.sh` retargets the PTX to sm_52, and an `LD_PRELOAD` shim swaps the
image in.

The shim also fences HSim's stream-0 launches. HSim launches on stream 0 while TensorFlow uses its own non-blocking
stream, which is a race on any GPU.

### `nodali/`

DALI's GPU reader fails on Maxwell (`MakeContiguous` invalid argument). This switch hides DALI from the DFC, which then
falls back to its TF dataset path.

The DFC can also fall back to a CPU-backed DALI reader (`HAILO_DALI_FORCE_CPU=1`), which is much faster than the TF
path. It isn't in the stock 3.33.1 wheel.

## Reproducing

**Requirements:**
- Hailo DFC 3.33.1 (`hailo_sdk_client`, `hailo_model_optimization`), proprietary and not included.
- PyTorch with sm_52 support. Official CUDA 12.8+ wheels dropped Maxwell; a source build was used.
- Ultralytics.

Scripts use the absolute paths of the original machine. Adjust `ROOT` in `hailoqat/*.py` and the paths in
`tools/*.sh` / `tools/env.sh`.

1. Level-0 quantization with 16-bit heads:
   `b2_level0.py PARSED.har L0a16.har 896 896 1024 configs/alls_a16heads.txt` (DFC env)
2. Ground truth plus layer parameters:
   `har_pipeline.sh L0a16.har L0a16` (`.hn` export, GPU capture, exactness check, eval)
3. Teacher cache:
   `teacher_cache.py OUT --shard k/n` (one process per GPU)
4. QAT, recommended recipe:
   `GPUS="0 1 2" ./exp.sh q3a --epochs 1 --eval_every 1000 --lr_k 6e-4 --lr_b 3e-2 --ovf 0 --k_radius 4 --b_radius 256`
5. Write-back, round-trip check and compile:
   1. `ints2npz.py`
   2. `writeback.py`
   3. `har_pipeline.sh`: must reproduce the training mAP to every digit
   4. `compile_hef.py`

Speed on 4x Tesla M40:
- Training: 13.7 img/s exact QAT, ~22 min per 18.4k-page epoch. The plain-PyTorch reference ran 1.31 img/s on
  2 GPUs.
- Full-val exact eval: 2.5 min (was 23 min).
