# Usage guide

This guide covers turning a trained Ultralytics YOLO26 checkpoint into a Hailo-8 HEF whose integers were trained on the
chip's exact arithmetic. There are two recipes:

| Mode | Precision | When to use | Training cost (4x M40) | Chip cost |
|---|---|---|---|---|
| **Fast** | int8 everywhere; 16-bit outputs on the 6 one2one head convs | Default. Best mAP per FPS | ~1 h | Baseline |
| **Accurate** | Fast mode, plus `a16_w16` on the 8 layers with the highest rounding-noise score | When every 0.01 mAP50-95 matters | ~2-3 h | Depends on the model: one extra context on YOLO26n (-30% FPS). See `docs/REPORT.md` for YOLO26m |

Both recipes share one pipeline. The accurate mode adds three steps: the sensitivity scan, the 16-bit model script and
the bias-lattice annotation.

```
 .pt ──export──> .onnx ──compat front end (silent-wrong op rewrites)──> parse ──> DFC level 0 (+alls) ──> HAR
                                                                                         │
            ┌──────────── capture (HSim bit-exact, 1 page) <───────────────────────────┘
            │  per-layer integers, scales, APU tables, ground-truth activations
            v
  exactness check (graph26: every layer bit-identical) ──> [16-bit: bias_lattice]
            │
            v
  exact QAT (qatfast, DDP) ──> checkpoints ──> bench_sel eval ──> pick
            │
            v
  ints2npz ──> writeback (HAR) ──> round-trip capture + eval ──> compile (stock, or our hcc) ──> guard ──> HEF ──> chip check
```

## How this differs from the stock Hailo flow

| Stage | Stock DFC 3.33.1 | This repo |
|---|---|---|
| ONNX → HAR | `translate_onnx_model` as is. Some ops are parsed into something the chip computes wrongly with no warning (GroupNorm(1), kernel = stride ConvTranspose, ConvTranspose with begin padding, dilated convs of some widths) | `tools/ops/onnx_hailo_compat.py` rewrites those ops into float-identical forms that are exact on the chip (verified on silicon, `docs/COMPILER_RULES.md`, RULES 28) |
| Quantization | Levels 0-4: calibration, bias correction, QFT, AdaRound. They optimize `SDK_QUANTIZED`, a simulation that skips the chip's per-product rounding and 16-bit accumulator wrap | Level 0 only, then **exact QAT**: training runs the chip's integer arithmetic bit-for-bit (`hailoqat/qatfast.py`), so it optimizes what the chip computes. YOLO26m: +0.020 mAP50-95 over level 2 |
| 16-bit layers | Chosen by hand | Chosen by measured sensitivity: box path first, per-layer weight-scale spread (`docs/PRECISION_RULES.md`) |
| Emulator | `SDK_BIT_EXACT` / HSim; differs from the chip on the APU edge segments, UINT16 inputs, dynamic-matmul operand signedness | `hailoqat/graph26.py`: matches the chip (908/916 pages bit-identical; mAP identical to 4 decimals) |
| Compile | One `compile()` call. Some lowerings it picks are silently wrong on the chip (DECONV for k = s, INPUT_FEATURES split of dynamic matmuls) | Optional `turbo/hcc.py`: per-context parallel compile + chip benchmark of candidates; blocks those lowerings and recompiles; ships the fastest candidate that passes `hailo8_guard.py`. Same HEF accuracy (bit-exact vs stock on yolo26m) |
| Check | none | `turbo/hailo8_guard.py M.hef C.har [auto.alls] [Q.har]`: flags every known silent-wrong pattern |

**Data.** QAT needs a training set: the images (for the teacher outputs and the integer forward) and their labels
(the detection loss term). It is the same data the FP model was trained on (`data_v14sec/train`, 18.4k pages); no new
labelling is needed. Calibration for level 0 uses 1,024 of those pages. The benchmark (`data_v14sec/bench`) is only
for selection (`bench_sel`) and reporting (`bench_test`), never for training or calibration.

## Paths (original machine)

| What | Path |
|---|---|
| Release repo (code, docs, configs) | `/mnt/storage/lib_project/hailo8-exact-qat` |
| Working tree (data, HARs, captures, runs) | `/mnt/storage/lib_project/models/Hailo_model/hailo_re` (= `R`) |
| FP checkpoint | `R/models/v14sec/best.pt` |
| Train / val / bench pages | `R/data_v14sec/{train,val,bench}/images` (+ `labels`), split `R/data_v14sec/bench_split.json` |
| Parsed HAR | `R/../yolo26m_v14sec_parsed.har` |
| Fast-mode model (int8 + 16-bit heads, QAT) | `R/../yolo26m_v14sec_q3a_s2000.har`, HN `R/q3a_s2000_har.hn`, capture `R/capture_q3a_s2000_har` |
| Accurate-mode model (top-8 16-bit, QAT) | `R/mopt/m_w16top8_qat.har`, `.hef`; HN `R/m_w16top8_qat.hn`, capture `R/capture_m_w16top8_qat` |
| Box-path 16-bit run (in progress) | `R/mopt/w16box.sh`, log `R/mopt/w16box.log`, run `R/runs/m_w16box_2ep`, HEF `R/mopt/m_w16box_qat.hef` |
| Teacher cache | `R/teacher_v14sec` |
| Every eval | `R/out/eval.jsonl` (tags `mb_*` = full bench) |
| Our compiler | `R/turbo/hcc.py`, `part_fix.py`, `compile_sub.py`, `hailo8_guard.py` (copies in `tools/fixes/`) |
| ONNX front end | `R/zoo/ops/onnx_hailo_compat.py` + `compat4.py` (copies in `tools/ops/`) |
| Chip-bug repro kit | `hailo8_repro/` (`models.py`, `run.py`, `check.py`) |
| Rules | `docs/PRECISION_RULES.md` (16-bit), `docs/COMPILER_RULES.md`, `R/probe/RULES.md` (compiler, sections 1-28) |

## 0. Environment

You need two Python environments:

1. **DFC env** (`hailo_env`). It needs Hailo Dataflow Compiler 3.33.1 (`hailo_sdk_client`, `hailo_model_optimization`),
   HailoRT 4.23 and TensorFlow. Source `tools/env.sh` before any DFC call. It sets:
   - `HPY`, the DFC interpreter;
   - a writable `VIRTUAL_ENV`;
   - the system cuDNN preload, because TensorFlow otherwise loads the older pip cuDNN and fails with "No DNN in stream
     executor";
   - the `hsim_sm52` shim, so HSim runs on Maxwell;
   - `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`.
2. **Torch env** (`general`). It needs PyTorch with your GPU's architecture and Ultralytics. Run it with
   `PYTHONNOUSERSITE=1` so a user-site torchvision cannot shadow the env's build (that shadowing breaks `torchvision::nms`).
   - `hailoqat/hq_ext/hq.cu` JIT-compiles on first import (about 1 min).
   - On Maxwell, set `TORCH_CUDA_ARCH_LIST=5.2` to stop a rebuild for every arch.

All scripts use the absolute paths of the original machine. Edit these before running anything:
- `ROOT` in `hailoqat/train_qat.py` and `hailoqat/evaluate.py`;
- `R=` and `P=` in `tools/*.sh` and `recipes/*.sh`.

Rules that keep the numbers honest:
- **Never train or calibrate on the benchmark.** The benchmark is split by author into `bench_sel`, used to choose
  checkpoints, and `bench_test`, used to report (`configs/bench_split.json`).
- Do not select on val. Val shares authors with train, so QAT gains on val did not transfer: +0.035 on val, +0.000 on
  bench for the all-16 nano.
- Remap labels by class **name**, not index.
- Check `nvidia-smi` first. A GPU shared with another job silently corrupts the timings, and an OOM retry corrupts
  the outputs.

## 1. Export and parse

```bash
# torch env
python -c "from ultralytics import YOLO; YOLO('best.pt').export(format='onnx', imgsz=896, opset=13, simplify=True, dynamic=False)"
# DFC env
. tools/env.sh
CUDA_VISIBLE_DEVICES=99 $HPY tools/parse_y26.py best.onnx model_parsed.har
```

`parse_y26.py` cuts the graph before the Detect decode. The six one2one head convolutions (`one2one_cv2/cv3 .2`)
become the HEF outputs, and decoding happens on the host (`evaluate.py`, `chip_map.py`).

Find the head layer names, needed for the 16-bit-head line:

```bash
CUDA_VISIBLE_DEVICES=99 $HPY - <<'PY'
from hailo_sdk_client import ClientRunner
hn = ClientRunner(har="model_parsed.har").get_hn_dict()
print([k.split("/")[-1] for k, v in hn["layers"].items()
       if any("one2one_cv" in o and o.endswith(".2/Conv") for o in v.get("original_names", []))])
PY
```

### Front end: rewrite ops the chip computes wrongly (always)

```bash
# DFC env, before parsing; prints the rewrites and the float check (must be 0.00e+00 or ~1e-7)
$HPY tools/ops/onnx_hailo_compat.py best.onnx best_compat.onnx
```

Rewrites the silent-wrong ops (GroupNorm(1), ConvTranspose kernel = stride and non-SAME pads, dilated convs whose
width the compiler would cut into a bad piece, K/V attention biases, ...). Each is float-identical; the run checks
it. A model without those ops comes out unchanged. YOLO26 has none of them.

### Optional: graph rewrites (experimental)

`tools/onnx_pe_spatial.py` and `tools/onnx_concat_pushdown.py` are exact ONNX transforms: onnxruntime outputs match,
and the scripts assert it.
- They were written to cut PCIe bytes at context cuts.
- On YOLO26m they changed neither the cuts nor the DDR portals, and gave 0% FPS (REPORT section 4.4). On YOLO26n the
  pe rewrite cost 5%.
- Before trying any rewrite, run `hefre/hef_dump.py` on a baseline HEF to see which tensors actually cross PCIe.

## 2. Quantize at DFC level 0 with the model script

Fast mode, `configs/alls_a16heads.txt`:

```
quantization_param([conv71, conv74, conv87, conv90, conv101, conv104], precision_mode=a8_w8_a16)
```

Accurate mode: add one line from the sensitivity scan in step 4a.

```
quantization_param([conv70, conv86, conv100, conv97, conv69, conv_feature_splitter11_2, conv67, conv60], precision_mode=a16_w16)
```

> **Trap:** `a8_w8_a16` on a layer that is *not* a graph output makes its consumer use a 16-bit *wrapping*
> accumulator. On YOLO26n that collapsed mAP to 0.02, and HSim agrees, so it is real hardware behaviour. Use
> `a16_w16` for intermediate layers. The DFC then inserts `precision_change` layers and uses 32-bit accumulators.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=dfc_patched TRAIN_GLOB="data/train/images/*" OPT_LEVEL=0 \
  $HPY tools/b2_levelN.py model_parsed.har model_L0.har 896 896 1024 alls.txt
```

About 17 min for YOLO26m (1,024 calibration pages). Level 0 is the right start because QAT replaces everything the
higher levels do. Levels 2 and 4 are only useful as baselines (see the report).

## 3. Capture and check exactness

```bash
EVAL_GPUS="0 1" tools/har_pipeline.sh model_L0.har TAG 0
```

The script does four things:
1. Exports `TAG.hn`.
2. Runs Hailo's bit-exact emulator (HSim, `SDK_BIT_EXACT`) on one page with every op hooked (`hailoqat/capture.py`).
   It writes `capture_TAG/`: per-layer integer kernels, biases, shifts, zero points, APU tables and the emulator's
   integer activations.
3. Runs the float64 reference executor `graph26.py` from raw pixels and compares every layer.
   - **It must print `N/N checked layers exact`.**
   - Any mismatch means the executor does not model an op yet, and QAT would optimize the wrong function.
4. Runs a full exact val eval with `qatfast` and appends `{"tag": TAG, ...}` to `out/eval.jsonl`. This is the PTQ
   floor.

**Accurate mode only:** annotate the 16-bit bias lattice:

```bash
CUDA_VISIBLE_DEVICES=99 $HPY hailoqat/bias_lattice.py model_L0.har capture_TAG
```

An `a16_w16` bias is stored as a 16-bit integer `i` with a per-layer `bias_factor` F and `bias_feed_repeat` R. The
accumulator bias is `f32(f32(F*i)*R)`, with both products rounded in float32 (bit-exact to HSim). The script checks
this identity on every channel. QAT then trains `i`, so the result stays storable. `qatfast.WideConv` refuses to
build without these keys.

## 4. Exact QAT

### 4a. Choose the 16-bit layers (accurate mode)

```bash
python hailoqat/sens.py --hn TAG.hn --cap capture_TAG --pt best.pt --teacher_cache teacher/ --imgsz 896 --n 256 --out sens.json
```

The score of a layer is `E[g²]/12` summed over its output elements, where `g` is the loss gradient with respect to
the layer's integer output.
- This is the first-order loss increase from int8 rounding noise.
- 16-bit outputs remove almost all of that noise, so the score is also the expected gain.
- Take the top 8 `conv`/`dw` layers that are not heads.
- The ranking barely changes after QAT, so a level-0 scan is enough.
- **This score misses two things; apply `docs/PRECISION_RULES.md` on top:**
  - the whole box-regression path in `a16_w16`, including the box output convs (8-bit weights with one scale per
    layer cost AP at IoU 0.90/0.95) and the neck layers that feed it;
  - check AP per IoU threshold (`hailoqat/precision/gap_by_iou.py`), not only mAP.
- For maximum accuracy, do NOT skip a layer because its output crosses a context cut. That only saves PCIe traffic:
  a 16-bit tensor doubles the cut's bytes, so it costs FPS, not accuracy.

### 4b. Teacher cache (once per dataset and image size)

```bash
for k in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$k python hailoqat/teacher_cache.py teacher/ --shard $k/4 --imgsz 896 & done; wait
```

This stores the FP model's head outputs for every training page as a memmap: 13 GB for 18.4k pages at 896.

> **Trap:** some filenames contain spaces. `files.txt` is read with `splitlines()`.

### 4c. Train

```bash
HN=TAG.hn CAP=capture_TAG GPUS="0 1 2" PORT=29531 EVAL_ALL=1 tools/exp2.sh RUN \
  --epochs 1 --eval_every 1000 --lr_k 6e-4 --lr_b 3e-2 --ovf 0 --k_radius 4 --b_radius 256
```

| Flag | Meaning | Recommended |
|---|---|---|
| `--lr_k`, `--lr_b` | Adam LR on latent kernel / bias integers | `6e-4`, `3e-2` |
| `--k_radius`, `--b_radius` | Latent integers stay within ±radius of the level-0 integers | 4, 256. Without a bound, noise-gradient weights drift and accuracy falls below level 0 |
| `--epochs` | 1 for int8; 2-3 for 16-bit, picked on bench_sel | |
| `--ovf` | Accumulator-overflow penalty | 0 (neutral) |
| `--apu_train` | Train the activation tables | Off (destructive at useful LRs) |
| `--imgsz` | Must equal the HN input size | 896 / 640 |
| `--bs` | Per-GPU batch size | 2 at 896, 8 at 640 |

The loss is the Ultralytics one2one detection loss on ground truth plus KD (BCE to the teacher's class
probabilities, confidence-weighted L1 on boxes). The forward pass is the chip's integer arithmetic:
- per-product banker's-rounded pre-accumulator shift;
- 16-bit wrapping accumulators (32-bit for `a16_w16`);
- integer bias;
- APU piecewise-linear tables.

The backward pass is a straight-through estimator.

Checkpoints `runs/RUN/ints_stepN.pt` hold the trained integers. The exact val eval of each is appended to
`out/eval.jsonl`.

### 4d. Select on bench_sel

```bash
for c in runs/RUN/ints_step*.pt; do s=${c##*step}; s=${s%.pt}
  VAL_DIR=data/bench/images tools/evalgpu.sh mb_RUN_s$s hailo TAG.hn capture_TAG --pt best.pt --bs 4 --ints $c; done
cd hailoqat && VAL_DIR=../data/bench/images python split_map.py mb_RUN_s1000 mb_RUN_s2000 ...
```

`split_map.py` prints mAP50/mAP50-95 on `bench_sel` and `bench_test` from one saved eval. Choose on `bench_sel`;
report `bench_test`.

To compare two models, run `hailoqat/boot_ci.py TAG_A TAG_B`. It gives a paired image bootstrap, and its CI of the
difference is what decides "better".

## 5. Write back, round-trip, compile

```bash
python hailoqat/ints2npz.py runs/RUN/ints_stepN.pt capture_TAG ints.npz                         # torch env
CUDA_VISIBLE_DEVICES=99 $HPY hailoqat/writeback.py model_L0.har ints.npz model_qat.har            # DFC env
EVAL_GPUS="0 1" tools/har_pipeline.sh model_qat.har TAG_rt 0                                      # must reproduce
CUDA_VISIBLE_DEVICES=99 $HPY tools/compile_hef.py model_qat.har model_qat.hef [configs/alls_perf_max.txt]
```

What `writeback.py` does:
- Loads the integers as TRANSLATED params.
- Exports.
- Compares the result with `get_params_translated()`.
- Rewrites biases whose zero point the export recomputed, then exports again.

Expected round-trip result:
- **int8 layers** round-trip exactly: the round-trip eval equals the training eval to every digit.
- **`a16_w16` layers** cannot be pinned exactly. The DFC re-quantizes 16-bit kernels (stored unsigned with a zero
  point) and re-decomposes biases at export. Expect a few layers to differ by a few LSB of 32767 and the mAP to move
  by about ±0.003.
- **Always report the round-tripped HAR's eval, not the training eval.**

Or compile with our compiler (parallel per-context search, silent-wrong lowerings blocked, chip-benchmarked; needs
a partition `PART/c{i}.layers.json` from `turbo/part_from_alls.py` or `part_gen.py`):

```bash
$HPY turbo/part_fix.py model_qat.har PART                       # legalize the partition + per-context hints
$HPY turbo/hcc.py model_qat.har PART WORK --batches 1,4 --ref model_qat.hef   # -> WORK/best/full.hef, WORK/hcc.json
```

Either way, check the HEF before trusting it:

```bash
$HPY turbo/hailo8_guard.py model_qat.hef model_qat_compiled.har "" model_qat.har    # exit 1 = known silent-wrong
```

Compile:
- Default compile takes 1.5-2 h for YOLO26m on CPU.
- `alls_perf_max.txt` (resources up to 0.7, more partition search) took about 30 h for +3% FPS.

## 6. Chip check

```bash
hailortcli run model_qat.hef --device-count 2 --batch-size 4 -t 10                  # throughput
hailortcli measure-nnc-performance model_qat.hef --batch-size 4 -s 0000:02:00.0      # HW-only (no host overhead)
python tools/chip_dump.py model_qat.hef chip.npz 0                                  # raw head integers from silicon (val pages)
python hailoqat/chip_compare.py chip.npz TAG_rt.hn capture_TAG_rt                     # chip vs exact executor, per page
python hailoqat/chip_map.py chip.npz TAG_rt.hn capture_TAG_rt best.pt                 # mAP from chip outputs
python hailoqat/viz_pages.py chip.npz TAG_rt.hn capture_TAG_rt best.pt viz/ 0.4       # GT / FP / chip renders
```

Expected chip-vs-executor result:
- On YOLO26m, 908/916 val pages are bit-identical between chip and executor.
- On the other 8 the chip also differs from Hailo's own emulator, by a few elements of one layer.
- Chip mAP equals emulator mAP to 4 decimals.

Deployment settings:
- **Batch 4-5.** Batch 6 and above hits HailoRT's 64K-descriptor scatter-gather limit and FPS drops.
- **Every chip you have.** Each Hailo-8 has its own PCIe x4 link, so 2 chips give 2x for a bandwidth-bound model.

## 7. Inspecting a HEF and the firmware

```bash
python hefre/hef_dump.py model.hef --json model_hef.json
```

The script prints, per context:
- config bytes;
- the action mix;
- every tensor that leaves or enters the chip (boundary I/O, inter-context activations, DDR portals) with its bytes
  per frame.

Use it to check where the compiler cut the network and what each cut costs on PCIe.

```bash
hailortcli run model.hef --batch-size 4 collect-runtime-data --output-path rt.json   # firmware per-action timings
python fwre/fw_split.py hailo8_fw.4.23.0.bin fw/                                    # APP / cert / CORE images
$GHIDRA/support/analyzeHeadless proj p -import fw/core_cpu.bin -processor ARM:LE:32:Cortex \
  -loader BinaryLoader -loader-baseAddr 0xC0000 -postScript fwre/ghidra_scripts/CortexMDump.java
```

The firmware image is Hailo's and is not redistributed. It ships with HailoRT.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `graph26` reports non-exact layers | An op variant is not modelled. Do not train. Examples seen: 16-bit APU overflow (int64), wide conv (nibble planes), group-interleaved concat feeding a grouped matmul |
| `WideConv` asserts `bias_factor` | Run `bias_lattice.py` on the capture |
| Write-back "bias not representable" | 16-bit layer trained without the lattice; retrain with the annotated capture |
| DFC level 3/4 dies in DALI `MakeContiguous` on Maxwell | `DALI_USE_VMM=0` (set in `dfc_patched`), or `PYTHONPATH=nodali` to fall back to the TF dataset path |
| TF "No DNN in stream executor" | Source `tools/env.sh` (system cuDNN preload) |
| mAP changes with the number of eval GPUs | Fixed: `evaluate.py` breaks int8 score ties with a hash of (image, rank) |
| `pkill -f pattern` kills your own shell | Kill by PID |
| `measure-nnc-performance` errors on device count | Pass `-s <PCIe BDF>` |
| FPS drops at batch ≥ 6 | Descriptor limit; use batch 4-5 |
