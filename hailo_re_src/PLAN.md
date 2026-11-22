# hailoqat — exact-numerics QAT for Hailo-8 (plan)

**TARGET (user decision 2026-10-22): YOLO26m `yolo26m_v14_sec.pt`** (sec_1/2/3 merged into `sec`, fnote gone;
slots 11/12 unused). Checkpoint val 0.901/0.851; **this harness's FP reference on `data_v14sec/val` (916 real
pages): mAP50 0.893 / mAP50-95 0.839** — v18 val labels were rebuilt 2 days after that run. All quantized
numbers compare to 0.893/0.839.

Data (`hailo_re/data_v14sec/`, built by `build_sec_sets.py`, NAME remap sec_* → sec, images symlinked):
train 18,401 pages (v18_all HCMUS + tex synth; DocLayNet excluded) / val 916 real / bench 2,770
(`yolo26m_bench_v1_14`, held out — final report only, never train). Pages are pre-padded 896×896 (black).

YOLO26m in DFC 3.33.1: parses (`parse_y26.py`, cut at the 6 one2one head convs; top-k runs on host).
New HN types vs YOLOv8m: matmul 4, softmax 2 (C2PSA attention), dw 8, standalone activation 2,
feature_splitter/shortcut/format_conversion (data movement). After level-0 the DFC **decomposes softmax** into
matmul → reduce_max → ew_sub (exp APU) → reduce_sum (inv_pos APU) → ew_mult → matmul, all a8_w8_a8 with
bit-exact emulation (the LUT SoftmaxOp is not used). Attention probabilities are 8-bit: a QAT focus point.

GPUs: M40 0–1 (2–3 belong to another job; check `nvidia-smi` before every launch). Chips: hailo0, hailo1.
TF-GPU fixed (env.sh preloads system cuDNN 9.5.1; the env's pip cuDNN 9.1 broke it). DFC runs on GPU.
torchvision::nms is broken in every env → evaluate.py uses Ultralytics' pure-torch NMS with no time cap.

---
Original YOLOv8m line (superseded, kept for the numerics findings): `best.pt` 19 classes; its March
dataset no longer exists (all datasets rebuilt June–July), scores 0.50 mAP50 on today's yolo_dataset_clean.

## Why this beats the DFC path

| | DFC (AdaRound + QFT) | hailoqat |
|---|---|---|
| Numerics trained against | `SDK_QUANTIZED` (≠ chip, score Δ up to 0.13) | bit-exact chip arithmetic (104/104 layers verified) |
| Signal | per-layer L2 to FP outputs (AdaRound), FP distillation (QFT) | real labels (YOLO loss) + FP teacher distillation |
| Cost | 20 h 52 min AdaRound on 1 GPU | level-0 calibration (minutes) + QAT on 2 GPUs |
| Rounding seen in training | none of: per-product shift, 16b wrap, APU truncation | all of them |

## Established (2026-10-22)

- `SDK_BIT_EXACT` == chip (scores identical; box Δ ≤ 1.4e-3, residual unattributed).
- `intops.py` + `graph.py`: 91 conv + 12 ew_add + normalization, chained from pixels → 6 heads, all EXACT.
- Write-back API: acceleras `export_acceleras()` + `export_hw_params()` → `load_params(HAILO_OPTIMIZED/TRANSLATED)`
  (what `ClientRunner.set_keras_model` does).

## Phases and gates

### A. Foundations
- **A1 Param import from HAR** (no capture): build the acceleras model of any quantized HAR and read
  ints/shifts/APU tables from the op objects. Gate: graph.py EXACT using imported params.
- **A2 Write-back round trip**: torch ints → acceleras op natives (`kernel = int·scale`, bias) → export →
  new HAR → `SDK_BIT_EXACT` equals torch on 4 images. Perturb kernels first to prove the write-back is real.
- **A3 Op test matrix**: every op × precision mode × activation × shift × stride/pad/groups/dilation used by
  candidate configs, synthetic adversarial inputs (wrap, ties, clip edges, PWL boundaries) vs `SDK_BIT_EXACT`.
- **A4 Chip ground truth**: tiny 1–3-layer HEFs per matrix cell, raw outputs read on hailo0 (no NMS);
  explains or removes the 1.4e-3 box residual.

### B. Fast initial quantization (minutes)
- **B0 Preprocessing parity (found 2026-10-22)**: pages are portrait ~1754×1241; Ultralytics trained/validated
  with letterbox (gray 114) at 896, but `hailo_calib_*`/`hailo_run_val_896.py` stretch to 896×896 and
  `hailo_infer_pdf.py` letterboxes at 640. One shared letterbox for calibration, QAT, eval and deployment.
  **Rect input 896×640** (H×W, = Ultralytics rect val shape) is a candidate: 29% fewer pixels, no pad waste.
- B1 `yolov8m_layout_finetuned_896.har` is already the parse of `best.onnx` (exported 11:28, parsed 11:29).
  Rect candidate needs `best.pt` → ONNX at (896, 640) → parse.
- B2 `optimization_level=0`, 1024 train-image calibration, no AdaRound/QFT. Candidates:
  (a) defaults, (b) 16-bit head outputs (`a8_w8_a16` on conv57/58/70/71/82/83), (c) per-channel weights
  if the Hailo-8 compiler accepts `quantization_groups>1` (A3/A4 decides).
- Gate: exact-integer val mAP of each candidate (this is the PTQ floor QAT starts from).

### C. QAT in PyTorch (the core)
- **Student**: `.hn` graph with latent float kernels/biases; forward = exact elementwise stages (bias wrap,
  APU PWL, clips, final shift) with STE; MAC per-product rounding by its mean (fast fp32 path) during
  training, exact path (fp32 split + residue-table conv) for every evaluation.
- **Constraints**: kernels on the int8 grid within the range that keeps `pre_acc_shift` valid;
  accumulator-wrap penalty (Hailo's own `wraparound loss` idea); concat inputs share one scale; biases in the
  representable bias-decomposition grid.
- **Loss**: Ultralytics v8 detection loss on dequantized heads + KD from the FP teacher (best.pt):
  cls logits (BCE/KL, T≈2) + DFL distributions (KL) + neck-feature L2 at the 3 strides.
- **Recipe**: AdamW lr 1e-5…1e-4 cosine, EMA, DDP/NCCL on 2 GPUs, Ultralytics augmentation with mosaic
  off for the last epoch, 2–5 epochs (or a 20k-image subset first to measure slope).
  Oscillation dampening + iterative freezing ([Nagel et al. 2022](https://arxiv.org/pdf/2203.11086)) in the
  last epoch.
- **C2 (if C1 plateaus)**: learnable activation ranges → port APU/shift `create_hw_params` so scales can move.
- Gate per epoch: exact-integer mAP on a fixed 1,000-image val subset; full val at the end.

### D. Export and on-chip validation
- Write back (A2) → HAR → `SDK_BIT_EXACT` == torch → compile → HEF.
- Full val set on hailo0+hailo1 with the deployed NMS; must match torch-exact mAP.
- Report both standard mAP (conf 0.001) and as-deployed (HEF score 0.2 / IoU 0.6).

### E. Efficiency
- Current HEF: 3 contexts, compiler estimate 175.7 FPS / 30.6 ms. Try `compiler_optimization_level=max`,
  4-bit weights on large low-sensitivity layers (QAT recovers them — hailoqat sees the exact a8_w4 math)
  to reach fewer contexts; measure with `hailortcli benchmark`. Accuracy gate from C stays.

## Risks
- Bias-decomposition and APU re-derivation are the least explored parts (C constraints, C2).
- M40 throughput: YOLOv8m at 896 is slow to train; measure img/s in C before committing epochs.
- Compiler acceptance of changed kernels/precisions is only proven by compiling (D, A4).

## Status log

### 2026-10-22 (YOLO26m v14_sec)
- **Exactness:** `graph26.py` reproduces SDK_BIT_EXACT on all 155 checked layers chained from raw pixels (conv 117,
  dw/dw-add 8, ew_add 16, ew_sub 2, matmul 4, reduce_sum 2, ew_mult 2, activation 2, normalization 1).
  `qatnet.py` training forward == exact path on all 189 layers (0 differences).
- **Write-back verified:** `writeback.py` (kernels + zero-point-aware biases through acceleras export) → new HAR
  whose SDK_BIT_EXACT equals hailoqat with the same ints on 144/144 layers (132 changed by the test edit).
- **PTQ floor (level 0, exact int, 916 val):** mAP50 **0.855** / mAP50-95 **0.734** vs FP 0.893 / 0.839.
- **Accumulator wrap in level 0:** conv49 wraps on 1.6% of outputs (|acc| up to 47k), conv95/conv43 too →
  QAT overflow penalty; alternative experiment: larger pre_acc_shift for those layers.
- Pitfalls fixed on the way (each silently wrong, none raised an error):
  1. per-product rounding cannot be approximated by sum-then-round (≥6 LSB drift by layer 3);
  2. a single fp32 conv over 512 taps of 255·127 is NOT exact (order-dependent) → nibble-split `nib_conv`;
  3. `s + (e - s).detach()` is not bit-exact in fp32 → `e.detach() + (s - s.detach())`;
  4. conv-add factor must be read after `enforce_encoding()` (integer), not from a fresh model (float);
  5. Hailo's HSim.so ships kernels only as relocatable sm_61 cubins + sm_61 PTX (its sm_52 image is an empty
     device-link stub) → `hsim_sm52/`: PTX retargeted to sm_52, ptxas+nvlink+fatbinary, swapped in by an LD_PRELOAD
     shim at __cudaRegisterFatBinary. Second HSim defect (any GPU): all launches on stream 0 while TF uses its own
     non-blocking stream → racy reads (random garbage in conv2's tail rows); shim fences HSim's cudaLaunchKernel with
     cudaDeviceSynchronize. Result: SDK_BIT_EXACT on GPU == CPU on 882/882 tensors (2 runs), 58 s vs 8 min 22 s.
- Throughput: QAT 1.29 img/s on 2×M40 (bs 2/GPU, 13 GB), exact eval 0.31 img/s/GPU.

### 2026-10-22 speed rounds (all 4 GPUs)
- `hailoqat/qatfast.py` + `hailoqat/hq_ext/hq.cu`: fused sm_52 kernels, bit-exact to graph26 (567/567 layer checks on 6 pages x 2 trained int sets; full-val L0 mAP identical to every digit), gradients equal to qatnet to 1.5e-8. `qatnet.py` stays as the reference.
- Train step bs2/GPU 2.78 -> 0.531 s; 4-GPU DDP with `--teacher_cache ../teacher_v14sec` 13.7 img/s (epoch ~22 min). Eval: `evaluate.py --impl qatfast` (default), `evalgpu.sh` shards over any GPU list.
- Harness: `bench_qat.py exact|grads|time|prof`, `bench_tail.py` (kernel microbench). Per-round backups: `hq_ext/hq.cu.r1..r6`, `qatfast.py.r5`.
- Open: training recipe (pilot and r1 below L0; r1 diverged at step ~600).
