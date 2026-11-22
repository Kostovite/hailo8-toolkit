# Hailo-8 conversion: one command per retrained checkpoint

```
cd /mnt/storage/lib_project/models/Hailo_model/convert
bash convert.sh RECIPE CKPT OUTDIR [GPUS=0,1,2,3] [FROM_STAGE]
```

Every stage logs to `OUTDIR/logs/<stage>.log` and leaves `OUTDIR/.done_<stage>`; a rerun resumes after the last
finished stage. `FROM_STAGE` forces a restart from that stage (e.g. `quant` after changing a precision plan).
The run stops at the first failed gate. `ALL GATES PASSED` in `logs/run.log` = the HEF ran on the real chip and met
the recipe's task metric against the torch model. Exit 0 of a single stage proves nothing by itself.

| Stage | What | Fails when |
|---|---|---|
| export | torch ckpt -> `model.onnx` (chip part only) + host contract in `host/` | exporter parity assert |
| prep | float-exact ONNX rewrites for the DFC | rewrite parity > 1e-5 (layout) / compat prints max rel diff |
| data | `calib.npy` (TRAINING data only), `eval_x.npy`, torch reference `ref_<i>.npy` | |
| parse | `parsed.har` | unsupported op |
| quant | `q.har` (+ `q.alls`, the exact model script used) | |
| emu | emulator FP + quantized on `eval_x`, sharded over GPUS | |
| gate_fp | emulator FP == torch (catches parse/rewrite bugs) | rel max > FP_TOL |
| gate_q | recipe task metric, quantized emulator vs torch | recipe threshold |
| compile | `model.hef` + `hailo8_guard` | allocator / guard |
| chip | real Hailo-8 (holds the chip `flock`) | |
| gate_chip | same task metric on chip output | recipe threshold |

Outputs are ordered by size (largest first) everywhere (emu, chip, ref), so `_0`, `_1`, ... always mean the same tensor.

## Recipes (`recipes/<name>.sh`)

| Recipe | Checkpoint (current) | Chip part / host part | Precision | Gate |
|---|---|---|---|---|
| `router` | `Dolphin/wordlid/data/grid/region_router.pt` | MLP on pooled encoder (1024) / pooling, `x / host/router_scale.npy`, overrides | all16 + 16-bit INPUT | decision agreement >= .99 per head (kind/body/lang) |
| `orient3` | `layout_teacher/staged_v2_20261114/orient3_10000.pt` | stem+body at 896 / 7x7 mean, fc, softmax, `decide_ll` | all16, rgb/255 | 0 decision differences vs torch on 2000 canvases |
| `flat3` | `.../flat3d_2500.pt` | UVDocnet mesh 712x488 / remap | all16 (+ compat `dil_pad`) | mean mesh diff vs torch <= .05 % frame |
| `gridnet` | `Dolphin/wordlid/data/grid/table_structure_final.pt` | trunk p2/p4 / every head + TableMergerV2 | all16, gray/255 | identical table structure vs torch on >= 99 % of 400 VN bench tables, exact not lower |
| `layout` | `layout_teacher/staged_v2_20261114/r5_ord5.pt` | backbone + hybrid encoder / decoder, heads, order head | mix16 (+ 8-bit upsample blocks) | box AP drop <= .005 on thesis / d4la / photo sets |

Thresholds are env overrides: `MIN_AGREE`, `MAX_DIFF`, `FLAT_TOL`, `MIN_SAME`, `MAX_DROP`, `FP_TOL`.

## Rules learned the hard way (do not undo)

* `lib/env.sh henv` PREPENDS to `LD_PRELOAD`. Replacing it drops the system cuDNN + sm52 shim -> TF "No DNN in stream executor".
* A float feature input (router) must be 16-bit AND scaled per feature on the host. One 8-bit scale over [-16.9, 3.6]
  with feature std .27 gave .87 chip agreement; per-feature scale + 16-bit input gives 1.000.
* PP-DocLayoutV3: a 16-bit bilinear Resize output does not compile (`resize3_output_reshape`); an 8-bit Resize output
  feeding a 16-bit consumer fails "precision mode is not accurate"; `a16_w16_a8` is rejected on Resize; grouped
  ConvTranspose is unsupported; ONNX edge `Pad` is silently turned into ZERO pad by the parser (use concat of slices);
  slices on the SMALL map before the upsample are skipped by the parser. The working rewrite is in `recipes/rewrite_resize.py nn`.
* Making the upsample consumers 8-bit (`ew_add22..24`, `conv109`) destroys layout AP (thesis .94 -> .60).
* Emulator scoring of 1500 layout pages is ~5 h on one M40: always shard (`GPUS=0,1,2,3`).
* Calibration never uses MDPBench, yolo26m_bench_v1, holdout authors or the eval sets the gates score.
* GridNet kind head stays on the host (DFC avgpool quantization failed); TableMergerV2 reuses the chip trunk features.

## After a retrain

1. `bash convert.sh <recipe> <new ckpt> runs/<name>_<date> 0,1,2,3`
2. Read `logs/run.log`; on a gate FAIL read that gate's log (it prints the per-set numbers).
3. Ship `model.hef` together with everything in `host/` (scales, bench indices, FP baseline).
