# Hailo-8 port of the page pipeline — handoff (2026-11-13, 12:10)

Owner's goal: convert every model of the page pipeline to Hailo-8 with the custom converter (`hailo8-exact-qat`) at the
**maximum accuracy**, using 16-bit wherever it helps. The models are turn, straighten and layout detect (+ order head).
Headline metric = the gap from the FP model to the chip.

## 0. Read first

- Converter repo: `/mnt/storage/lib_project/hailo8-exact-qat/` — `README.md`, `docs/USAGE.md` (the full pipeline),
  `docs/PRECISION_RULES.md` (how to pick 16-bit layers), `docs/COMPILER_RULES.md`, `docs/HAILO8_ISSUES.md`.
- Working tree: `/mnt/storage/lib_project/models/Hailo_model/hailo_re` (`hailoqat/` = exact emulator + QAT: `graph26.py`,
  `qatfast.py`, `qchain.py`, `atomic.py`, `capture.py`, `writeback.py`; `zoo/ops/` = op probes, compat front end,
  `host_split.py` StagedRunner).
- Closest worked example of a NON-YOLO port: `Hailo_model/gridnet_hailo/` (`export_onnx.py`, `parse.py`, `quantize.py`,
  `eval_hailo.py`, `run_q.sh`). Plain level-0 int8 was already ≈ FP there.
- Memory: `hailo-index.md`, `hailo-precision-rules.md`, `hailo-qat-goal-max-accuracy.md`, `hailo-op-support-probe.md`,
  `gridnet-hailo-port.md`.
- DFC env: `. /mnt/storage/lib_project/hailo8-exact-qat/tools/env.sh` then `$HPY` (CPU unless CUDA_VISIBLE_DEVICES set).
  Torch inside hailo_env needs the LD_PRELOAD of the env's libnccl (see `gridnet_hailo/run_q.sh`).
- Chips: two Hailo-8 on uslib (`hailortcli scan`: 0000:03:00.0, 0000:02:00.0).

## 1. Models to port (FP sources)

| Step | Checkpoint | Class / code | Serving input | Notes |
|---|---|---|---|---|
| Turn | `layout_teacher/train/runs/orient2/ori_5000.pt` | `train_orient.OriNet` (ResNet-18-shaped, 2.8M) | page long side 896, other side ×32, `x/255 - .5`; `dense()`: 7×7 window mean → 1×1 fc → softmax; rule `decide_ll(t=1.0)` | Variable shape in serving → chip needs a FIXED canvas (exported 896×896). Must re-check the decision on the orient eval / MDPBench turned pages (30/120 turned). |
| Straighten | `layout_teacher/models_flat/flat2.pt` | `Hailo_model/uvdoc/src/model.UVDocnet(num_filter=32, kernel_size=5)`, output `[0]` = 45×31×2 mesh | RGB 0..1, 712×488 (H×W); remap on host (`docparse/condition/flatten_worker.py`) | Many DILATED convs (d = 2,3,4,5,6,7,8,12,18): Hailo silently miscomputes some widths → compat front end MUST run (`dil_pad`). Metric: mesh error % of frame (`train_flat.py --eval`), flat T1 scan move (`flat_gate.py`), MDPBench e2e. |
| Layout | r3b today: `layout_teacher/train/runs/r3/ema_24000.pt`; successor r5 (see §3) | PP-DocLayoutV3 (RT-DETR family, HF `layout_teacher/weights/ppdl3_hf`, `train_ppdl3.build_model`) | 896×896 black letterbox, `/255` | Deformable-attention decoder uses GridSample + TopK query selection = HOST ops. Plan: chip = backbone + hybrid encoder; host = query select + decoder + heads (StagedRunner), or measure all-host decoder cost. |
| Order head | `r3b_ord1.ppdlft` (ord1, `train_order.py`) | order logits on decoder queries | — | Host with the decoder. |

## 2. Done so far (`Hailo_model/preproc_hailo/`)

- `export_onnx.py` → `orient2_chip.onnx` (stem+body, 896×896, input RGB 0..1, `-.5` inside) and `flat2_chip.onnx`
  (712×488 → mesh). Exported with opset 13, torch in `layout_teacher/venv_hf`.
- `orient2_compat.onnx` was written by `tools/ops/onnx_hailo_compat.py`; the run was interrupted. Re-run the compat
  step for both and check the printed float error (must be 0 or ~1e-7).
- `parse.py` (copy of gridnet's). Nothing is parsed or quantized yet.

## 3. Suggested plan (small first, as the converter docs say)

1. orient2: compat → parse → level 0 with all layers `a16_w16` (it is small) → HSim/graph26 bit-exact check → the
   decision on the orient eval set equals FP? → compile → chip check. Calibration: real pages from the training
   manifest (`photo_data.manifest()`), photos + scans, NEVER MDPBench / bench_v1 / holdout authors.
2. flat2: same; compare mesh error FP vs chip on held-out synthetic photos + UVDoc val (393 captures,
   `layout_teacher/data_ext/uvdoc/UVDoc_final`, `train_flat.uv_list`) + flat-scan move. Watch the dilated convs.
3. Layout: export the chip part (backbone + encoder) at 896; sensitivity scan → 16-bit layers by box-path rule; exact
   QAT (qatfast) if level 0 + 16-bit leaves a gap. Gate on box AP / AP75 (`train_ppdl3.evaluate` sets) and MDPBench.
4. Every model: report FP vs chip gap, FPS (`hailortcli benchmark X.hef -t 5`), contexts.

## 4. Other work running when this was written

- **3090 migration** (owner: train only on the 3090 `user@gpu-box`; password from the owner, never write it down).
  `~/gridnet/data` (128 GB) was copied to `/mnt/storage/lib_project/models/gridnet_3090_data` (11:47). A chain
  (`scratchpad/setup_3090.sh`, log `setup_3090.log` in this session's scratchpad) then checksum-verifies the copy,
  deletes the remote copy only if all checks match (owner-approved), mirrors the training data to the same absolute
  paths on the 3090, and builds `layout_teacher/venv3090` (torch 2.7.1 cu126, transformers 5.22.2).
  **Result 12:13: the chain STOPPED before deleting anything.** Checksum pass = 0 differing files, 3,081 files on both
  sides, but my size check failed: (a) the remote count and bytes came back on two lines, so the string compare broke;
  (b) `du -sb` differs by 12,288 bytes, which looks like directory-entry sizes on two filesystems, not file data.
  Nothing was deleted on the 3090, no data was pushed, no venv was built. Next: compare the summed FILE sizes
  (`find -type f -printf '%s\n' | awk '{s+=$1} END {print s}'`) on both sides; if they are equal, delete
  `~/gridnet/data` on the 3090 (owner-approved) and run the push + venv steps of `setup_3090.sh` by hand.
- `train_ppdl3.py` now has `LAYOUT_AMP=fp16|bf16` (tensor cores; unset = fp32 as before). Not yet probed on the 3090.
- **r5 layout fine-tune** (r3b on straightened photos, `launch_r5.sh`, flat2): at step 3000 photo box AP +7..+13
  points, clean −1..−3. A watcher stops it on the M40s right after its step-6000 checkpoint is saved (~12:30), to free
  the M40s. Then resume on the 3090 from `runs/r5/latest.pt` (step 6000) with the same global batch 12
  (`--batch 12`, single GPU, `LAYOUT_AMP=fp16`, `LAYOUT_FLAT_CKPT=.../models_flat/flat2.pt`) and copy the results back.
  Check that `nvidia-smi` shows 0 MiB on cuda:0,1 after the stop (orphaned workers).
- Notes for this work: `layout_teacher/RUNS.md`, design doc §8f, `/mnt/backup/Dolphin_training/github_release_moe_20261023/layout_preprocess/`.
- Rules: never touch cuda:2,3 (llama-server) or live docparse :9123; never train or calibrate on MDPBench, yolo26m_bench_v1
  or the holdout authors; back up before editing; kill exact PIDs, never `pkill -f`.
