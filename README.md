# hailo8-toolkit

Tools for converting PyTorch / ONNX models to the Hailo-8 accelerator correctly and quickly. They were built while porting
document-layout, page-preprocessing and table-structure models. The toolkit includes:

- an exact GPU executor for Hailo-8 integer arithmetic;
- a GPU test suite;
- a one-command conversion pipeline with accuracy gates;
- compiler speed-ups;
- reverse-engineering notes on the Hailo DFC compiler and the chip.

It is not affiliated with Hailo. It contains no Hailo source code, binaries or firmware. You need your own installation of
the Hailo Dataflow Compiler (tested with DFC 3.33) and HailoRT (4.23).

## What is here

| Folder | Contents |
|---|---|
| `hailo8-exact-qat/` | `hailoqat/`: bit-exact Hailo-8 integer executor on CUDA. `graph26.py` is the float64 reference; `qatfast.py` is the fast fused-kernel executor and supports exact QAT. Prebuilt kernels are in `hq_ext/prebuilt/hq_ext.so` (sm_52 + sm_86 + PTX); rebuild with `hq_ext/build_prebuilt.sh`. `tools/ops/`: ONNX compatibility front end and the GPU test suite (`gpu_suite.py`, `gpu_suite_all.sh`). `results/gpu_suite/`: per-case results. Also docs on precision rules, compiler rules and chip issues. |
| `convert/` | `convert_gpu.sh`: one-command conversion (export, prep, data, parse, quant, GPU emulator gate, compile, chip, chip gate). `recipes/` has one recipe per model, `lib/` holds the stages. `fastc/` is the fast parallel compile plus the chip check. `CPU_GPU_SPLIT.md` explains where every step runs and lists every executor fix. `run_evidence/` has logs and gate results. |
| `hailo_re_src/` | Compiler and HEF reverse engineering. `turbo/turbo2.cpp` is an LD_PRELOAD library for the DFC `compiler` binary: it adds the CP-SAT solve cache and the level-max step control. Also there: own partitioner and per-context compiler (`turbo/`), HEF parser and differ (`hefre/`), op and hardware probes (`probe/`, `probe/RULES.md`), and the ONNX op coverage of 200 ops (`zoo/ops`). |
| `preproc_hailo/`, `gridnet_hailo/`, `session_20261119_hailo_trt/`, `tensorrt_candidates_src/` | Earlier per-model ports, session scripts and TensorRT build scripts. |

## Verified results

GPU executor vs Hailo's bit-exact simulator (HSim) on 144 cases: 13 zoo networks (ViT, DeiT, DaViT, LeViT, YOLOv5/6/8,
EfficientDet, ArcFace, HarDNet, RepVGG, CAS-ViT, Zero-DCE), the converted models, and 128 one-op probes in 8- and 16-bit.
All 144 pass under four tests:

- **T1:** every layer is exact when given HSim's own inputs.
- **T2:** the whole network, chained from the raw input, is exact.
- **T3:** results do not depend on batch size (2, 3, 5, 8, 16).
- **T4:** the fast executor builds and equals the reference.

Coverage: 31 layer types, every precision mode used (a8_w8_a8, a8_w8_a16, a16_w16_a8, a16_w16_a16) and 21 activations.

## Compiler findings (DFC 3.33)

- **One busy thread:** in each compiler process about 219 threads exist and exactly one is busy. The placement search is
  a sequence of single-worker OR-tools CP-SAT solves.
- **Repeated solves:** the same CP-SAT model is solved many times. `HAILO_CPCACHE=DIR` memoizes the solves exactly.
  Verify mode re-solves every cache hit and found 0 differences; the chip configuration in the HEF is identical.
- **Level max:** `compiler_optimization_level=max` walks the resource cap from 120 % to 75 % in 2.5 % steps, one attempt
  at a time, and each failing attempt waits for a wall-clock timeout (about 14 minutes on a mid-size model).
  `HAILO_UTIL_ONLY=X` starts the walk at X, so every step can run in parallel. On the models tested, a default-level
  compile with a fixed cap gave the same chip FPS in a fraction of the time (`convert/fastc/`).
- **Stock HEFs are not reproducible:** two runs differ in a timestamp and in uninitialized memory inside the `mapping`
  block. Use `hailo_re_src/hefre/hefdiff.py` to compare what the chip actually loads.

## Notes

Scripts use the absolute paths of the machine they were developed on (`/mnt/storage/...`). Set your own paths in
`convert/lib/env.sh` and `hailo8-exact-qat/tools/env.sh`. Model weights, datasets, HARs and HEFs are not included.
