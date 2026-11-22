# Compile guide: model to verified Hailo-8 HEF

This guide covers building the toolkit, installing the fast compiler, and running a conversion that ends in a HEF checked
on the chip. Tested with Hailo DFC 3.33, HailoRT 4.23, CUDA 12.6, Tesla M40 (sm_52) and RTX 3090 (sm_86).

## 1. Requirements

| What | Why |
|---|---|
| Hailo Dataflow Compiler (its own Python env) | parse, quantize, compile (not included in this repo) |
| HailoRT + `hailortcli` + a Hailo-8 | chip runs and FPS benchmarks |
| A CUDA GPU + PyTorch env | exact GPU emulator (`hailo8-exact-qat`), torch references, gates |
| g++ >= 9 | builds the compiler hook |

Set your own paths in `convert/lib/env.sh` (`HPY` is the DFC python, plus the torch env, the chip lock and the repo root)
and in `hailo8-exact-qat/tools/env.sh`. The scripts use absolute paths.

## 2. Build

```bash
# GPU kernels for the exact executor (prebuilt .so for sm_52 + sm_86 + PTX ships in the repo)
hailo8-exact-qat/hailoqat/hq_ext/build_prebuilt.sh        # -O3, -Xptxas -O3, no fast-math (exactness)

# compiler hook: CP-SAT memo cache + level-max control
hailo_re_src/turbo/build.sh                               # -> turbo/bin/libhailo_turbo.so (-O3, x86-64-v3 = any AVX2 CPU)
```

## 3. Install the fast compiler (once)

```bash
hailo_re_src/turbo/install_fast.sh             # no root: every compile from the DFC python loads the hook
hailo_re_src/turbo/install_fast.sh --root      # optional: also wraps hailo_tools/build/compiler (any user / launcher)
hailo_re_src/turbo/install_fast.sh --uninstall # back to stock
```

After installing, every `ClientRunner.compile()`, the `hailo` CLI and the pipeline use the hook automatically.
`HAILO_FAST=0` turns it off for a single run.

What the hook does:

- **CP-SAT memo cache (`HAILO_CPCACHE`, a shared directory).** The compiler's placement search solves the same CP-SAT
  model many times; the router compile makes 111 solves, only 17 of them distinct. Each model is stored with its answer,
  keyed by its exact bytes plus the call site. A hit returns the stored answer. A rebuild or retrain of the same
  architecture is served almost entirely from the cache. The cache is safe to share between concurrent compiles.
- **Exactness checks.** `HAILO_CPCACHE_VERIFY=1` re-solves every hit and counts any difference; 0 were found. The HEF is
  identical to stock in everything the chip loads (check with `hailo_re_src/hefre/hefdiff.py`).
- **Audit log.** `CACHE/stats.log` gets one line per compile: calls, hits and seconds saved.
- **Level-max control.** `HAILO_UTIL_ONLY=X` starts the level-max utilization descent at X.

## 4. Convert a model (one command)

```bash
convert/convert_gpu.sh RECIPE CHECKPOINT OUT_DIR GPUS
#   e.g. convert/convert_gpu.sh orient3 orient3_10000.pt runs/orient3 0,1,2,3
```

Stages: `export prep data parse quant emu gate_fp gate_q compile chip gate_chip`.

- Each stage leaves an `OUT/.done_<stage>` marker and a `logs/<stage>.log`. Delete a marker to rerun from that stage.
- `emu` is the exact GPU emulator; it refuses to run unless the coverage against HSim is exact.
- `gate_q` stops the run if accuracy on the emulator fails.
- `compile` is described in section 5.
- `chip` and `gate_chip` run the eval set on the Hailo-8 and apply the same task metric.

## 5. The compile stage (`convert/lib/compile_fast.sh`)

1. **Replay.** The architecture signature (layer names, types, shapes, precision; not the weights) is looked up in
   `compile_cache/RECIPE_SIG.auto.alls`. On a hit, the stored allocator solution is replayed and the search is skipped
   (router: 37 s to 16 s, identical outputs).
2. **Fast portfolio, on a miss.** The compiler runs with the hook at default level, as parallel candidates.
   - **Stock-solver candidates:** default caps, plus `resources_param` at each value in `FASTC_UTILS` (default
     `0.7 0.8 0.9`).
   - **Modern-solver candidates:** `srv_d` and `srv_u0.8` (set with `FASTC_SRV_UTILS`; `FASTC_SRV_DEFAULT=0` drops them).
     Their cache misses go to `hailo_re_src/turbo/cpserver.py`, which runs modern multi-worker CP-SAT and is started
     automatically. It solves in seconds problems the embedded solver gives up on after about 50 s; see
     `hailo_re_src/turbo/SOLVER_FINDINGS.md`.
   - **Separate caches:** each solver has its own memo cache (`compile_cache/cpsat`, `compile_cache/cpsat_srv`).
   - **Isolation:** every candidate is RAM-capped by systemd (`MEMMAX`, default 24G) and has a private SDK home.
     Concurrent DFC compilers otherwise race on the log-file rotation and abort with "unexpected crash".
   - **Guard:** every candidate that compiles must pass `hailo8_guard` (all 6 known silent-wrong chip patterns), run on
     its COMPILED HAR. The checks for a split attention matmul and a wrongly lowered deconv need the allocator's own
     script, which the quantized HAR does not have.
   - **Pick:** guard-clean candidates are benchmarked on the chip with `hailortcli`. The fastest wins; it must be more
     than 0.5 % faster to beat an earlier candidate. Its solution goes into the replay cache.
3. **`COMPILE_OPT=max`** forces the stock `compiler_optimization_level=max`. It is not recommended:
   - level max walks the cap from 120 % down to 75 % in 2.5 % steps, one attempt at a time, and each failed attempt
     waits for a wall-clock timeout (about 14 min on orient3);
   - on orient3 all 19 steps failed ("No valid partition found");
   - on the router it ended on the default caps;
   - fixed-cap candidates gave the same chip FPS (orient3: 90.09 FPS at caps 0.85 to 1.2) in about 31 min instead of 4+ h.

Chip check of any HEF against a run's eval set and gate: `convert/fastc/chipcheck.sh RUN_DIR HEF TAG`.
Parallel candidates for an existing run: `convert/fastc/fastc.sh RUN_DIR NAME [UTIL]`.

## 6. Verify the GPU executor

```bash
hailo8-exact-qat/tools/ops/gpu_suite_all.sh GPU CASE_LIST   # lines: NAME HN CAPTURE INPUT
```

Each case runs four tests:

- **T1:** every layer is exact when given HSim's own inputs.
- **T2:** the network, chained from the raw input, is exact.
- **T3:** results do not depend on batch size (2, 3, 5, 8, 16).
- **T4:** the fast executor builds and equals the float64 reference.

Current result: 144 / 144 cases (13 zoo nets, the converted models, 128 one-op probes in 8- and 16-bit).
See `convert/CPU_GPU_SPLIT.md` for every executor fix.

## 7. Where each step runs

| Step | Runs on | Notes |
|---|---|---|
| torch reference, emulator gate, task gates | GPU | exact integer executor; replaces DFC `SDK_QUANTIZED` (one CPU core for 16-bit) |
| QAT / fine-tune of integer weights | GPU | `qatfast` |
| HSim capture (one input per HAR) | CPU, minutes | records the integers and scales the GPU executor uses |
| DFC parse + quantize statistics | CPU (+ DFC's TF) | minutes |
| compile (placement search) | CPU | Hailo's search is serial CP-SAT; sped up by the memo cache, replay and parallel candidates, not ported to GPU |
| chip run, FPS | Hailo-8 | |

## 8. Traps

- **Always prepend to `LD_PRELOAD`, never replace it.** The DFC env preloads the system cuDNN and a GPU shim.
- **Never edit a running shell script.** Bash reads scripts as it goes; write a new file and `mv` it into place.
- **Do not judge a HEF by its md5.** Stock HEFs differ between identical runs: a timestamp, plus uninitialized memory in
  the `mapping` block. Use `hefdiff.py`.
- **GPU exactness rules.** cuDNN may pick FFT or Winograd convolutions depending on batch size, which is not exact; the
  executor forces im2col+GEMM (`HQ_CONV=gemm`). CUDA float32 division is not correctly rounded; input quantization
  divides in float64 and rounds to float32.
- **Load.** Each candidate's early feasibility phase uses 3 to 7 cores, while the search itself uses one core. Keep the
  candidate count times RAM per compile within the machine (`MEMMAX`).
