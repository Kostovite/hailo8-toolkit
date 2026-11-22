# Host-side op or on-chip approximation? (measured on Hailo-8, HailoRT 4.23)

Test model: 224x224x3 -> conv s2 32 -> conv s2 64 -> **OP on a 56x56x64 map** -> conv 64 -> conv s2 128 -> conv 32.
`chip` = one HEF, OP rewritten for the chip (onnx_hailo_compat + pwl_inject). `host` = the network split around OP:
part1 HEF -> OP in onnxruntime on the CPU -> part2 HEF. Accuracy = SQNR of the final output vs float (bit-exact
emulation). Speed = real chips, end-to-end through HailoRT (`hvc_time.py`), both chips idle.

## Accuracy: on-chip approximations lose nothing at model level
| OP | chip | host | |
|---|---|---|---|
| Identity (reference) | 43.6 dB | 43.3 dB | the split itself costs ~0.3 dB (boundary requantization) |
| Sin (PWL 9 pieces) | 43.5 | 43.4 | equal |
| Erf (PWL) | 43.1 | 42.9 | equal |
| Atan (PWL) | 43.0 | 43.0 | equal |
| SiLU (Hailo's own fit) | **37.9** | 44.0 | native SiLU costs 6 dB here: use the free-offset PWL fit (act_opt.py, +0.9..2.3 dB per op) |
| TopK / ScatterElements, GridSample, Hardmax(one-hot), bicubic resize | does not parse | host only | |

## 16-bit (PREC=max16: a16_w16 wherever the layer supports it, 16-bit boundary in the split)
| OP | chip 8-bit | host 8-bit | chip 16-bit | host 16-bit (16-bit boundary) |
|---|---|---|---|---|
| Identity | 43.6 | 43.3 | **75.3** | 75.5 |
| Sin (PWL 9 / 17 pieces) | 43.5 | 43.4 | 73.6 / 74.5 | 75.4 |
| Erf (PWL 9 / 17) | 43.1 | 42.9 | 71.8 / 71.7 | 73.5 |
| Atan (PWL 9) | 43.0 | 43.0 | 73.4 | 75.1 |
| SiLU, Hailo's built-in fit | 37.9 | 44.0 | **39.6** (the fit is the ceiling) | 71.7 |
| SiLU, our PWL (pwl_inject, PWL_SILU=1) | **44.1** | | **71.0** (17 pieces 71.4) | |
16-bit lifts everything ~30 dB. The PWL fits then show (1.7-1.8 dB under the host op, at 72-75 dB: negligible).
Hailo's built-in SiLU caps at ~39 dB in any precision: always replace it with our fit (+6 dB at 8 bit, +31 dB at 16).
16-bit speed: chip 1,829 FPS streamed vs 2,082 at 8 bit (-12%); staged split 690-770 FPS (boundary bytes x2).

## Speed: the split, not the host op, is the cost
| OP (mid-network) | chip FPS | host, 1 chip | host, 2 chips pipelined | host op time / frame |
|---|---|---|---|---|
| Identity, batch 1 | 1017 | 116 (8.8x slower) | 290 (3.5x) | 0.13 ms |
| Sin, batch 1 | 990 | 106 | 249 | 0.8 ms |
| SiLU, batch 1 | 1028 | 113 | 281 | 0.26 ms |
| TopK, batch 1 | n/a | 101 | 221 | 1.5 ms |
| GridSample, batch 1 | n/a | 130 | 265 | 0.34 ms |
| Identity, batch 8 | 2059 | 303 (6.8x) | 513 | 0.2 ms |
| TopK, batch 8 | n/a | 228 | 305 | 1.6 ms |
| Identity, batch 32 | 2276 | 322 | 487 | 0.8 ms |

On ONE chip every split makes HailoRT switch network groups per frame: **+7.6 ms/frame at batch 1, +2.8 ms/frame
at batch 8** (plus the host op). On TWO chips the halves pipeline, but the pair still delivers 3.5-4x less than one
chip running the whole model, and uses both chips.

Expected FPS with one mid-network host op: `1 / (1/fps_chip + 7.6 ms + t_op)` (batch 1).
YOLO26m-sized model (25 ms/frame): -23% at batch 1, ~-10% batched. A 1 ms/frame model: -88%.

## STAGED mode (host_split.py StagedRunner): the fix for a mid-network host op on one chip
Chunk of K frames through part1 -> boundary kept as raw chip integers in host RAM -> host op on the whole chunk
(onnxruntime, float32, Q/DQ fused into the host graph, 8 threads) -> chunk through part2. One network switch per
chunk; the host op of chunk k overlaps part1 of chunk k+1. Output identical to the float per-frame path (340 dB).
| OP (mid-network, 8-bit) | all on chip (streamed) | per-frame split, 1 chip | **staged, 1 chip (K=64-256)** | 2 chips pipelined |
|---|---|---|---|---|
| Identity | 2,082 | 140 | **1,350-1,405** | 296 |
| Sin | 2,290 | 125 | **920-1,092** | 247 |
| SiLU | 2,369 | 130 | **1,150-1,172** | 266 |
| TopK | n/a | 106 | **1,045-1,087** | 218 |
| GridSample (per-frame host op, threaded) | n/a | 132 | **896-936** | 239 |
| Identity, 16-bit boundary | 1,829 | 347 (b8) | **765** | 415 |
What it took: per-call `infer` of the whole chunk (not per frame); raw UINT8/UINT16 boundary vstreams (HailoRT's
float conversion of a 56x56x64 map costs 0.4 ms/frame); (de)quantization inside onnxruntime (numpy on one core
was 1.3 ms/frame). K = 64-256 is best; 1024 is slower on heavy host ops (cache). Latency = one chunk.
Workflow: `host_split.py plan M.onnx` (host-only regions, cut tensors, head/tail/mid) -> `split` -> compile part1
and part2 (PREC=max16 for a 16-bit boundary) -> `StagedRunner(part1.hef, host.onnx, part2.hef, chunk=128)`.

## Host RAM budget (edge devices): `StagedRunner(..., ram_mb=N)` + streaming `run_iter(frames)`
The chunk size is derived from the two HEFs' per-frame bytes: (queue_depth + 1) x (raw part1 output + raw part2 input)
+ host-op float temporaries (CPU host only) + the float input/output chunk; `mem_plan` reports it. `run_iter` takes
any frame iterator (camera, video) and yields outputs chunk by chunk, so nothing else is held. Measured on the test
net (56x56x64 boundary, 8 bit, CPU host op, 2,048 frames streamed one at a time; runtime baseline ~140 MB not counted):
| ram_mb | chunk | planned | measured RSS growth | FPS |
|---|---|---|---|---|
| 16 | 4 | 13.4 MB | ~7 MB | 562 |
| 64 | 19 | 63.6 MB | ~31 MB | 1,097 |
| 256 | 76 | 254.6 MB | ~123 MB | 1,200 |
The plan is conservative (~2x measured). Throughput saturates around 50-100 frames per chunk (64-256 MB here);
below ~10 frames the per-chunk network switch dominates. queue_depth=1 lowers RAM further at some overlap cost.
16-bit boundaries double the cache bytes; a GPU host op moves the float temporaries to GPU memory.

## Host op on a CUDA GPU (M40, GPU 0; `. gpu_env.sh`, StagedRunner(device="cuda:0"))
Two GPU backends: **onnx** = the same host.onnx (Q/DQ fused) on onnxruntime-gpu 1.19.2's CUDA EP (installed outside
the hailo env in hailo_re/ortgpu so the DFC keeps its CPU onnxruntime); **torch** = a torch function (NHWC float in/out),
raw chip integers copied to the GPU (1-2 bytes/value), Q/DQ on the GPU. Staged FPS, one Hailo chip, 8-bit boundary:
| OP (mid-network) | CPU onnxruntime | GPU onnxruntime | GPU torch | note |
|---|---|---|---|---|
| Identity | 1,350-1,420 | **1,446-1,534** | | chip-bound (chip-only staged ceiling ~1,760) |
| Sin | 920-1,092 | **1,468-1,549** | 1,240-1,332 | GPU +40-68% |
| SiLU | 1,150-1,172 | **1,471-1,536** | | |
| TopK | **1,045-1,102** | 818-839 | 992-1,090 | exported graph is batch-1 only: per-frame GPU calls lose; torch (batched) = CPU |
| GridSample | 896-970 | 1,083-1,140 | **1,143-1,236** | |
| Sin, 16-bit boundary | 672-688 | **764-809** | 718-741 | PCIe-bound |
Agreement with the CPU float path: bit-identical (340 dB) except GridSample (63-73 dB: GPU interpolation rounding
flips single LSBs at requantization) and 16-bit Sin (100-106 dB, single 16-bit LSBs).
Use the GPU when the host op is heavy (> ~0.5 ms/frame on the CPU: transcendental maps, GridSample, RoiAlign, a host
sub-network); light ops are chip-bound either way. Export batch-dynamic host graphs (or use torch) for the GPU.
Never GPUs 2/3 (production llama-server): gpu_env.sh pins CUDA_VISIBLE_DEVICES=${HOST_GPU:-0}.

## Decision rule
1. **An on-chip form exists with <= ~1 dB model-level loss -> on chip.** Every unary function (PWL), all the
   rewrites in onnx_hailo_compat, 16-bit modes where the layer supports them.
2. **No on-chip form, at the start or the end of the graph -> host** (parse with start/end nodes). Costs only the
   host op time (as NMS does today): TopK on the outputs, final softmax, NMS, decoding.
3. **No on-chip form, in the middle -> split + STAGED mode** (one chip, 45-70% of the all-chip rate on this 1 ms
   net, ~100% on heavy nets where the chip time dominates; latency = one chunk). Per-frame splitting only when
   frame latency matters (+7.6 ms/frame). For max accuracy build the boundary 16-bit (PREC=max16).
4. Recurrent layers (LSTM > 16 steps, GRU): compile-limited on chip; host or replace in the model.
