# Why 16-bit Hailo-8 models compile for hours (and fail): findings 2026-11-22/23

Hailo DFC 3.33 `compiler`, Hailo-8, model PP-DocLayoutV3 (document layout detector), mixed 8/16-bit
("mix16": upsample blocks and heads 16-bit). Full rule list with addresses: `probe/RULES.md` section 29 (A-H).

## 1. "Assignment of the node needs super-defuse" = 16-bit tensors on a token row

The AIFI transformer encoder works on 784 tokens (28x28). In token layout a 16-bit tensor is one 784-wide row
(392 KiB) that no cluster can hold. Fix, float-exact: `convert/recipes/aifi_spatial.py` moves every per-token op to
NCHW 28x28 (MatMul with constant weight -> 1x1 conv with the bias folded in, LayerNorm over the channel axis,
positional embedding as a 28x28 constant, the P.V bias folded into the output projection). The DFC parser still puts
the K and V projections behind the token-layout conversion, so `convert/recipes/kv8_alls.py` forces those two convs
to `a8_w8_a8`. Accuracy gate (emulator box AP vs float): all four sets within 0.0022.

## 2. The partition search starts from a node COUNT, not a cost

`allocator::Slotter` (`slotter.cpp`):

* per-context start budget (0x15f5960): `B = max(1, floor(min(n_compute_nodes, ceil(128 * 0.6)) * x1))`, at most
  77 nodes. `128` = layer controllers; `n_compute_nodes` (0xe1c670) counts graph nodes passing a type filter.
* `find_initial_partition` (0x1608e70): `n_ctx = ceil(n_compute_nodes / B)`, then nodes in topological order, a
  context closes when it holds B compute nodes.

Every node weighs 1. A 1.7 MB 16-bit 1x1 conv counts the same as an element-wise add, so for 16-bit nets every
starting context is overloaded and the hill-climb spends hours walking back to a feasible split.

Measured with conv-chain pseudo-models (`probe/slotcount/`, 100 chained 3x3 convs at 56x56, all start B = 77 =
2 contexts):

| chain | final contexts | Slotter iterations | partition time |
|---|---|---|---|
| 64 ch, 8-bit | 2 | 76 | 7 m 20 s |
| 64 ch, 16-bit | 4 | 243 | 15 m 26 s |
| 256 ch, 8-bit | 6 | 283 | 29 m 12 s |
| 256 ch, 16-bit | 9 | 440 | 1 h 1 m |
| 256 ch, 16-bit, start B = 11 (hook) | 10 | 61 | **11 m 5 s** |

A start near the real context count cuts the search 5.5x. The hook (`turbo/turbo2.cpp`, `ctx_hook` on 0x15f5960):
`HAILO_INIT_UNITS=n` (fixed B), `HAILO_INIT_SCALE=s` (B x s), `HAILO_INIT_LOG=1`. On PP-DocLayoutV3 the stock start
found no valid partition in 3-7 h; s = 0.12 found 28 contexts, s = 0.08 found 41 contexts in 53 min.

## 3. A "valid" partition can still fail final mapping

The Slotter's validity check is weaker than the per-context placement (Resolver / Splitter). Twice the 28-context
partition failed after ~3 h at the same context: 16 pieces of a 16-bit 1664->512 1x1 conv fed by a 1664-channel
16-bit concat, 22 cluster-mapping reverts. Longer `allocator_param(splitter_timeout, cluster_timeout)` did not help.
Compiled alone, that conv fits only in a context of its own.

## 4. Partition by compiling, not estimating: `turbo/ctx_grow.py`

Each context is grown along the topological order by real single-context compiles of candidate spans
(`context_switch_param(mode=disabled)`, several ends in parallel, results cached), so every context of the result is
known to compile. Then `turbo/hcc.py` compiles every context in several variants, assembles them into one model
script and runs the full compile.

Two compiler behaviours found on the way:

* `map::at` in "Trying to apply higher utilization solution": the span FITS, the compiler crashes while spreading it
  over the idle chip. `performance_param(fps=20)` compiles it (fps = 100 still crashes). The fit test therefore runs
  at a capped throughput (`CTXG_FPS_LADDER`, default `20`; `20,5` for very heavy layers), and `hcc.py` has capped
  variants `f20 / f50 / f100 / f200`.
* A layer's footprint follows the throughput target: alone at the default level a 112-wide 16-bit conv takes 15 HW
  nodes / 113 subclusters; with `performance_param(fps=20)` 4 nodes / 10 subclusters.

## 5. Compiler FPS estimate vs the chip

24 chained 3x3 convs at 56x56, `hailortcli run -t 5`:

| chain | contexts | estimate FPS | chip FPS |
|---|---|---|---|
| 64 ch, 8-bit | 1 | 1907 | 1679 |
| 64 ch, 16-bit | 1 | 265 | 263 |
| 256 ch, 8-bit | 1 | 144 | 143 |
| 256 ch, 16-bit | 2 | 23.1 | 7.99 (batch 8: 7.99) |

Single-context estimates are within 12 %; multi-context 16-bit is ~2.9x optimistic (weights are reloaded on every
context switch). 16-bit layers cost 6-18x their 8-bit form on the chip, so heavy 16-bit layers at high resolution
decide the speed of the whole model.

## 6. Other tools

* `probe/layer_cost.py`: every layer compiled alone, resources read back from the HEF (upper bounds at the default
  level; set `LC_ALLS="performance_param(fps=20)"` for the minimum footprint).
* `turbo/part_pack.py`: pack contexts against measured per-layer costs.
* `cprof/fnmap_lib.py`, `cprof/phases.py`: name stripped compiler frames from `__PRETTY_FUNCTION__` / assert strings.
* Ghidra (12.1.4) on the 50 MB compiler binary does not finish importing within 5.5 min; objdump + the string map
  was enough for the functions above.

## Status (2026-11-23)

PP-DocLayoutV3 mix16: `ctx_grow` verified 14 contexts for layers 0-213 of 217; the last three (224x224 16-bit mask
head, two-input depthwise conv) do not place in one context yet. No HEF yet.
