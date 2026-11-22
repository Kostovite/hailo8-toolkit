# Hailo DFC compiler: where the time goes, and a better solver

Measured 2026-11-22 on DFC 3.33. Tools: `cprof/gsample.sh` (gdb stack sampling; perf is blocked on the VM),
`turbo/turbo2.cpp` (hook), `turbo/cpbench.py`, `turbo/oldproto.py`, `turbo/cpserver.py`.

## 1. The compiler uses one core

- **One busy thread.** Each `compiler` process has about 219 threads, and exactly one of them is busy.
- **The hot path is serial.** Every sample lands in Hailo's cluster-mapping agent ("Trying parallel mapping"), which
  calls OR-tools `SolveCpModel` (vaddr 0x2aee3b0). That solver is OR-tools 7.x CP-SAT, running in sequential mode with
  one worker.
- **The "parallel" workers never overlap.** The other threads are a timeout monitor, a report-queue consumer and the
  resolver; they wait on that one agent.

## 2. The same problems are solved again and again

- **Router:** 111 solves, only 17 distinct problems.
- **orient3:** repeats are 89 % of solve time.
- **The cache.** `HAILO_CPCACHE` memoizes each solve, keyed by the exact model bytes plus the call site. A verify mode
  re-solved every cache hit and found 0 differences; the chip configuration in the HEF is identical.

## 3. Level max is a slow linear search

- `allocator::ResourceUtilizationDecrementer` (vaddr 0x140ec80; arguments 1.2, 0.9, -0.025, -0.01, 1.0) walks the cap
  from 120 % down to 75 % in 2.5 % steps, one attempt at a time. Each failing attempt waits for a wall-clock timeout,
  about 14 minutes on orient3.
- **orient3:** all 19 steps fail ("No valid partition found").
- **Router:** the walk ends on the default caps.
- **Faster route:** fixed-cap default-level compiles give the same chip FPS in a fraction of the time
  (`convert/lib/compile_fast.sh`).

## 4. The embedded solver gives up on solvable problems

Corpus: the 830 distinct problems from today's compiles (4,028 s of solving), read with the ORIGINAL schema, which
`oldproto.py` extracts from the binary.

| Hailo solve (wall) | Hailo result | modern CP-SAT 1 worker | 8 workers | 16 workers |
|---|---|---|---|---|
| 51.8 s | UNKNOWN | UNKNOWN 52 s | **OPTIMAL 3.1 s** | OPTIMAL 11.1 s |
| 49.3 s | UNKNOWN | UNKNOWN 49 s | **OPTIMAL 0.8 s** | OPTIMAL 2.3 s |
| 48.5 s | UNKNOWN | UNKNOWN 48 s | **OPTIMAL 8.2 s** | OPTIMAL 6.2 s |
| 46.4 s | OPTIMAL | UNKNOWN 46 s | **OPTIMAL 1.4 s** | OPTIMAL 1.5 s |

These are the largest problems (6 to 7 thousand variables, 8 thousand constraints). The gain comes from CP-SAT's
multi-worker portfolio (LNS and diverse heuristics), not from the newer version: one modern worker fails too.

**Earlier negative result, explained.** In October, setting `num_search_workers` INSIDE the embedded 7.x solver made
compiles fail. Its deterministic time budget is spent differently across workers, so solves stop before they finish.
The server below sidesteps this.

## 5. The modern solver server

`turbo/cpserver.py SOCKET` runs current OR-tools on a local socket; it is started automatically by
`convert/lib/compile_fast.sh`. With `HAILO_CPSERVER=SOCKET` set, the hook sends every cache miss there.
- The model is upgraded to the current schema, solved with N workers, and returned as an old-schema
  `CpSolverResponse`.
- If the server is unreachable or refuses a model, the hook falls back to the stock in-process solve.

| Model | stock solver (+ cache) | modern server | chip |
|---|---|---|---|
| router | 36 s | 22 s | both gate PASS, identical outputs metric, same FPS |
| orient3 | 1,846 s, 90.1 FPS | 1,065 s (1.7x faster), 88.9 FPS | both 0 / 2,000 decision diffs, guard clean |

The server finds different valid placements. They are not always faster on the chip, so `compile_fast.sh` runs both
solvers' candidates in parallel and the chip benchmark picks. Every candidate must pass `hailo8_guard`, run on the
COMPILED HAR: the checks for a split attention matmul and a wrongly lowered deconv read the allocator's own script.

## 6. A Hailo bug under concurrency

The DFC rotates its log files in `$VIRTUAL_ENV/etc/hailo/dfc`. Concurrent compilers race that rename and abort
("rotating_file_sink: failed renaming" -> "Compilation failed with unexpected crash"). Fix: give each compile its own
`VIRTUAL_ENV` / `CONDA_PREFIX` (done in `compile_fast.sh` and `fastc.sh`).

## Not done (where more time could still be won)

- The allocator's own graph code: in the "Applying partition" phase of multi-context models, about half the samples are
  in Hailo C++ rather than CP-SAT (`Resolver::ResolvePrepostFeasible`, `KernelCycles`). This would need re-implementing
  Hailo's resource model.
- The multi-context partition search (Slotter hill climb) is itself sequential. The own partitioner (`turbo/hcc.py`)
  compiles contexts in parallel and beat stock on yolo-nano (+8.8 % FPS, bit-exact); it fails on large attention
  contexts (matmul memory units).
