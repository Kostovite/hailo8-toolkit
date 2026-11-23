# Hailo-8 HEF back end: config-write format, address map, weight layout (2026-11-23)

Goal: decode the bytes the compiler writes into the chip, as the first step towards emitting HEFs ourselves.
Tools: `ccwdec.py` (CCW decoder), `addrmap_survey.py`, `cluster_sub.py`, `wexp/` (controlled weight experiment).

## 1. Config writes (CCW) — the format

Every context's configuration is a list of config-channel writes (HailoRT 4.23 `hef.cpp` `parse_ccw_buffer`):

```
u32 words_count_minus_1 | u32 address | (words_count_minus_1 + 1) * 4 data bytes | pad to 8 bytes
```

A `write_data_ccw_ptr` action points into the CCW blob after the protobuf; the region is a back-to-back stream of
CCWs. Decoded cleanly (no malformed record) for all 649 grid probe HEFs.

## 2. Address map

| address | what | evidence |
|---|---|---|
| `0x1100_0000 + k * 0x0200_0000`, k = 0..7 | cluster k window | top byte 0x13/15/17/1b/1d/1f written only when the sequencer enables cluster k: 1,814 / 1,814 HEF-cluster pairs |
| cluster + `0x00_8000 ...` | subcluster L2 memory (weights when `mixed_mem=enabled`), 32 KB slots | raw weights found here (section 4) |
| cluster + `0x40_0000 + x` | the same memory written **through the decompressor** | compression off moves every block from +0x40_8000 to +0x00_8000 |
| cluster + `0x00_7fc0` (64 B) | per-cluster header written before each memory block | every HEF |
| `0x1180_0000`, `0x1980_0000` (248 B) | decompressor dictionary, cluster 0 and cluster 4 (one per cluster group 0-3 / 4-7) | written in all 649 HEFs; gone when compression is off |
| `0x0020_0684` | one-hot bit per write group (a done / enable mask) | between blocks, bits 0x400 .. 0x400000 |
| `0x2020_xxxx`, `0x2024_xxxx`, `0x206x_xxxx`, `0x207x_xxxx`, `0x2904_xxxx`, `0x3000_xxxx` | non-cluster units (streams, aligners, DMA / sequencer tables) | always written; not decoded yet |

## 3. Config compression (on by default)

* `hef_param(params_load_time_compression=allowed)` (default) compresses the cluster memory images:
  a 62-entry byte dictionary (248 B, one u32 per entry: `00 00 01 02 10 08 ff 04 80 7f 20 0e ...`, most common
  first) + a nibble-coded stream (`1` = the most common byte, 0x00). Block magic `20 d3 15 11` / `20 63 6e 44`.
* `hef_param(params_load_time_compression=disabled, should_compress_sequencer_data=False)` writes plain bytes
  (the cluster images grow 4.2 KB -> 7.8-8.0 KB for an 8x8 1x1 conv). Use this for every decoding experiment.
* `compilation_param(L, compress_l3_weights=disabled)` is a different thing (L3 weights) and does nothing when the
  weights sit in L2.
* The codec itself (`cat::Cat::create_memory_mcw_action`, `../src/cat/catware.cpp`) is not decoded yet. It is only
  needed to write small HEFs; the chip accepts uncompressed images.

## 4. Weight layout — 1x1 conv, 8 -> 8 channels, 8-bit (controlled experiment `wexp/`)

Three models differing in exactly one quantized weight each (A; B: K[cin5][cout3] -37 -> 101; C: K[cin2][cout6]
-31 -> -102), compression off. One weight change = the weight byte (twice) + its output channel's bias/zero-point
words (`merge_residue_into_bias`), nothing else.

* The compiler defused the conv by output channel: `conv1_d0` (cout 0-3) on cluster 6, `conv1_d1` (cout 4-7) on
  cluster 5, each 2 subclusters (indices 0 and 57), `use_16x4_sc=enabled`, `mixed_mem=enabled`; `conv1_dc`
  (the concat) on cluster 4.
* Weights are **plain int8** at cluster + 0x8000 + ~5200 B, **one 16-byte group per input channel, cin 7 -> 0**
  (descending). Inside a group: 4-byte words `[w(cin, co), w(cin, co + 2), 0, 0]`, each word written **twice**
  (one per subcluster), pairs (co, co+2) = (0, 2), (1, 3) on cluster 6 and (4, 6), (5, 7) on cluster 5.
  The two zero lanes are the idle half of the MAC array (8 channels; RULES section 4 "channels < 32 waste it").
* Per-output-channel params follow at +~7360: words like `7fff`, `630e`, `0408`, replicated per lane; they change
  with any weight of that channel (zero-point correction folded into the bias). Not decoded yet.

### Chip-verified (hailo0, 2026-11-23)

`hefpatch.py` edited the two weight bytes of K[cin5][cout3] (-37 -> 101) in A's HEF (cluster 6 + 0x8000 + 5249 /
5253) and re-hashed the header (v2: XXH3-64 of everything after the 44-byte header, big-endian at byte 12).
HailoRT loaded it; on the chip **only output channel 3 changed** (96% of pixels), the other 7 channels are
bit-identical to A; the change correlates 0.83 with the recompiled B's channel-3 change (B also has a new output
scale and bias correction, which the patch did not touch). So: weight location + format confirmed on silicon, and
HEFs can be edited without the compiler (`chiprun.py OUT.npz DEV HEF...` runs them on a fixed input).

## 5. Next experiments (in order)

1. Same 3-variant method over cin / cout / k (3x3) / 16-bit, to get the general weight-layout rule
   (lane packing, kernel position order, the 16x4 SC mode vs normal).
2. Per-channel param block: vary bias, activation, output scale one at a time.
3. The `0x20xx` / `0x29xx` / `0x30xx` tables: vary H, W, stride, buffer sizes (streams, aligners, DMA).
4. DONE for weights (section 4). Next: patch weight + bias words and predict the chip output exactly from
   the integer arithmetic (hailoqat intops), i.e. close the loop bit-exact.
5. Codec: decode the nibble stream against the plain image of the same HEF (both are available now).
