# Hailo-8 precision modes, measured (DFC 3.33.1)

All numbers: one op in `x -> op -> identity 1x1 conv`, level-0 quantization, Hailo's bit-exact emulator (HSim =
chip) vs onnxruntime float, SQNR in dB (higher is better). Per-case results: `precision_results.json`,
`matrix*/<mode>/<case>/result.json`. Per-layer-class support table: `precision_table.txt`.

## 1. What the hardware offers
A precision mode is `a<input bits>_w<weight bits>[_a<output bits>]`. The SDK defines 18; each Hailo layer class
supports a subset (`SUPPORTED_PRECISION_MODE`, dumped in `precision_table.txt`). The ones that matter:

| layer class | 16-bit? | notes |
|---|---|---|
| Conv, Dense, Depthwise | a16_w16, a16_w16_a16 | also a16_w8_* / a16_w4_* (see 3: a16_w8_a8 fails to compile) |
| Standalone activation, LayerNorm, ReduceSum, AvgPool, EW add, Resize (MAC / NN), Input | a16_w16 | |
| **Matmul** | **no** (a8_w8_a16 = 16-bit OUTPUT only) | attention's two matmuls stay 8-bit |
| **ElementwiseMult, FeatureMultiplier (x*y, x^2)** | **no** | any product of two activations is 8-bit |
| **Softmax** (and its internal ew_mult) | **no** (a8_w8_a16) | |
| ArgMax | a16_w16 is accepted but does not compile ("16x4 kernel") | keep 8-bit |
| `a16_w8` (plain), all `_exp` modes | not on Hailo-8 | no layer class accepts them |

## 2. What 16 bits buys (per op, chip-exact)
| mode | works | median vs a8_w8 | notes |
|---|---|---|---|
| a8_w8 (default) | 203/203 | (median 35.6 dB) | |
| **a16_w16 incl. 16-bit input** | 187/203 | **+21.2 dB** (median 58.0 dB) | p90 +42 dB |
| a16_w16, 8-bit input | 187/203 | +1.1 dB | **the 8-bit input is the floor**: 16-bit gains need a 16-bit input stream (uint16 host format) or a 16-bit producer |
| **max16** (best supported mode per layer, solved jointly) | **all 203 compile** | **+14.6 dB** | includes every attention layout (plain a16_w16 compiles none) and the ViT block 32.7 -> 47.8 dB |
| a8_w8_a16 (16-bit output only) | 199/203 | 0.0 dB | |
| a8_w4 (4-bit weights on conv/dense) | 149/203 | -11 dB on convs | compression only |

Selected ops (a8_w8 -> 16-bit): conv3x3 33.6 -> 73.5, GlobalAvgPool 11.0 -> 78.7, BatchNorm 35.0 -> 59.7,
InstanceNorm 36.1 -> 60.4, GroupNorm 35.9 -> 59.1, LayerNorm 36.7 -> 49.2, RMSNorm 36.7 -> 49.6, Exp 23.6 -> 44.8,
1/x 25.9 -> 34.8, rsqrt 35.6 -> 43.7, GELU 34.6 -> 38.5, SiLU 33.0 -> 36.5. Activations gain least: the 9-piece APU
fit becomes the limit. Products (x*y 26 -> 28, x^2 33 -> 33) and attention (+0.2..0.8 dB) barely move: 8-bit only.

## 3. Rules learned (each from a failing case)
1. **Modes must agree along edges**: a producer's output bits = every consumer's input bits. `max16` picks, per
   layer, one supported mode by a small MILP (maximize #16-bit tensors, tie-break wider weights): the only way to
   put 16-bit LayerNorm/MLP around an 8-bit-only matmul (ViT block 32.7 -> 47.8 dB).
2. Give the modes to the layers that exist **after** the full-precision stage (the optimizer renames/creates
   layers, e.g. `mul_and_add`, softmax parts); names from the parsed HN give "Layer X not found".
3. **a16_w8_a8 passes the quantizer but the allocator fails** ("Assignment of the node needs super-defuse").
4. **w4 on weight-less layers** is accepted by the class table but rejected by the validator.
5. ArgMax a16_w16: compile fails ("16x4 is not supported").

## 4. Attention: why it is the weakest op on Hailo-8, and what does not fix it
Per-layer SQNR (layer_sqnr.py) of 2-head attention: q, k, v 35 dB -> **q k^T 20 dB** -> softmax 12 dB -> out 20 dB.
The matmul itself keeps ~13 significant bits (16-bit output, 1-bit shift). The loss is the **zero-point
correction**: q is symmetric around 0, so it gets an unsigned 8-bit encoding (zero point ~125); the DFC appends a
correction channel sum(k) computed by an inserted reduce_sum layer **quantized to 8 bits**, and its error is
multiplied by the zero point. Hailo's matmul requires a zero data zero-point unless transposed, and none of its
knobs remove this: `matmul_correction` zp_comp_block 19.7 / block_2 20.4 (compile crash) / block_3 **20.4**,
`matmul_decomposition` a16_w8 19.8 / a16_w16 (SDK config bug), `matmul_equalization` (refused). In trained
networks exact QAT recovers it (YOLO26m's PSA attention through exactly this arithmetic: no measurable loss).

## 5. Reproduce
```
PREC=max16 python probe.py run CASE OUT_DIR          # also: a16_w16, a8_w4, a8_w8_a16, a16_w8_a16 ...
PREC_NO_INPUT=1 ...                                  # keep the input layer 8-bit
python layer_sqnr.py Q.har M.onnx INPUT.npy          # per-layer SQNR: where the accuracy goes
```
