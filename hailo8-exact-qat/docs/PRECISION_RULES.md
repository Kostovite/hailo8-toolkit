# Hailo-8 precision rules: which layers and operations need 16-bit for maximum accuracy

Measured on YOLO26m (896 × 896, 14 classes) with the exact chip executor (`hailoqat/graph26.py`, chip == executor) and
the full 2,770-page bench (39 unseen authors; never trained on). Last update 2026-10-26.
Scripts: `hailoqat/precision/` (`gap_by_iou.py`, `fq_sim.py`, `layer_snr.py`); raw results in `precision/results/`.

## 1. Where the remaining accuracy is lost

| Model | mAP50-95 | gap to FP |
|---|---|---|
| FP (ceiling) | 0.8559 | — |
| Best chip model: top-8 `a16_w16` + 16-bit head outputs, exact QAT 2 ep (`m_w16top8_qat`) | 0.8462 | 0.0097 |
| int8 + 16-bit head outputs, exact QAT (`q3a`) | 0.8401 | 0.0158 |

The gap is **box-edge precision at strict IoU**, nothing else (`gap_by_iou.py`):

| IoU | 0.50 | 0.60 | 0.70 | 0.80 | 0.85 | **0.90** | **0.95** |
|---|---|---|---|---|---|---|---|
| AP gap FP − chip | −0.001 | 0.000 | −0.002 | +0.001 | +0.005 | **+0.025** | **+0.070** |

* At IoU <= 0.85 the chip model ties or beats FP: detection and classification are not the problem.
* It is concentrated in thin text-line classes: `foot` −0.034, `sec` −0.021, `equ` −0.019, `cap` −0.016
  (a 1 px edge error on a 15 px line is IoU < 0.95).
* It is spread over authors (30 of 32 lose a little): not an author / generalization effect.

## 2. What costs box precision (FP model with one quantization effect at a time, `fq_sim.py`)

8-bit activations (per-tensor asymmetric grid, min/max from 32 train pages):

| Region in 8-bit | mAP50-95 | cost | AP95 |
|---|---|---|---|
| none | 0.8559 | — | 0.658 |
| everything | 0.8130 | 0.043 | 0.585 |
| everything except the box branch | 0.8437 | 0.012 | 0.649 |
| **box branch only (6 convs)** | 0.8251 | **0.031** | 0.593 |
| class branch only (12 convs) | 0.8451 | 0.011 | 0.650 |
| neck outputs only | 0.8553 | 0.001 | 0.655 |

8-bit weights (BN folded, symmetric):

| Weights in 8-bit | per-layer scale (Hailo-8) | per-channel scale |
|---|---|---|
| box branch convs | 0.8426, AP95 **0.549** | 0.8556, AP95 0.657 |
| whole network | 0.8370, AP95 0.520 | 0.8548, AP95 0.655 |
| exactly the layers the best chip model keeps 8-bit | 0.8525, AP95 0.652 | |

**Hailo-8 has ONE weight scale per layer** (`mult_shift` is a scalar). Channels with a small weight range lose
bits: in `conv85` the median output channel peaks at |k| = 50 / 127, the weakest at 18; in the box output convs
(4 channels = left / top / right / bottom distance) the weakest side uses 17 / 127 (~5 bits).

Per-layer SNR of the chip model vs FP (`layer_snr.py`, 4 val pages): the neck outputs feeding the box branch are at
9–12 dB (`model.19.cv2` 9.4, `model.22.cv2` 9.9); box convs 8–16 dB; box outputs 22–25 dB. Layers with a fused
residual add (`elementwise_add=True`: `dw1`, `conv46`, `conv48`, `dw6`, `conv95`) show negative SNR only because their
output is the sum, not the torch module output; they are not broken.

## 3. Rules

**P1. Head outputs are always 16-bit** (`a8_w8_a16` on the 6 one2one output convs). Level 0: 0.731 → 0.762
mAP50-95 (val); int8 head outputs alone cost 0.029, mostly box precision.

**P2. The box-regression path is the most sensitive part of a detector; give it 16-bit first.** Activations: the 6
box convs (`one2one_cv2.{0,1,2}.{0,1}`) carry 3/4 of the whole network's 8-bit activation cost (0.031 of 0.043).
On YOLO26m: `conv69 conv70 conv85 conv86 conv99 conv100` (top-8 already had 4 of them).

**P3. Score weight precision as well as output rounding.** `sens.py` scores only the output-rounding noise (E[g²]/12),
so it never flags a layer whose OUTPUT is already 16-bit but whose WEIGHTS are 8-bit with one scale per layer: the box
output convs `conv71 conv87 conv101` (`a8_w8_a16`). Flag any layer with few output channels or a wide spread of
per-channel weight range (median max|k| per channel << 127). For box outputs use `a16_w16`: their inputs are already
16-bit, and a 64→4 1×1 conv costs almost nothing.

**P4. A 16-bit layer cannot restore precision its input never had.** Put the producers of the box path's inputs in
16-bit too (YOLO26m: `conv67` = `model.16.cv2`, `conv83` = `model.19.cv2`, `conv98` = `model.22.cv2`).

**P5. The class branch can stay 8-bit.** It costs 0.011 before QAT, and QAT recovers it fully (chip AP50 is above FP).

**P6. For maximum accuracy, do not exclude a layer because its tensor crosses a context cut.** The original top-8 pick
skipped `conv83 conv85 conv98` for that reason: an FPS optimization, not an accuracy one.

**P7. All-16-bit is not the answer.** Nano: all layers `a16_w16` removes the rounding noise but gains nothing more on
unseen authors (0.698 vs top-8 0.698 mAP50-95) at 3.7× the chip time. Pick by sensitivity.

**P8. Precision modes that are broken or refused:**
* `a8_w8_a16` on a NON-output layer gives its consumer a 16-bit WRAPPING accumulator (mAP 0.02). Use `a16_w16` for
  internal layers, `a8_w8_a16` only on network outputs.
* `a8_w8_a16` on PWL-activation layers: compile refusal ("precision mode is not accurate").
* Concat advertises `a8_w8_a16` but rejects it; ReduceSum 16x4 fails; relu `a16_w16_a8` unsupported.
* 16-bit layers differ from SDK_BIT_EXACT by 1–2 LSB on ~0.2% of outputs (A5, open; chip == our executor).

**P9. Operations that are 8-bit-only by construction and need other fixes, not 16-bit:** the softmax decomposition
(matmul → reduce_max → ew_sub/exp → reduce_sum/1÷x → ew_mult) is all 8-bit; its denominator accumulator wraps above the
calibrated range (2.6% of yolo26m C2PSA rows). Fix with `turbo/smx_headroom.py` BEFORE QAT (after QAT it lowers mAP:
the model learned the wrap).

**P10. Cost of 16-bit on Hailo-8:** 3×3 conv 50 → 2.8 MAC/SC/cycle (18×), 1×1 48 → 7.4 (6.5×), depthwise ~1×;
16-bit add 2.5×, mul 7×, concat 2×, activation 2.5×; a 16-bit tensor crossing a context doubles its link bytes.

## 4. Result: the whole box path in 16-bit (`mopt/w16box.sh`, run `m_w16box_2ep`, 2026-10-26)

`a16_w16` on top-8 + `conv85 conv99 conv71 conv87 conv101 conv83 conv98` (P2-P4), same recipe as `m_w16top8`.

| Model (full bench mAP50-95) | full | bench_sel | bench_test |
|---|---|---|---|
| FP | 0.8559 | 0.8679 | 0.8553 |
| previous best `m_w16top8_qat` (round-tripped) | 0.8462 | 0.8660 | 0.8415 |
| box path, step 6900 = pick on bench_sel, **round-tripped HAR / HEF** | **0.8517** | 0.8664 | 0.8499 |
| box path, final step 9200 (no selection), training eval | **0.8559** | 0.8646 | 0.8563 |

* At equal training steps the box-path run is ahead from step 3450 on; the final step reaches FP on the full bench.
* Write-back round trip: 168/168 layers exact; the two new 16-bit neck biases (`conv83`, `conv98`) were re-decomposed
  by the DFC (s6900: 0.8524 trained -> 0.8517 stored).
* Chip cost: 6 contexts (was 5); batch-1 15.7 FPS vs 20.8 for `m_w16top8_qat` (same chip, same session).
