# Hailo-8 operator support, measured (DFC 3.33.1, HailoRT 4.23)

Every number below is measured, not taken from documentation: `probe.py` (169 hand-built ONNX cases) and
`torch_cases.py` (88 cases exported by `torch.onnx.export`, opset 17) push one op through
`x -> op -> identity 1x1 conv`, parse, quantize at optimization level 0, run Hailo's bit-exact emulator (HSim, which
matches the chip) and compare with onnxruntime float. **chip dB** = SQNR of the bit-exact output; 8-bit floor
(no op) = **39.0 dB**. Cases that compile also produce a HEF.

## Summary
- 259 cases: 142 work natively, 117 do not (100 fail,
  7 compile but compute the WRONG function, 10 lossy).
- **57 of 117 non-native cases fixed** by the two tools here; 60 open (list at the end).

## Hardware facts that drive the workarounds
1. **The APU (activation unit) is a 9-piece PWL with non-negative slopes.** Every activation, native or not, is
   evaluated as at most 9 linear pieces per channel; on Hailo-8 (SAGE) the slope mantissa is unsigned, so a
   non-monotone function cannot be one APU layer. `pwl_inject.py` fits any unary f (least squares on the calibrated
   input distribution), splits it f = sum G - sum H into non-decreasing layers that are flat outside their window,
   and injects them as native `pwl` activations. Offsets are free per piece, so step functions are exact.
2. **The 8-bit input is the precision floor.** Sin over +-7.5 tops out at ~31 dB however many pieces: the input step
   (0.059) times the slope. Floor/Round/Sign are exact in float but 15-22 dB on chip for the same reason. More
   precision needs a 16-bit producer; 9 pieces / 8-bit is the robust default (17 pieces + a16 for periodic f).
3. **The parser reads attributes without ONNX defaults** and silently ignores some semantics (pad mode,
   output_padding, Floor). Always run `onnx_hailo_compat.py` (it constant-folds, writes defaults, and checks
   float equivalence with onnxruntime) and compare `sqnr_native_db` if you add a new op.

## Natively supported (142 cases)
| case | status | chip dB | fix | after dB |
|---|---|---|---|---|
| Abs | native | 36.7 |  |  |
| Add_tensors | native | 34.3 |  |  |
| AvgPool2s2 | native | 36.1 |  |  |
| AvgPool3s1_pad | native | 34.1 |  |  |
| AvgPool3s1_pad_incl | native | 36.3 |  |  |
| AvgPool3s2 | native | 34.5 |  |  |
| AvgPool7s1 | native | 33.3 |  |  |
| BatchNorm | native | 35.0 |  |  |
| ChannelShuffle_pattern | native | 39.0 |  |  |
| Clip_pm1 | native | 36.6 |  |  |
| Clip_relu6 | native | 38.3 |  |  |
| Concat_C | native | 34.3 |  |  |
| Concat_W | native | 35.6 |  |  |
| Const_div_x | native | 25.9 |  |  |
| Conv1d_pattern | native | 33.9 |  |  |
| Conv3x3 | native | 33.6 |  |  |
| Conv3x3_dil2 | native | 33.7 |  |  |
| Conv3x3_s2 | native | 34.1 |  |  |
| Conv7x7 | native | 33.3 |  |  |
| ConvTranspose_k4s2 | native | 34.4 |  |  |
| ConvTranspose_s2 | native | 33.6 |  |  |
| DWConv3x3 | native | 34.1 |  |  |
| DepthToSpace | native | 39.0 |  |  |
| Div_by_const | native | 39.0 |  |  |
| Div_tensors | native | 28.2 |  |  |
| Elu | native | 36.0 |  |  |
| Gather_C | native | 36.3 |  |  |
| Gelu_sigmoid_pattern | native | 35.2 |  |  |
| GlobalMaxPool | native | 48.0 |  |  |
| GroupConv_g4 | native | 33.5 |  |  |
| HardSigmoid | native | 44.5 |  |  |
| HardSwish | native | 30.7 |  |  |
| HardSwish_pattern | native | 30.7 |  |  |
| InstanceNorm | native | 33.8 |  |  |
| L2Norm_pattern | native | 35.9 |  |  |
| LayerNorm_C | native | 34.8 |  |  |
| LeakyRelu | native | 36.8 |  |  |
| Log | native | 36.8 |  |  |
| LogSoftmax_C | native | 43.7 |  |  |
| LogSoftmax_W | native | 44.4 |  |  |
| MaxPool2s2 | native | 39.3 |  |  |
| MaxPool3s1 | native | 41.4 |  |  |
| MaxPool5s1 | native | 43.8 |  |  |
| Max_tensors | native | 27.5 | compat rewrite | 34.2 |
| Min_tensors | native | 33.1 | compat rewrite | 34.0 |
| Mish_pattern | native | 32.2 |  |  |
| Mul_tensors | native | 26.0 |  |  |
| Neg | native | 39.0 |  |  |
| PRelu | native | 35.6 |  |  |
| Pad_const | native | 39.0 |  |  |
| PixelShuffle_pattern | native | 39.0 |  |  |
| Pow05 | native | 50.3 | compat rewrite | 50.3 |
| Pow2 | native | 33.1 | compat rewrite | 33.1 |
| Reciprocal | native | 25.9 |  |  |
| ReduceL2_C | native | 44.0 |  |  |
| ReduceL2_HW | native | 50.8 |  |  |
| ReduceL2_W | native | 45.3 |  |  |
| ReduceMax_C | native | 43.2 |  |  |
| ReduceMean_C | native | 36.3 |  |  |
| ReduceMean_W | native | 36.4 |  |  |
| ReduceMin_C | native | 43.1 |  |  |
| ReduceSumSquare_C | native | 39.2 |  |  |
| ReduceSumSquare_HW | native | 45.3 |  |  |
| ReduceSumSquare_W | native | 40.6 |  |  |
| ReduceSum_C | native | 36.3 |  |  |
| ReduceSum_HW | native | 31.5 |  |  |
| ReduceSum_W | native | 36.1 |  |  |
| Relu | native | 38.0 |  |  |
| Reshape_flatHW | native | 39.0 |  |  |
| Resize2x_nearest | native | 39.0 |  |  |
| Rsqrt_pattern | native | 35.6 |  |  |
| SiLU_pattern | native | 33.0 |  |  |
| Sigmoid | native | 40.7 |  |  |
| Slice_C | native | 36.0 |  |  |
| Slice_HW_stride2 | native | 36.3 |  |  |
| Softmax_C | native | 34.5 |  |  |
| Softmax_W | native | 33.1 |  |  |
| Softplus | native | 38.1 |  |  |
| Softsign | native | 33.7 |  |  |
| Sqrt | native | 50.3 |  |  |
| Square_mul | native | 33.1 |  |  |
| Sub_tensors | native | 34.2 |  |  |
| Swish_beta | native | 35.2 |  |  |
| T_Abs | native | 36.7 |  |  |
| T_AdaptiveAvg4 | native | 36.8 |  |  |
| T_AdaptiveMax4 | native | 44.7 |  |  |
| T_Amax_C | native | 43.2 |  |  |
| T_AvgPool3s1 | native | 36.3 |  |  |
| T_AvgPool3s1_nopadcount | native | 34.1 |  |  |
| T_BatchNorm2d | native | 39.0 |  |  |
| T_Chunk_add | native | 36.0 |  |  |
| T_Clamp | native | 37.6 |  |  |
| T_Conv1d_seq | native | 35.3 |  |  |
| T_Conv_5x1 | native | 34.9 |  |  |
| T_Conv_dil3 | native | 35.0 |  |  |
| T_Div_tensor | native | 31.0 |  |  |
| T_ELU | native | 36.0 |  |  |
| T_F_normalize_C | native | 35.9 |  |  |
| T_GELU | native | 34.6 |  |  |
| T_GroupNorm | native | 35.9 |  |  |
| T_GroupNorm1 | native | 36.4 |  |  |
| T_Hardsigmoid | native | 45.3 |  |  |
| T_Hardswish | native | 30.7 |  |  |
| T_Hardtanh | native | 36.6 |  |  |
| T_InstanceNorm2d | native | 36.1 |  |  |
| T_LayerNorm_C | native | 36.7 |  |  |
| T_LayerNorm_CHW | native | 31.5 |  |  |
| T_Linear_tokens | native | 34.5 |  |  |
| T_Log | native | 36.8 |  |  |
| T_LogSigmoid | native | 29.3 |  |  |
| T_LogSoftmax_C | native | 43.7 |  |  |
| T_MaxPool3s2 | native | 41.6 |  |  |
| T_MaxPool_ceil | native | 38.4 |  |  |
| T_Mean_C | native | 36.3 |  |  |
| T_Mish | native | 32.2 |  |  |
| T_Norm_C | native | 44.0 |  |  |
| T_PixelShuffle | native | 39.0 |  |  |
| T_PixelUnshuffle | native | 39.0 |  |  |
| T_Pow05 | native | 50.3 |  |  |
| T_Pow2 | native | 33.1 |  |  |
| T_QuickGELU | native | 35.2 |  |  |
| T_RMSNorm_C | native | 36.7 |  |  |
| T_ReLU6 | native | 38.3 |  |  |
| T_Reciprocal | native | 25.9 |  |  |
| T_Roll_W | native | 39.0 |  |  |
| T_Rsqrt | native | 35.6 |  |  |
| T_SiLU | native | 33.0 |  |  |
| T_Sigmoid_gate | native | 33.8 |  |  |
| T_Softmax_C | native | 34.5 |  |  |
| T_Softmax_W | native | 33.1 |  |  |
| T_Softplus | native | 38.1 |  |  |
| T_Sqrt | native | 50.3 |  |  |
| T_Square | native | 33.1 |  |  |
| T_Std_HW | native | 49.9 |  |  |
| T_Tanhshrink | native | 33.5 |  |  |
| T_Var_C | native | 39.3 |  |  |
| T_ViT_block | native | 32.7 |  |  |
| T_ZeroPad | native | 39.0 |  |  |
| Tanh | native | 35.5 |  |  |
| Tile_W | native | 36.4 |  |  |
| Transpose_HW | native | 39.0 |  |  |
| baseline | native | 39.0 |  |  |

## Fixed by rewrite or PWL (57 cases)
| case | status | chip dB | fix | after dB |
|---|---|---|---|---|
| Acos | FAIL@parse |  | pwl 9p none (2 APU) | 49.1 |
| Acosh | FAIL@parse |  | pwl 9p none (1 APU) | 49.7 |
| ArgMin_C | FAIL@parse |  | compat rewrite | 20.0 |
| Asin | FAIL@parse |  | pwl 9p none (1 APU) | 42.6 |
| Asinh | FAIL@parse |  | pwl 9p none (1 APU) | 36.3 |
| Atan | FAIL@parse |  | pwl 9p none (1 APU) | 36.1 |
| Atanh | FAIL@parse |  | pwl 9p none (1 APU) | 39.0 |
| Ceil | FAIL@parse |  | pwl step (2 APU) | 21.6 |
| Celu | FAIL@parse |  | pwl 9p none (1 APU) | 36.9 |
| Cos | FAIL@parse |  | pwl 9p none (2 APU) | 26.4 |
| Cosh | FAIL@parse |  | pwl 9p none (2 APU) | 37.3 |
| Erf | FAIL@parse |  | pwl 9p none (1 APU) | 36.5 |
| Expand_W | FAIL@parse |  | compat rewrite | 36.4 |
| Floor | SILENT-WRONG | 8.9 | pwl 9p none (2 APU) | 21.6 |
| Gelu_op20 | FAIL@parse |  | compat rewrite | 34.6 |
| Gelu_tanh_op20 | FAIL@parse |  | compat rewrite | 34.6 |
| Gelu_tanh_pattern | FAIL@parse |  | compat rewrite | 34.6 |
| Mish_op18 | FAIL@parse |  | compat rewrite (fold) | 32.2 |
| Pad_edge | SILENT-WRONG | 8.3 | compat rewrite | 39.0 |
| Pad_reflect | SILENT-WRONG | 8.3 | compat rewrite | 39.0 |
| Pow15 | FAIL@parse |  | compat rewrite | 48.1 |
| Pow3 | FAIL@parse |  | compat rewrite | 40.1 |
| Pow_m05 | native-lossy | -10.7 | compat rewrite | 35.6 |
| ReduceL1_C | FAIL@parse |  | compat rewrite | 46.0 |
| ReduceL1_W | FAIL@parse |  | compat rewrite | 47.4 |
| ReduceMax_HW | FAIL@parse |  | compat rewrite | 48.0 |
| ReduceMax_W | FAIL@parse |  | compat rewrite | 44.0 |
| ReduceMin_HW | FAIL@parse |  | compat rewrite | 47.2 |
| ReduceMin_W | FAIL@parse |  | compat rewrite | 43.7 |
| Round | FAIL@parse |  | pwl 9p none (2 APU) | 22.4 |
| Selu | FAIL@parse |  | pwl 9p none (1 APU) | 36.7 |
| Shrink | FAIL@parse |  | compat rewrite | 30.6 |
| Sign | FAIL@parse |  | pwl 9p none (1 APU) | 14.9 |
| Sin | FAIL@parse |  | pwl 9p none (2 APU) | 28.1 |
| Sinh | FAIL@parse |  | pwl 9p none (1 APU) | 40.0 |
| Split_C | FAIL@parse |  | compat rewrite | 36.0 |
| T_Amax_HW | FAIL@parse |  | compat rewrite | 48.0 |
| T_Amax_W | FAIL@parse |  | compat rewrite | 44.0 |
| T_Amin_W | FAIL@parse |  | compat rewrite | 43.7 |
| T_Atan | FAIL@parse |  | pwl 9p none (1 APU) | 36.1 |
| T_CELU | FAIL@parse |  | pwl 9p none (1 APU) | 36.9 |
| T_ConvTranspose_k3s2 | SILENT-WRONG | -3.0 | compat rewrite | 34.2 |
| T_Cos | FAIL@parse |  | pwl 9p none (2 APU) | 26.4 |
| T_Erf | FAIL@parse |  | pwl 9p none (1 APU) | 36.5 |
| T_Floor | SILENT-WRONG | 8.9 | pwl 9p none (2 APU) | 21.6 |
| T_GELU_tanh | FAIL@compile | 34.7 | compat rewrite | 34.6 |
| T_Max_tensor | native-lossy | 10.7 | compat rewrite | 34.7 |
| T_Pow3 | FAIL@parse |  | compat rewrite | 40.1 |
| T_ReflectPad | SILENT-WRONG | 8.3 | compat rewrite | 39.0 |
| T_ReplicatePad | SILENT-WRONG | 8.3 | compat rewrite | 39.0 |
| T_Round | FAIL@parse |  | pwl 9p none (2 APU) | 22.4 |
| T_SELU | FAIL@parse |  | pwl 9p none (1 APU) | 36.7 |
| T_Sign | FAIL@parse |  | pwl 9p none (1 APU) | 14.9 |
| T_Sin | FAIL@parse |  | pwl 9p none (2 APU) | 28.1 |
| T_Softshrink | FAIL@parse |  | compat rewrite | 31.1 |
| Tan | FAIL@parse |  | pwl 9p none (1 APU) | 42.5 |
| ThresholdedRelu | FAIL@parse |  | pwl exact (1 APU) | 32.9 |

## Still open (60 cases)
| case | status | chip dB | fix | after dB |
|---|---|---|---|---|
| ArgMax_C | native-lossy | 21.4 |  |  |
| Attention_pattern | native-lossy | 18.0 |  |  |
| CumSum_W | FAIL@parse |  |  |  |
| DFT_W | FAIL@parse |  |  |  |
| Einsum_bchw | FAIL@parse |  |  |  |
| Equal | FAIL@parse |  |  |  |
| Exp | native-lossy | 23.6 |  |  |
| GRU_W | FAIL@parse |  |  |  |
| Gelu_erf_pattern | FAIL@parse |  |  |  |
| Gemm_flat | native-lossy | 8.3 |  |  |
| GlobalAvgPool | native-lossy | 11.0 |  |  |
| Greater_where | FAIL@parse |  |  |  |
| GridSample | FAIL@parse |  |  |  |
| GroupNorm_op18 | FAIL@parse |  |  |  |
| Hardmax_C | FAIL@parse |  |  |  |
| Hardmax_W | FAIL@parse |  |  |  |
| LRN | FAIL@parse |  |  |  |
| LSTM_W | FAIL@parse |  |  |  |
| LayerNorm_W | FAIL@parse |  |  |  |
| LpNorm_C | FAIL@parse |  |  |  |
| LpPool | FAIL@parse |  |  |  |
| MatMul_const_W | FAIL@parse |  |  |  |
| MatMul_tensors | FAIL@parse |  |  |  |
| MeanVarNorm | FAIL@parse |  |  |  |
| Mod | FAIL@parse |  |  |  |
| Pow_tensors | FAIL@parse |  |  |  |
| RMSNorm_pattern | FAIL@parse |  |  |  |
| RNN_W | FAIL@parse |  |  |  |
| ReduceL1_HW | FAIL@parse |  |  |  |
| ReduceLogSumExp_C | FAIL@parse |  |  |  |
| ReduceLogSumExp_HW | FAIL@parse |  |  |  |
| ReduceLogSumExp_W | FAIL@parse |  |  |  |
| ReduceMean_HW | native-lossy | 11.0 |  |  |
| ReduceProd_C | FAIL@parse |  |  |  |
| ReduceProd_HW | FAIL@parse |  |  |  |
| ReduceProd_W | FAIL@parse |  |  |  |
| Resize2x_bicubic | FAIL@parse |  |  |  |
| Resize2x_bilinear | FAIL@emulate |  |  |  |
| Resize2x_bilinear_ac | FAIL@emulate |  |  |  |
| Resize_down_bilinear | FAIL@emulate |  |  |  |
| RoiAlign | FAIL@parse |  |  |  |
| STFT_pattern_conv | FAIL@parse |  |  |  |
| SpaceToDepth | FAIL@parse |  |  |  |
| T_Argmax_C | native-lossy | 21.4 |  |  |
| T_Attention_manual | FAIL@parse |  |  |  |
| T_Attention_sdpa | FAIL@parse |  |  |  |
| T_Cumsum_W | FAIL@parse |  |  |  |
| T_Exp | native-lossy | 23.6 |  |  |
| T_Flip_W | FAIL@parse |  |  |  |
| T_GRU | FAIL@parse |  |  |  |
| T_GridSample | FAIL@parse |  |  |  |
| T_Hardshrink | FAIL@parse |  |  |  |
| T_Interp_down_bilinear | FAIL@emulate |  |  |  |
| T_LSTM | FAIL@parse |  |  |  |
| T_LocalRespNorm | FAIL@parse |  |  |  |
| T_Logsumexp_C | FAIL@parse |  |  |  |
| T_Upsample_bicubic | FAIL@parse |  |  |  |
| T_Upsample_bilinear | FAIL@emulate |  |  |  |
| T_Upsample_bilinear_ac | FAIL@emulate |  |  |  |
| TopK_C | FAIL@parse |  |  |  |

## Tools
```
python onnx_hailo_compat.py IN.onnx OUT.onnx            # exact rewrites + float equivalence check
python pwl_inject.py fit IN.onnx CALIB.npy FIT.json --ops Sin Erf [--pieces 9]
python pwl_inject.py mark IN.onnx FIT.json MARKED.onnx  # unsupported unary ops -> PWL placeholders
# DFC side, after runner.translate_onnx_model(MARKED.onnx): from pwl_inject import apply_pwl; apply_pwl(runner, FIT.json)
python probe.py run CASE OUT_DIR                        # measure one case (parse/quantize/bit-exact/compile)
```
