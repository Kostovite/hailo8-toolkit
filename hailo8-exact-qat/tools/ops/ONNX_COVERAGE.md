# ONNX operator coverage on Hailo-8 (DFC 3.33.1, measured)

All **200** default-domain ONNX ops (opset 27) are supported: each runs on the chip, folds away, or runs on the host (staged mode mid-network). dB = op-alone probe SQNR at 8 bit / 16-bit max precision (x -> op -> identity conv; the 8-bit input floor is 39 dB, 16 bit 81 dB).

| status | ops | share |
|---|---|---|
| native | 47 | 24% |
| chip | 48 | 24% |
| rewrite | 37 | 18% |
| lossy | 5 | 2% |
| folded | 18 | 9% |
| host | 45 | 22% |

- **On the chip: 137/200 = 68%** (132 at >= 25 dB at the best precision, 5 limited: ArgMax, ArgMin, Attention, MaxUnpool, OneHot).
- Folded at conversion: 18; host-side: 45 (no on-chip form or not tensor math).
- native = parses as exported; chip/rewrite = onnx_hailo_compat + compat2/3 + pwl_inject; lossy = on chip but < 25 dB even at 16 bit (host also possible).

| op | status | note |
|---|---|---|
| Abs | native |  |
| Add | native |  |
| AveragePool | native |  |
| BatchNormalization | native |  |
| Clip | native |  |
| Concat | native |  |
| Conv | native |  |
| ConvTranspose | native |  |
| DepthToSpace | native |  |
| Div | native |  |
| Elu | native |  |
| Exp | native | 23.6 dB alone (native fit); exp(x/2)^2 or 16-bit improve it |
| Gather | native |  |
| Gemm | native | 8-11 dB after a 1x1 global pool (precision collapse); fine elsewhere |
| GlobalAveragePool | native |  |
| GlobalMaxPool | native |  |
| HardSigmoid | native |  |
| HardSwish | native |  |
| InstanceNormalization | native |  |
| LayerNormalization | native |  |
| LeakyRelu | native |  |
| Log | native |  |
| LogSoftmax | native |  |
| MatMul | native | attention MatMul in the nn.MultiheadAttention token layout; const over W -> rewrite 34.8 |
| MaxPool | native |  |
| Mul | native |  |
| Neg | native |  |
| PRelu | native |  |
| RNN | native | 28.8 dB, unrolled |
| Reciprocal | native |  |
| ReduceL2 | native |  |
| ReduceMean | native |  |
| ReduceSum | native |  |
| ReduceSumSquare | native |  |
| Relu | native |  |
| Reshape | native |  |
| Resize | native | nearest/bilinear native (bilinear: no bit-exact emulation); cubic -> host |
| Sigmoid | native |  |
| Slice | native |  |
| Softmax | native |  |
| Softplus | native |  |
| Softsign | native |  |
| Sqrt | native |  |
| Sub | native |  |
| Tanh | native |  |
| Tile | native |  |
| Transpose | native |  |
| And | chip | linear 0/1 forms (Relu(m1+m2-1), |m1-m2|) 13.2 = the 8-bit-input oracle 12.8 for a (-1, 1) window |
| CausalConvWithState | chip | rank-3 causal depthwise Conv1d (+SiLU) 31.2 / 33.3 |
| Ceil | chip | PWL staircase 21.5 / 41.6 |
| CenterCropPad | chip | Slice / Pad 36.2-39.0 / 78.4-81.2 |
| Col2Im | chip | Reshape + one-hot ConvTranspose 39.0 / 81.2 (exact) |
| Compress | chip | const condition -> Gather 36.2 / 52.4 |
| ConvInteger | chip | QDQ / QOperator / dynamic-quant -> float graph, requantized by the DFC: 34.2-34.5 / 63.1-63.5 |
| CumProd | chip | log-step shift-and-multiply 27.9 / 28.0 (ew_mult has no 16-bit mode) |
| DFT | chip | MatMul with cos/sin -> per-column depthwise convs 35.8 / 77.0 |
| DeformConv | chip | const zero offsets -> Conv 33.5 / 72.3; learned offsets (bilinear sampling) -> host |
| DequantizeLinear | chip | QDQ / QOperator / dynamic-quant -> float graph, requantized by the DFC: 34.2-34.5 / 63.1-63.5 |
| DynamicQuantizeLinear | chip | QDQ / QOperator / dynamic-quant -> float graph, requantized by the DFC: 34.2-34.5 / 63.1-63.5 |
| Einsum | chip | const channel contraction -> 1x1 Conv 33.7 / 63.5; broadcast product 27.5 / 29.2; tensor x tensor contraction -> broadcast Mul + ReduceSum 17.3 |
| Equal | chip | Round/Floor/Ceil folded into the window 16.7 / 37.2 |
| Floor | chip | PWL staircase 21.6 / 41.6 (natively SILENT-WRONG: dropped) |
| GatherElements | chip | const indices -> channel Gather 36.3 / 47.7; dynamic indices -> host |
| GatherND | chip | const indices -> channel Gather 36.3 / 47.7; dynamic indices -> host |
| GlobalLpPool | chip | p=1 / 2 -> ReduceL1 / L2: 53.9 / 76.3, 50.8 / 51.0 |
| GreaterOrEqual | chip | 0/1 masks 32.1 (= the 8-bit-input oracle) |
| Hardmax | chip | steep ramp 8.9 / 48.1 (16 bit); OneHot(ArgMax) 15.4 |
| IsInf | chip | all-false on finite tensors; Where(IsNaN) -> branch: 39.0 / 81.2 |
| IsNaN | chip | all-false on finite tensors; Where(IsNaN) -> branch: 39.0 / 81.2 |
| LessOrEqual | chip | 0/1 masks 32.1 (= the 8-bit-input oracle) |
| LinearAttention | chip | 'linear' rule: scale (QK^T * causal) V with Relu split 27.5 / 29.1; gated/delta -> host |
| MatMulInteger | chip | QDQ / QOperator / dynamic-quant -> float graph, requantized by the DFC: 34.2-34.5 / 63.1-63.5 |
| Mean | chip | Add chain * 1/n 34.3 / 66.7 |
| Mod | chip | x - m Trunc(x/m), Trunc = one PWL staircase 21.8 / 76.9 |
| Not | chip | 0/1 masks 32.1 (= the 8-bit-input oracle) |
| Or | chip | linear 0/1 forms (Relu(m1+m2-1), |m1-m2|) 13.2 = the 8-bit-input oracle 12.8 for a (-1, 1) window |
| QLinearConv | chip | QDQ / QOperator / dynamic-quant -> float graph, requantized by the DFC: 34.2-34.5 / 63.1-63.5 |
| QLinearMatMul | chip | QDQ / QOperator / dynamic-quant -> float graph, requantized by the DFC: 34.2-34.5 / 63.1-63.5 |
| QuantizeLinear | chip | QDQ / QOperator / dynamic-quant -> float graph, requantized by the DFC: 34.2-34.5 / 63.1-63.5 |
| RMSNormalization | chip | torch pattern 34.2 / 35.7 |
| ReduceLogSum | chip | Log(ReduceSum) 54.1 / 69.7 |
| ReduceProd | chip | pairwise Mul tree (any sign) 40.5 / 40.3 |
| ReverseSequence | chip | reversed slices + Concat 39.0 / 81.2 (exact) |
| RotaryEmbedding | chip | x cos + rotate_half(x) sin 33.5 / 34.8 (interleaved 33.7 / 34.7) |
| Round | chip | PWL staircase 22.2 / 45.3 |
| STFT | chip | windowed cos/sin strided conv, complex axis folded into the consumer 31.6 / 32.1 |
| ScatterElements | chip | const contiguous block -> Slice + Concat 35.1 / 58.9 (TensorScatter 76.8); dynamic indices -> host |
| ScatterND | chip | const contiguous block -> Slice + Concat 35.1 / 58.9 (TensorScatter 76.8); dynamic indices -> host |
| Sign | chip | PWL step 14.9 / 37.9 |
| SpaceToDepth | chip | torch order + channel Gather 39.0 / 81.2 (exact) |
| Sum | chip | Add chain 34.4 / 61.8 dB (8 / 16 bit) |
| Swish | chip | x * Sigmoid(a x) 35.2 / 40.3 |
| TensorScatter | chip | const contiguous block -> Slice + Concat 35.1 / 58.9 (TensorScatter 76.8); dynamic indices -> host |
| Trilu | chip | constant 0/1 mask Mul 36.1 |
| Xor | chip | linear 0/1 forms (Relu(m1+m2-1), |m1-m2|) 13.2 = the 8-bit-input oracle 12.8 for a (-1, 1) window |
| Acos | rewrite | PWL 49.1 dB alone at 8 bit, ~73 dB in 16 bit |
| Acosh | rewrite | PWL 49.7 dB alone at 8 bit, ~73 dB in 16 bit |
| Asin | rewrite | PWL 42.6 dB alone at 8 bit, ~73 dB in 16 bit |
| Asinh | rewrite | PWL 36.3 dB alone at 8 bit, ~73 dB in 16 bit |
| Atan | rewrite | PWL 36.1 dB alone at 8 bit, ~73 dB in 16 bit |
| Atanh | rewrite | PWL 39.0 dB alone at 8 bit, ~73 dB in 16 bit |
| Celu | rewrite | PWL 36.9 dB alone at 8 bit, ~73 dB in 16 bit |
| Cos | rewrite | PWL 26.4 dB alone at 8 bit, ~73 dB in 16 bit |
| Cosh | rewrite | PWL 37.3 dB alone at 8 bit, ~73 dB in 16 bit |
| CumSum | rewrite | log-step shift-and-add 30.1 dB |
| Erf | rewrite | PWL 36.5 dB alone at 8 bit, ~73 dB in 16 bit |
| Expand | rewrite | -> Tile 36.4 dB |
| Gelu | rewrite | op20 erf/tanh and any node order -> native gelu 34.6 dB |
| Greater | rewrite | 0/1 arithmetic (Hardshrink 33.6 dB) |
| GroupNormalization | rewrite | Reshape -> InstanceNorm -> affine 33.3 dB |
| LRN | rewrite | band 1x1 conv of x^2 39.0 dB |
| Less | rewrite | 0/1 arithmetic |
| LpNormalization | rewrite | x / ReduceL2 36.7 dB |
| LpPool | rewrite | sqrt(k AvgPool(x^2)) 35.3 dB |
| Max | rewrite | tensor Max is SILENT-WRONG natively (10.7 dB) -> a + Relu(b - a) 34.7 dB |
| MeanVarianceNormalization | rewrite | InstanceNorm 36.1 dB |
| Min | rewrite | a - Relu(a - b) 34.0 dB |
| Mish | rewrite | op18 -> pattern 32.2 dB |
| Pad | rewrite | constant native; reflect/edge SILENT-WRONG -> Slice + Concat 39.0 dB |
| Pow | rewrite | x^-0.5 SILENT-WRONG natively -> inv_sqrt 35.6; x^3 40.1; x^1.5 48.1; tensor^tensor 40.0 |
| ReduceL1 | rewrite | Abs + ReduceSum 46-47 dB |
| ReduceLogSumExp | rewrite | max + log(sum(exp(x - max))) 46.7-49 dB |
| ReduceMax | rewrite | C native; W/HW -> MaxPool 44-48 dB |
| ReduceMin | rewrite | -max(-x) 43.7-47.2 dB |
| Selu | rewrite | PWL 36.7 dB alone at 8 bit, ~73 dB in 16 bit |
| Shrink | rewrite | x - clip(x) 30.6 dB |
| Sin | rewrite | PWL 28.1 dB alone at 8 bit, ~73 dB in 16 bit |
| Sinh | rewrite | PWL 40.0 dB alone at 8 bit, ~73 dB in 16 bit |
| Split | rewrite | -> Slices 36.0 dB |
| Tan | rewrite | PWL 42.5 dB alone at 8 bit, ~73 dB in 16 bit |
| ThresholdedRelu | rewrite | exact PWL |
| Where | rewrite | b + m (a - b) 32.1 dB |
| ArgMax | lossy | native 21.4 / 20.0: ArgMax is 8-bit only (near-ties at 8-bit input) |
| ArgMin | lossy | native 21.4 / 20.0: ArgMax is 8-bit only (near-ties at 8-bit input) |
| Attention | lossy | MatMul/Softmax decomposition 15.3 / 15.4 (mask: additive, head-tiled; Hailo matmul + softmax have no 16-bit) -> QAT recovers it in practice |
| MaxUnpool | lossy | nearest up * (1 - (x - up < 0)) 20.6 (window ties at 8-bit input) |
| OneHot | lossy | Less windows + Sub 15.4 (indices from 8-bit-only ArgMax); any index source works |
| BlackmanWindow | folded | constant-folded (Folded_misc 33.3 / 35.5, EyeLike 36.6; Blackman + Mel -> one const Mul) |
| Cast | folded | structural: folded / absorbed at parse |
| CastLike | folded | constant-folded (Folded_misc 33.3 / 35.5, EyeLike 36.6; Blackman + Mel -> one const Mul) |
| Constant | folded | structural: folded / absorbed at parse |
| ConstantOfShape | folded | structural: folded / absorbed at parse |
| Dropout | folded | constant-folded (Folded_misc 33.3 / 35.5, EyeLike 36.6; Blackman + Mel -> one const Mul) |
| EyeLike | folded | constant-folded (Folded_misc 33.3 / 35.5, EyeLike 36.6; Blackman + Mel -> one const Mul) |
| Flatten | folded | structural: folded / absorbed at parse |
| HammingWindow | folded | constant-folded (Folded_misc 33.3 / 35.5, EyeLike 36.6; Blackman + Mel -> one const Mul) |
| HannWindow | folded | constant-folded (Folded_misc 33.3 / 35.5, EyeLike 36.6; Blackman + Mel -> one const Mul) |
| Identity | folded | structural: folded / absorbed at parse |
| If | folded | constant condition folded; dynamic -> host |
| MelWeightMatrix | folded | constant-folded (Folded_misc 33.3 / 35.5, EyeLike 36.6; Blackman + Mel -> one const Mul) |
| Range | folded | constant-folded (Folded_misc 33.3 / 35.5, EyeLike 36.6; Blackman + Mel -> one const Mul) |
| Shape | folded | structural: folded / absorbed at parse |
| Size | folded | constant-folded (Folded_misc 33.3 / 35.5, EyeLike 36.6; Blackman + Mel -> one const Mul) |
| Squeeze | folded | structural: folded / absorbed at parse |
| Unsqueeze | folded | structural: folded / absorbed at parse |
| AffineGrid | host | no chip form: host (onnxruntime / onnx reference), staged mode in mid-network; conformance tests pass |
| Bernoulli | host | not image-tensor math: host (conformance tests pass) |
| BitCast | host | not image-tensor math: host (conformance tests pass) |
| BitShift | host | not image-tensor math: host (conformance tests pass) |
| BitwiseAnd | host | not image-tensor math: host (conformance tests pass) |
| BitwiseNot | host | not image-tensor math: host (conformance tests pass) |
| BitwiseOr | host | not image-tensor math: host (conformance tests pass) |
| BitwiseXor | host | not image-tensor math: host (conformance tests pass) |
| ConcatFromSequence | host | not image-tensor math: host (conformance tests pass) |
| Det | host | no chip form: host (onnxruntime / onnx reference), staged mode in mid-network; conformance tests pass |
| GRU | host | no chip form: host (onnxruntime / onnx reference), staged mode in mid-network; conformance tests pass |
| GridSample | host | no chip form: host (onnxruntime / onnx reference), staged mode in mid-network; conformance tests pass |
| ImageDecoder | host | not image-tensor math: host (conformance tests pass) |
| LSTM | host | <= 16 steps on chip (unrolled); longer: host |
| Loop | host | no chip form: host (onnxruntime / onnx reference), staged mode in mid-network; conformance tests pass |
| MaxRoiPool | host | no chip form: host (onnxruntime / onnx reference), staged mode in mid-network; conformance tests pass |
| Multinomial | host | not image-tensor math: host (conformance tests pass) |
| NegativeLogLikelihoodLoss | host | not image-tensor math: host (conformance tests pass) |
| NonMaxSuppression | host | no chip form: host (onnxruntime / onnx reference), staged mode in mid-network; conformance tests pass |
| NonZero | host | no chip form: host (onnxruntime / onnx reference), staged mode in mid-network; conformance tests pass |
| Optional | host | not image-tensor math: host (conformance tests pass) |
| OptionalGetElement | host | not image-tensor math: host (conformance tests pass) |
| OptionalHasElement | host | not image-tensor math: host (conformance tests pass) |
| RandomNormal | host | not image-tensor math: host (conformance tests pass) |
| RandomNormalLike | host | not image-tensor math: host (conformance tests pass) |
| RandomUniform | host | random generator: host |
| RandomUniformLike | host | not image-tensor math: host (conformance tests pass) |
| RegexFullMatch | host | not image-tensor math: host (conformance tests pass) |
| RoiAlign | host | no chip form: host (onnxruntime / onnx reference), staged mode in mid-network; conformance tests pass |
| Scan | host | no chip form: host (onnxruntime / onnx reference), staged mode in mid-network; conformance tests pass |
| SequenceAt | host | not image-tensor math: host (conformance tests pass) |
| SequenceConstruct | host | not image-tensor math: host (conformance tests pass) |
| SequenceEmpty | host | not image-tensor math: host (conformance tests pass) |
| SequenceErase | host | not image-tensor math: host (conformance tests pass) |
| SequenceInsert | host | not image-tensor math: host (conformance tests pass) |
| SequenceLength | host | not image-tensor math: host (conformance tests pass) |
| SequenceMap | host | not image-tensor math: host (conformance tests pass) |
| SoftmaxCrossEntropyLoss | host | not image-tensor math: host (conformance tests pass) |
| SplitToSequence | host | not image-tensor math: host (conformance tests pass) |
| StringConcat | host | not image-tensor math: host (conformance tests pass) |
| StringNormalizer | host | not image-tensor math: host (conformance tests pass) |
| StringSplit | host | not image-tensor math: host (conformance tests pass) |
| TfIdfVectorizer | host | not image-tensor math: host (conformance tests pass) |
| TopK | host | no chip form: host (onnxruntime / onnx reference), staged mode in mid-network; conformance tests pass |
| Unique | host | no chip form: host (onnxruntime / onnx reference), staged mode in mid-network; conformance tests pass |
