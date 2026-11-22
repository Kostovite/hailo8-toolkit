"""ONNX default-domain operator coverage on Hailo-8 (DFC 3.33.1), from the measured probes.

Every op of onnx_ops.json["all"] (200 ops, latest opset) gets one status. Numbers are the op-alone 8-bit probe
SQNR (floor 39 dB; in a model the op costs ~0 dB, HOST_VS_CHIP.md). Writes ONNX_COVERAGE.md.

  native   parses and runs on the chip as exported
  rewrite  on the chip after onnx_hailo_compat / compat2 / pwl_inject (verified bit-exact, compiles)
  lossy    on the chip but < 25 dB op-alone at 8 bit (step functions, one-hots, products)
  host     no on-chip form: host op at head/tail, or split + StagedRunner in the middle (host_split.py)
  folded   no runtime op: constant-folded or removed by the exporter / onnxsim (shapes, casts, windows, ...)
  open     should be possible on the chip, not working yet
  chip     batch 3 rewrites (compat3.py, cases3.py), measured at 8 and 16 bit
"""
import json
import os

S = {}


def put(status, ops, note=""):
    for o in ops.split():
        S[o] = (status, note)


put("native", "Abs Add AveragePool BatchNormalization Clip Concat Conv ConvTranspose DepthToSpace Div Elu Gather "
    "GlobalAveragePool GlobalMaxPool HardSigmoid HardSwish LeakyRelu Log LogSoftmax MaxPool Mul Neg PRelu "
    "Reciprocal ReduceL2 ReduceMean ReduceSum ReduceSumSquare Relu Reshape Sigmoid Slice Softmax Softplus Softsign "
    "Sqrt Sub Tanh Tile Transpose InstanceNormalization LayerNormalization Exp Gemm Max Min")
S["Exp"] = ("native", "23.6 dB alone (native fit); exp(x/2)^2 or 16-bit improve it")
S["Gemm"] = ("native", "8-11 dB after a 1x1 global pool (precision collapse); fine elsewhere")
S["Max"] = ("rewrite", "tensor Max is SILENT-WRONG natively (10.7 dB) -> a + Relu(b - a) 34.7 dB")
S["Min"] = ("rewrite", "a - Relu(a - b) 34.0 dB")
S["Pad"] = ("rewrite", "constant native; reflect/edge SILENT-WRONG -> Slice + Concat 39.0 dB")
S["Resize"] = ("native", "nearest/bilinear native (bilinear: no bit-exact emulation); cubic -> host")
S["MatMul"] = ("native", "attention MatMul in the nn.MultiheadAttention token layout; const over W -> rewrite 34.8")
S["Split"] = ("rewrite", "-> Slices 36.0 dB")
S["Expand"] = ("rewrite", "-> Tile 36.4 dB")
S["ArgMin"] = ("lossy", "ArgMax(-x) 20.0 dB (= ArgMax)")
S["ArgMax"] = ("lossy", "native, 21.4 dB (8-bit index output)")
for o, db in {"Acos": 49.1, "Acosh": 49.7, "Asin": 42.6, "Asinh": 36.3, "Atan": 36.1, "Atanh": 39.0, "Celu": 36.9,
              "Cos": 26.4, "Cosh": 37.3, "Erf": 36.5, "Selu": 36.7, "Sin": 28.1, "Sinh": 40.0, "Tan": 42.5}.items():
    S[o] = ("rewrite", f"PWL {db} dB alone at 8 bit, ~73 dB in 16 bit")
S["ThresholdedRelu"] = ("rewrite", "exact PWL")
S["Shrink"] = ("rewrite", "x - clip(x) 30.6 dB")
S["Pow"] = ("rewrite", "x^-0.5 SILENT-WRONG natively -> inv_sqrt 35.6; x^3 40.1; x^1.5 48.1; tensor^tensor 40.0")
S["ReduceL1"] = ("rewrite", "Abs + ReduceSum 46-47 dB")
S["ReduceMax"] = ("rewrite", "C native; W/HW -> MaxPool 44-48 dB")
S["ReduceMin"] = ("rewrite", "-max(-x) 43.7-47.2 dB")
S["ReduceLogSumExp"] = ("rewrite", "max + log(sum(exp(x - max))) 46.7-49 dB")
S["Gelu"] = ("rewrite", "op20 erf/tanh and any node order -> native gelu 34.6 dB")
S["Mish"] = ("rewrite", "op18 -> pattern 32.2 dB")
S["LRN"] = ("rewrite", "band 1x1 conv of x^2 39.0 dB")
S["CumSum"] = ("rewrite", "log-step shift-and-add 30.1 dB")
S["LpNormalization"] = ("rewrite", "x / ReduceL2 36.7 dB")
S["LpPool"] = ("rewrite", "sqrt(k AvgPool(x^2)) 35.3 dB")
S["MeanVarianceNormalization"] = ("rewrite", "InstanceNorm 36.1 dB")
S["GroupNormalization"] = ("rewrite", "Reshape -> InstanceNorm -> affine 33.3 dB")
S["Where"] = ("rewrite", "b + m (a - b) 32.1 dB")
S["Greater"] = ("rewrite", "0/1 arithmetic (Hardshrink 33.6 dB)")
S["Less"] = ("rewrite", "0/1 arithmetic")
S["Or"] = ("rewrite", "0/1 arithmetic (max of masks)")
S["RNN"] = ("native", "28.8 dB, unrolled")
S["LSTM"] = ("host", "<= 16 steps parse (24 steps quantize 23.2 dB but compile runs out of resources)")
S["GRU"] = ("host", "quantizes 11.4 dB, compiler crash")
S["Floor"] = ("lossy", "SILENT-WRONG natively (dropped) -> PWL staircase 21.6 dB")
S["Round"] = ("lossy", "PWL staircase 22.4 dB")
S["Sign"] = ("lossy", "PWL step 14.9 dB")
S["Ceil"] = ("lossy", "PWL staircase")
S["Hardmax"] = ("lossy", "ramp 6-9 dB -> host")
S["ReduceProd"] = ("lossy", "exp(sum log x) 8.9 dB -> host")
put("host", "TopK GridSample RoiAlign")
S["TopK"] = ("host", "StagedRunner 1,045-1,102 FPS vs 106 per-frame (test net)")
S["GridSample"] = ("host", "StagedRunner 896-970 FPS vs 132 per-frame (test net)")
put("folded", "Cast Constant ConstantOfShape Flatten Identity Shape Squeeze Unsqueeze",
    "structural: folded / absorbed at parse")
# ---- batch 3 (cases3.py + compat3.py, c3/ = 8 bit, c3_16/ = 16-bit max precision; every case compiles)
def chip(ops, note):
    put("chip", ops, note)


chip("Sum", "Add chain 34.4 / 61.8 dB (8 / 16 bit)")
chip("Mean", "Add chain * 1/n 34.3 / 66.7")
chip("Swish", "x * Sigmoid(a x) 35.2 / 40.3")
chip("GreaterOrEqual LessOrEqual Not", "0/1 masks 32.1 (= the 8-bit-input oracle)")
chip("And Xor Or", "linear 0/1 forms (Relu(m1+m2-1), |m1-m2|) 13.2 = the 8-bit-input oracle 12.8 for a (-1, 1) window")
chip("IsNaN IsInf", "all-false on finite tensors; Where(IsNaN) -> branch: 39.0 / 81.2")
chip("GlobalLpPool", "p=1 / 2 -> ReduceL1 / L2: 53.9 / 76.3, 50.8 / 51.0")
chip("ReduceLogSum", "Log(ReduceSum) 54.1 / 69.7")
chip("CumProd", "log-step shift-and-multiply 27.9 / 28.0 (ew_mult has no 16-bit mode)")
chip("Trilu", "constant 0/1 mask Mul 36.1")
chip("RMSNormalization", "torch pattern 34.2 / 35.7")
put("lossy", "Attention", "MatMul/Softmax decomposition 15.3 / 15.4 (mask: additive, head-tiled; Hailo matmul + "
    "softmax have no 16-bit) -> QAT recovers it in practice")
chip("RotaryEmbedding", "x cos + rotate_half(x) sin 33.5 / 34.8 (interleaved 33.7 / 34.7)")
chip("LinearAttention", "'linear' rule: scale (QK^T * causal) V with Relu split 27.5 / 29.1; gated/delta -> host")
chip("GatherElements GatherND", "const indices -> channel Gather 36.3 / 47.7; dynamic indices -> host")
chip("ScatterND ScatterElements TensorScatter", "const contiguous block -> Slice + Concat 35.1 / 58.9 (TensorScatter "
     "76.8); dynamic indices -> host")
put("lossy", "OneHot", "Less windows + Sub 15.4 (indices from 8-bit-only ArgMax); any index source works")
chip("Compress", "const condition -> Gather 36.2 / 52.4")
chip("CenterCropPad", "Slice / Pad 36.2-39.0 / 78.4-81.2")
chip("ReverseSequence", "reversed slices + Concat 39.0 / 81.2 (exact)")
chip("Col2Im", "Reshape + one-hot ConvTranspose 39.0 / 81.2 (exact)")
put("lossy", "MaxUnpool", "nearest up * (1 - (x - up < 0)) 20.6 (window ties at 8-bit input)")
chip("DeformConv", "const zero offsets -> Conv 33.5 / 72.3; learned offsets (bilinear sampling) -> host")
chip("CausalConvWithState", "rank-3 causal depthwise Conv1d (+SiLU) 31.2 / 33.3")
chip("STFT", "windowed cos/sin strided conv, complex axis folded into the consumer 31.6 / 32.1")
chip("DFT", "MatMul with cos/sin -> per-column depthwise convs 35.8 / 77.0")
chip("Einsum", "const channel contraction -> 1x1 Conv 33.7 / 63.5; broadcast product 27.5 / 29.2; tensor x tensor "
     "contraction -> broadcast Mul + ReduceSum 17.3")
chip("QuantizeLinear DequantizeLinear QLinearConv QLinearMatMul DynamicQuantizeLinear MatMulInteger ConvInteger",
     "QDQ / QOperator / dynamic-quant -> float graph, requantized by the DFC: 34.2-34.5 / 63.1-63.5")
chip("SpaceToDepth", "torch order + channel Gather 39.0 / 81.2 (exact)")
chip("Mod", "x - m Trunc(x/m), Trunc = one PWL staircase 21.8 / 76.9")
chip("Equal", "Round/Floor/Ceil folded into the window 16.7 / 37.2")
chip("Floor", "PWL staircase 21.6 / 41.6 (natively SILENT-WRONG: dropped)")
chip("Round", "PWL staircase 22.2 / 45.3")
chip("Ceil", "PWL staircase 21.5 / 41.6")
chip("Sign", "PWL step 14.9 / 37.9")
chip("Hardmax", "steep ramp 8.9 / 48.1 (16 bit); OneHot(ArgMax) 15.4")
chip("ReduceProd", "pairwise Mul tree (any sign) 40.5 / 40.3")
put("lossy", "ArgMax ArgMin", "native 21.4 / 20.0: ArgMax is 8-bit only (near-ties at 8-bit input)")
put("folded", "Dropout CastLike Range Size EyeLike BlackmanWindow HammingWindow HannWindow MelWeightMatrix",
    "constant-folded (Folded_misc 33.3 / 35.5, EyeLike 36.6; Blackman + Mel -> one const Mul)")
_HOST_NOTE = "no chip form: host (onnxruntime / onnx reference), staged mode in mid-network; conformance tests pass"
put("host", "TopK GridSample RoiAlign MaxRoiPool GRU NonZero NonMaxSuppression Det Unique AffineGrid Loop Scan",
    _HOST_NOTE)
S["LSTM"] = ("host", "<= 16 steps on chip (unrolled); longer: host")
S["RNN"] = ("native", "28.8 dB, unrolled")
S["If"] = ("folded", "constant condition folded; dynamic -> host")
put("host", "Bernoulli BitCast BitShift BitwiseAnd BitwiseNot BitwiseOr BitwiseXor ConcatFromSequence ImageDecoder "
    "Multinomial NegativeLogLikelihoodLoss Optional OptionalGetElement OptionalHasElement RandomNormal "
    "RandomNormalLike RandomUniformLike RegexFullMatch SequenceAt SequenceConstruct SequenceEmpty SequenceErase "
    "SequenceInsert SequenceLength SequenceMap SoftmaxCrossEntropyLoss SplitToSequence StringConcat "
    "StringNormalizer StringSplit TfIdfVectorizer", "not image-tensor math: host (conformance tests pass)")
S["RandomUniform"] = ("host", "random generator: host")
for _o in ("Floor", "Round", "Ceil", "Sign"):
    pass

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    allops = json.load(open(os.path.join(here, "onnx_ops.json")))["all"]
    missing = [o for o in allops if o not in S]
    extra = [o for o in S if o not in allops]
    assert not missing and not extra, (missing, extra)
    order = ["native", "chip", "rewrite", "lossy", "folded", "host"]
    cnt = {k: sum(1 for o in allops if S[o][0] == k) for k in order}
    tot = len(allops)
    onchip = cnt["native"] + cnt["chip"] + cnt["rewrite"]
    lines = ["# ONNX operator coverage on Hailo-8 (DFC 3.33.1, measured)", "",
             f"All **{tot}** default-domain ONNX ops (opset 27) are supported: each runs on the chip, folds away, or "
             "runs on the host (staged mode mid-network). dB = op-alone probe SQNR at 8 bit / 16-bit max precision "
             "(x -> op -> identity conv; the 8-bit input floor is 39 dB, 16 bit 81 dB).", "",
             "| status | ops | share |", "|---|---|---|"]
    for k in order:
        lines.append(f"| {k} | {cnt[k]} | {100 * cnt[k] / tot:.0f}% |")
    lines += ["", f"- **On the chip: {onchip + cnt['lossy']}/{tot} = {100 * (onchip + cnt['lossy']) / tot:.0f}%** "
              f"({onchip} at >= 25 dB at the best precision, {cnt['lossy']} limited: {', '.join(o for o in allops if S[o][0] == 'lossy')}).",
              f"- Folded at conversion: {cnt['folded']}; host-side: {cnt['host']} (no on-chip form or not tensor math).",
              "- native = parses as exported; chip/rewrite = onnx_hailo_compat + compat2/3 + pwl_inject;"
              " lossy = on chip but < 25 dB even at 16 bit (host also possible).", "",
              "| op | status | note |", "|---|---|---|"]
    for k in order:
        for o in sorted(o for o in allops if S[o][0] == k):
            lines.append(f"| {o} | {k} | {S[o][1]} |")
    open(os.path.join(here, "ONNX_COVERAGE.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[:16]))
