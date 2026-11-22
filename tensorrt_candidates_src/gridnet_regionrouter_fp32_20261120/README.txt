GridNet and RegionRouter FP32 TensorRT baseline candidates, copied 2026-11-20.
Source: isolated RTX 3090 workspace; built with TensorRT 10.13.3.9. Engines are platform/GPU/TensorRT-version dependent.

RegionRouter: router_fp32.engine consumes pooled encoder features (B,1024). On 100 held-out vectors, all three head argmax decisions matched FP32 reference. This is not an end-to-end reader evaluation, and host pooling/routing overrides remain necessary. It is FP32, not QAT.

GridNet: grid_fp32.engine is only the trunk p2 features and kind head. Remaining model heads, table merger, and postprocessing are not included in the engine. On 30 held-out images, p2 relative MAE was approximately 0.00390 and kind decisions matched 30/30. No end-to-end table metric or merger validation has been performed. TF32 was enabled in this candidate. It is FP32, not QAT.

The original source checkpoints, ONNX, build/test scripts and available logs accompany these candidates. Checkpoints and artifacts should not be presented as fully validated or maximum-accuracy deployment models. SHA256SUMS verifies files copied from the source.
