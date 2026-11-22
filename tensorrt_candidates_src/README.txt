PRELIMINARY TENSORRT CANDIDATES, not a complete Docparse package.
TensorRT 10.13.3.9; built on RTX 3090. Plan files can be GPU, platform and runtime version specific.
Orient3 engine covers feature extractor only; host decision head required.
Layout engine covers backbone/encoder only; host decoder and reading-order path required.
Flat3D output head is FP32; remaining graph contains QAT Q/DQ.
See included logs for limited-sample parity/accuracy checks and scripts; full benchmarks, integration and deployment approval remain.
Not evidence of maximum possible accuracy. QAT checkpoints and original staged checkpoints included.
