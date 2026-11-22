# orient preprocessing model. Precision = all16 (passed physical chip 2026-11-20 for the orient3 checkpoint).
QMODE=all16; NORM=rgb255; OPT=0; INPUT16=0; CHIP_FMT=uint8; FP_TOL=${FP_TOL:-1e-3}
r_export(){ venv; export CUDA_VISIBLE_DEVICES=$G0; $VENV_HF -u $CV/recipes/preproc_torch.py export orient $CKPT $OUT; }
r_prep(){ tenv; mv $OUT/model.onnx $OUT/model.raw.onnx; python -u $QAT/tools/ops/onnx_hailo_compat.py $OUT/model.raw.onnx $OUT/model.onnx; }
r_data(){ venv; export CUDA_VISIBLE_DEVICES=$G0; $VENV_HF -u $CV/recipes/preproc_torch.py data orient $CKPT $OUT; }
r_gate(){ venv; $VENV_HF -u $CV/recipes/preproc_gate.py orient $OUT $1 $CKPT; }
