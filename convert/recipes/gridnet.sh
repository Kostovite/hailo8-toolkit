# GridNet table structure (table_structure_final.pt layout: net + TableMergerV2). Chip = trunk (p2, p4) all16, input gray/255.
# Host = row/col/band heads, kind head (avgpool kept on host: the DFC avgpool quantization failed), TableMergerV2 on the chip features.
QMODE=all16; NORM=gray255; OPT=0; INPUT16=0; CHIP_FMT=uint8; FP_TOL=${FP_TOL:-2e-2}; export EMU_F16=1
GRID_CALIB_N=${GRID_CALIB_N:-2048}
r_export(){ venv; export CUDA_VISIBLE_DEVICES=$G0; $VENV_HF -u $CV/recipes/grid_torch.py export $CKPT $OUT; }
r_prep(){ tenv; mv $OUT/model.onnx $OUT/model.raw.onnx; python -u $QAT/tools/ops/onnx_hailo_compat.py $OUT/model.raw.onnx $OUT/model.onnx; }
r_data(){ ( henv; export CUDA_VISIBLE_DEVICES=99; "$HPY" -u $CV/recipes/grid_calib.py $GRID_CALIB_N $OUT/calib.npy ) || return 1
          venv; export CUDA_VISIBLE_DEVICES=$G0; $VENV_HF -u $CV/recipes/grid_torch.py data $CKPT $OUT && $VENV_HF -u $CV/recipes/grid_gate.py $CKPT $OUT torch; }
r_gate(){ venv; export CUDA_VISIBLE_DEVICES=$G0; $VENV_HF -u $CV/recipes/grid_gate.py $CKPT $OUT $1; }
FP_ROWS=32
