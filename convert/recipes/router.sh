# RegionRouter (pooled encoder feature 1024 -> kind 6 / body 3 / lang 21). Host: pooling, /router_scale, overrides.
# Precision: all 16-bit INCLUDING the input layer, plus per-feature host scaling. Root cause of the old .87 kind agreement:
# 8-bit input with one scale over [-16.9, 3.6] while feature std is .27.
A=/mnt/storage/lib_project/models/moe/runs/codex_fusion_20261119/hailo_current_20261119/aux_models
ROUTER_TRAIN=${ROUTER_TRAIN:-$A/router_more16_train.npy}; ROUTER_EVAL=${ROUTER_EVAL:-$A/router_more16_val.npy}
QMODE=all16; NORM=none; OPT=0; INPUT16=1; CHIP_FMT=float32; FP_TOL=1e-3
r_export(){ venv; $VENV_HF -u $CV/recipes/router_export.py $CKPT $ROUTER_TRAIN $OUT; }
r_prep(){ cp $OUT/model.onnx $OUT/model.raw.onnx; echo "no rewrite needed (Gemm/ReLU only)"; }
r_data(){ venv; $VENV_HF -u $CV/recipes/router_data.py $CKPT $ROUTER_TRAIN $ROUTER_EVAL $OUT; }
r_gate(){ python3 $CV/recipes/router_gate.py $OUT $1; }
