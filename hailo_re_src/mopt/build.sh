#!/bin/bash
# mopt/build.sh NAME ONNX GPU: parse -> level 0 + 16-bit heads (896, 1024 calib) -> default compile -> HEF dump
set -e
R=/mnt/storage/lib_project/models/Hailo_model/hailo_re; M=$R/mopt; N=$1; ONNX=$2; G=$3; cd $R; . env.sh
CUDA_VISIBLE_DEVICES=99 $HPY parse_y26.py $ONNX $M/${N}_parsed.har 2>&1 | tail -1
CUDA_VISIBLE_DEVICES=99 $HPY - <<PY
from hailo_sdk_client import ClientRunner
hn = ClientRunner(har="$M/${N}_parsed.har").get_hn_dict()
heads = [k.split("/")[-1] for k, v in hn["layers"].items()
         if any("one2one_cv" in o and o.endswith(".2/Conv") for o in v.get("original_names", []))]
assert len(heads) == 6, heads
open("$M/${N}_heads.txt", "w").write(f"quantization_param([{', '.join(heads)}], precision_mode=a8_w8_a16)\\n")
print("$N heads", heads)
PY
CUDA_VISIBLE_DEVICES=$G TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONPATH=$R/dfc_patched TRAIN_GLOB="$R/data_v14sec/train/images/*" NMS="" \
  OPT_LEVEL=0 $HPY b2_levelN.py $M/${N}_parsed.har $M/${N}_L0a16.har 896 896 1024 $M/${N}_heads.txt > $M/${N}_quant.log 2>&1
echo "$N quantized"
CUDA_VISIBLE_DEVICES=99 $HPY compile_hef.py $M/${N}_L0a16.har $M/${N}.hef > $M/${N}_compile.log 2>&1
echo "$N compiled"
/mnt/storage/anaconda3/envs/general/bin/python $R/hefre/hef_dump.py $M/${N}.hef --json $M/${N}_hef.json | head -3
