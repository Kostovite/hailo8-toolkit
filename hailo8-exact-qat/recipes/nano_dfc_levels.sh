#!/bin/bash
# Lane A (GPU 0): YOLO26n@640 -> ONNX -> HAR -> DFC level 0 / 2 / 4 (16-bit heads), each scored on exact arithmetic.
set -e
R=/mnt/storage/lib_project/models/Hailo_model/hailo_re; S=$R/small; P=/mnt/storage/anaconda3/envs/general
PT=$S/y26n_v14sec/weights/best.pt
cd $S
PYTHONNOUSERSITE=1 env -u LD_PRELOAD LD_LIBRARY_PATH=$P/lib CUDA_VISIBLE_DEVICES="" $P/bin/python - <<PY
from ultralytics import YOLO
YOLO("$PT").export(format="onnx", imgsz=640, dynamic=False, opset=13, simplify=True)
PY
cd $R; . env.sh; export PT
CUDA_VISIBLE_DEVICES=99 $HPY parse_y26.py $S/y26n_v14sec/weights/best.onnx $S/y26n_parsed.har | tail -1
CUDA_VISIBLE_DEVICES=99 $HPY - <<PY
import json
from hailo_sdk_client import ClientRunner
hn = ClientRunner(har="$S/y26n_parsed.har").get_hn_dict()
heads = [k.split("/")[-1] for k, v in hn["layers"].items()
         if any("one2one_cv" in o and o.endswith(".2/Conv") for o in v.get("original_names", []))]
assert len(heads) == 6, heads
open("$S/alls_a16heads_n.txt", "w").write(f"quantization_param([{', '.join(heads)}], precision_mode=a8_w8_a16)\n")
print("heads", heads)
PY
echo "LANE_A parsed"
export CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONPATH=$R/dfc_patched TRAIN_GLOB="$R/data_v14sec/train/images/*" NMS=""
for L in 0 2 4; do
  s=$(date +%s)
  OPT_LEVEL=$L $HPY b2_levelN.py $S/y26n_parsed.har $S/y26n_L${L}a16.har 640 640 1024 $S/alls_a16heads_n.txt > $S/quant_L$L.log 2>&1
  echo "LANE_A level $L quantized in $(( $(date +%s) - s )) s"
  EVAL_GPUS="0" ./har_pipeline.sh $S/y26n_L${L}a16.har n_L${L}a16 0 2>&1 | grep -E "exact|tag" | sed "s/^/LANE_A L$L /"
  [ $L = 0 ] && touch $S/L0_ready
done
echo "LANE_A DONE"
