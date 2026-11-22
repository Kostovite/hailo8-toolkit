#!/bin/bash
# nano: parse the pe-spatial ONNX -> level 0 (int8 + 16-bit heads) -> compile -> HEF dump + benchmark
set -e
R=/mnt/storage/lib_project/models/Hailo_model/hailo_re; S=$R/small; cd $R; . env.sh
CUDA_VISIBLE_DEVICES=99 $HPY parse_y26.py $S/y26n_pespatial.onnx $S/y26n_pes_parsed.har 2>&1 | tail -1
CUDA_VISIBLE_DEVICES=99 $HPY - <<PY
from hailo_sdk_client import ClientRunner
hn = ClientRunner(har="$S/y26n_pes_parsed.har").get_hn_dict()
heads = [k.split("/")[-1] for k, v in hn["layers"].items()
         if any("one2one_cv" in o and o.endswith(".2/Conv") for o in v.get("original_names", []))]
assert len(heads) == 6, heads
pe = [k for k, v in hn["layers"].items() if any("pe_v" in o for o in v.get("original_names", []))]
open("$S/alls_a16heads_pes.txt", "w").write(f"quantization_param([{', '.join(heads)}], precision_mode=a8_w8_a16)\n")
print("heads", heads, "pe_v layers", pe)
PY
export CUDA_VISIBLE_DEVICES=1 TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONPATH=$R/dfc_patched TRAIN_GLOB="$R/data_v14sec/train/images/*" NMS=""
OPT_LEVEL=0 $HPY b2_levelN.py $S/y26n_pes_parsed.har $S/y26n_pes_L0a16.har 640 640 1024 $S/alls_a16heads_pes.txt > $S/quant_pes_L0.log 2>&1
echo "PES quantized"
CUDA_VISIBLE_DEVICES=99 $HPY compile_hef.py $S/y26n_pes_L0a16.har $S/y26n_pes_L0a16.hef > $S/compile_pes.log 2>&1
echo "PES compiled"
