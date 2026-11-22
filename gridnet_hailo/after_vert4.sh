#!/bin/bash
# Waits for gridnet_vert4 training, then: real-book PC run + audits (both GPUs), val compare (GPU 0) || Hailo conversion (GPU 1).
WV=/mnt/storage/lib_project/models/Dolphin/wordlid_vert; G=$WV/data/grid
H=/mnt/storage/lib_project/models/Hailo_model/gridnet_hailo
S=/mnt/storage/tmp/claude-1005/-mnt-storage-lib-project-models/a1a17afb-fc44-40ee-8b36-7e0de44cb907/scratchpad
GPY="env PYTHONNOUSERSITE=1 LD_LIBRARY_PATH=/mnt/storage/anaconda3/envs/general/lib /mnt/storage/anaconda3/envs/general/bin/python -s"
P=$(cat $WV/logs/train_vert4.pid)
while kill -0 $P 2>/dev/null; do sleep 60; done
NEW=$G/gridnet_vert4_best.pt
[ -f $NEW ] || { echo "no $NEW" > $H/logs/after_vert4.log; exit 1; }
bash $S/fullrun.sh v4book --ckpt $NEW
(
  cd $WV
  for v in g5val vtval vnval chval cvval bkval; do
    GRID_DROP_LANG="g5:zho_Hans,jpn_Jpan" CUDA_VISIBLE_DEVICES=0 $GPY eval_ckpts.py $v 1200 $G/gridnet_vert2_best.pt $NEW 2>&1 | grep " {"
  done > $H/logs/pc_compare_v4.log
) &
(
  cd $H
  $GPY export_onnx.py $NEW gridnet_vert4_chip.onnx > logs/export_vert4.log 2>&1
  . /mnt/storage/lib_project/hailo8-exact-qat/tools/env.sh
  export LD_PRELOAD=/mnt/storage/anaconda3/envs/hailo_env/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2:$LD_PRELOAD
  CUDA_VISIBLE_DEVICES=99 $HPY parse.py gridnet_vert4_chip.onnx gridnet_vert4_parsed.har > logs/parse_vert4.log 2>&1
  CK=$NEW CUDA_VISIBLE_DEVICES=1 TF_FORCE_GPU_ALLOW_GROWTH=true OPT_LEVEL=0 $HPY quantize.py gridnet_vert4_parsed.har v4_L0_cal8k.har 8192 alls_base.txt > logs/q_v4_L0_cal8k.log 2>&1
  CUDA_VISIBLE_DEVICES=99 $HPY /mnt/storage/lib_project/models/Hailo_model/hailo_re/compile_hef.py v4_L0_cal8k.har v4_L0_cal8k.hef > logs/compile_v4.log 2>&1
  echo EXIT $? >> logs/compile_v4.log
) &
wait
echo ALLDONE >> $H/logs/after_vert4.log
