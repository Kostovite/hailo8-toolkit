#!/bin/bash
# Runs once gridnet_vert2 training has exited: PC comparison (GPU 0) and the Hailo conversion (GPU 1) in parallel.
G=/mnt/storage/lib_project/models/Dolphin/wordlid/data/grid; WV=/mnt/storage/lib_project/models/Dolphin/wordlid_vert
H=/mnt/storage/lib_project/models/Hailo_model/gridnet_hailo; T1=/mnt/storage/lib_project/models/Hailo_model/nlp/T1_out
GPY="env PYTHONNOUSERSITE=1 LD_LIBRARY_PATH=/mnt/storage/anaconda3/envs/general/lib /mnt/storage/anaconda3/envs/general/bin/python -s"
NEW=$G/gridnet_vert2_best.pt
[ -f $NEW ] || { echo "no $NEW" > $H/logs/after_vert2.log; exit 1; }
(
  cd $WV
  for v in g5val vtval vnval chval cvval; do
    GRID_DROP_LANG="g5:zho_Hans,jpn_Jpan" CUDA_VISIBLE_DEVICES=0 $GPY eval_ckpts.py $v 1200 $G/gridnet_vert1_best.pt $NEW 2>&1 | grep " {"
  done > $H/logs/pc_compare.log
  for ck in vert1 vert2; do      # real book, raw cuts (no ink fit): where do the model's own line cuts sit?
    CUDA_VISIBLE_DEVICES=0 $GPY scan_units.py $T1/pages $(seq -s, 3 4 338) $H/logs/book_$ck --ckpt $G/gridnet_${ck}_best.pt --clean --no-fit --no-viz > /dev/null 2>&1
    echo "== $ck" >> $H/logs/book_audit.log
    $GPY $H/tools_audit/cut_audit.py $H/logs/book_$ck/units.jsonl $T1/pages >> $H/logs/book_audit.log 2>&1
    $GPY $H/tools_audit/word_audit.py $H/logs/book_$ck/units.jsonl $T1/pages >> $H/logs/book_audit.log 2>&1
  done
) &
(
  cd $H
  $GPY export_onnx.py $NEW gridnet_vert2_chip.onnx > logs/export_vert2.log 2>&1
  . /mnt/storage/lib_project/hailo8-exact-qat/tools/env.sh
  export LD_PRELOAD=/mnt/storage/anaconda3/envs/hailo_env/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2:$LD_PRELOAD
  CUDA_VISIBLE_DEVICES=99 $HPY parse.py gridnet_vert2_chip.onnx gridnet_vert2_parsed.har > logs/parse_vert2.log 2>&1
  CK=$NEW CUDA_VISIBLE_DEVICES=1 TF_FORCE_GPU_ALLOW_GROWTH=true OPT_LEVEL=0 $HPY quantize.py gridnet_vert2_parsed.har v2_L0_cal8k.har 8192 alls_base.txt > logs/q_v2_L0_cal8k.log 2>&1
  for v in vtval vnval chval cvval g5val; do
    CK=$NEW CUDA_VISIBLE_DEVICES=1 TF_FORCE_GPU_ALLOW_GROWTH=true GRID_DROP_LANG="g5:zho_Hans,jpn_Jpan" $HPY eval_hailo.py v2_L0_cal8k.har quant $v 200 >> logs/eval_v2_quant.log 2>&1
  done
  CUDA_VISIBLE_DEVICES=99 $HPY /mnt/storage/lib_project/models/Hailo_model/hailo_re/compile_hef.py v2_L0_cal8k.har v2_L0_cal8k.hef > logs/compile_v2.log 2>&1
  echo EXIT $? >> logs/compile_v2.log
) &
wait
echo ALLDONE >> $H/logs/after_vert2.log
