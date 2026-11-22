#!/bin/bash
# Lane B (GPU 1): our exact QAT on YOLO26n@640, starting from level 0 + 16-bit heads (q3a recipe).
set -e
R=/mnt/storage/lib_project/models/Hailo_model/hailo_re; S=$R/small; P=/mnt/storage/anaconda3/envs/general
PT=$S/y26n_v14sec/weights/best.pt
until [ -f $S/L0_ready ]; do sleep 30; done
cd $R/hailoqat
s=$(date +%s)
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 env -u LD_PRELOAD LD_LIBRARY_PATH=$P/lib $P/bin/python teacher_cache.py $S/teacher_n640 --pt $PT --imgsz 640 --bs 16 > $S/tcache.log 2>&1
echo "LANE_B teacher cache $(( $(date +%s) - s )) s"
s=$(date +%s)
cd $R; export PT
HN=$R/n_L0a16.hn CAP=$R/capture_n_L0a16 PORT=29640 GPUS="1" EVAL_ALL=1 ./exp2.sh n_qat_e3 --epochs 1 --eval_every 2000 \
  --lr_k 6e-4 --lr_b 3e-2 --ovf 0 --k_radius 4 --b_radius 256 --bs 8 --imgsz 640 --pt $PT --teacher_cache $S/teacher_n640 2>&1 | sed "s/^/LANE_B /"
echo "LANE_B QAT+eval $(( $(date +%s) - s )) s"
echo "LANE_B DONE"
