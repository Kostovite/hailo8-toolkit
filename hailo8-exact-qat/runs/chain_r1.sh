#!/bin/bash
# r1: pilot regressed (0.817 vs L0 0.855) -> 5x lower kernel LR, 10x lower bias LR, 10x kd_cls; eval step400 + final
P=/mnt/storage/anaconda3/envs/general
R=/mnt/storage/lib_project/models/Hailo_model/hailo_re
cd $R/hailoqat
echo "CHAIN start train $(date)"
CUDA_VISIBLE_DEVICES=0,1 env -u LD_PRELOAD LD_LIBRARY_PATH=$P/lib PATH=$P/bin:/usr/bin:/bin NCCL_P2P_DISABLE=1 \
  $P/bin/torchrun --nproc_per_node=2 train_qat.py --out ../runs/r1 --max_steps 1200 --bs 2 --eval_every 400 \
  --lr_k 1e-3 --lr_b 0.05 --kd_cls 10 > ../runs/r1/stdout.log 2>&1 || { echo "CHAIN FAILED train rc=$?"; exit 1; }
echo "CHAIN train done $(date)"
for s in 400 1200; do
  bash $R/eval2gpu.sh r1_s$s hailo $R/yolo26m_v14sec_L0.hn ../capture_y26g --pt ../models/v14sec/best.pt --bs 4 \
    --ints ../runs/r1/ints_step$s.pt | sed "s/^/CHAIN EVAL /" || echo "CHAIN FAILED eval s$s"
done
echo "CHAIN DONE $(date)"
