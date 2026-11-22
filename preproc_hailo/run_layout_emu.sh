#!/bin/bash
# usage: run_layout_emu.sh HAR fp|q TAG GPU   -> eval/lay_TAG_{0..3}.npy, then AP via layout_gate.py score
cd /mnt/storage/lib_project/models/Hailo_model/preproc_hailo; E=$PWD/eval
( . /mnt/storage/lib_project/hailo8-exact-qat/tools/env.sh
  export LD_PRELOAD=/mnt/storage/anaconda3/envs/hailo_env/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2:$LD_PRELOAD CUDA_VISIBLE_DEVICES=$4 TF_FORCE_GPU_ALLOW_GROWTH=true
  $HPY emu.py $1 $E/layout_x.npy $E/lay_$3.npy $2 > logs/emul_$3.log 2>&1 )
( source /mnt/storage/anaconda3/etc/profile.d/conda.sh; conda activate general; export PATH="$CONDA_PREFIX/bin:$PATH" LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=$4
  /mnt/storage/lib_project/models/layout_teacher/venv_hf/bin/python layout_gate.py score /mnt/storage/lib_project/models/layout_teacher/train/runs/r3/ema_24000.pt $E/lay_$3 > logs/layout_gate_$3.txt 2>&1 )
echo DONE >> logs/layout_gate_$3.txt
