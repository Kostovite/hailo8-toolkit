#!/bin/bash
set -uo pipefail
E2=/mnt/storage/lib_project/models/moe/runs/codex_fusion_20261119/hailo_current_20261119/grid_e2e; A=$E2/../aux_models; cd $E2
tenv(){ source /mnt/storage/anaconda3/etc/profile.d/conda.sh; conda activate general; export PATH="$CONDA_PREFIX/bin:$PATH" LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8; unset LD_PRELOAD; }
[ -f ix.npy ] || { (tenv; CUDA_VISIBLE_DEVICES=2 python -u grid_e2e.py dump 400 $E2) > dump.log 2>&1 || { echo dump > e2e.exit; exit 1; }; }
(tenv; CUDA_VISIBLE_DEVICES=2 python -u grid_e2e.py eval $E2 torch) > eval_torch.log 2>&1 &
( source /mnt/storage/lib_project/hailo8-exact-qat/tools/env.sh
  export TF_FORCE_GPU_ALLOW_GROWTH=true OMP_NUM_THREADS=8 LD_PRELOAD="/mnt/storage/anaconda3/envs/hailo_env/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2:$LD_PRELOAD"
  $HPY -c "import numpy as np;x=np.load('canvas.npy',mmap_mode='r');np.save('c_a.npy',x[:200]);np.save('c_b.npy',x[200:])"
  CUDA_VISIBLE_DEVICES=3 $HPY -u /mnt/storage/lib_project/models/Hailo_model/preproc_hailo/emu.py $A/gridnet_trunk_all16.har c_a.npy qa.npy q > emu_a.log 2>&1 &
  CUDA_VISIBLE_DEVICES=2 $HPY -u /mnt/storage/lib_project/models/Hailo_model/preproc_hailo/emu.py $A/gridnet_trunk_all16.har c_b.npy qb.npy q > emu_b.log 2>&1; wait
  for k in 0 1; do $HPY -c "import numpy as np;a=np.load('qa_$k.npy',mmap_mode='r');b=np.load('qb_$k.npy',mmap_mode='r');o=np.lib.format.open_memmap('q_$k.npy','w+',a.dtype,(len(a)+len(b),)+a.shape[1:]);o[:len(a)]=a;o[len(a):]=b;o.flush()" && rm qa_$k.npy qb_$k.npy; done ) || { echo emu > e2e.exit; exit 1; }
wait
(tenv; CUDA_VISIBLE_DEVICES=3 python -u grid_e2e.py eval $E2 emu) > eval_emu.log 2>&1 || { echo eval_emu > e2e.exit; exit 1; }
grep RESULT eval_torch.log eval_emu.log; echo 0 > e2e.exit
