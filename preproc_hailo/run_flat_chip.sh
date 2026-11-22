#!/bin/bash
cd /mnt/storage/lib_project/models/Hailo_model/preproc_hailo; E=$PWD/eval
./run_c.sh flat2_L0_all16 & ./run_c.sh flat2_L0_int8 & wait
( . /mnt/storage/lib_project/hailo8-exact-qat/tools/env.sh; $HPY chip.py flat2_L0_all16.hef $E/flat_frames.npy $E/chipmesh_all16.npy > logs/chipf_all16.log 2>&1 )
( source /mnt/storage/anaconda3/etc/profile.d/conda.sh; conda activate general; export PATH="$CONDA_PREFIX/bin:$PATH" LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=
  /mnt/storage/lib_project/models/layout_teacher/venv_hf/bin/python flat_gate.py score $E/mesh_all16.npy $E/chipmesh_all16.npy > logs/flat_chip_gate.txt 2>&1 )
echo DONE >> logs/flat_chip_gate.txt
