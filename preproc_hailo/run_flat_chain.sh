#!/bin/bash
cd /mnt/storage/lib_project/models/Hailo_model/preproc_hailo; E=$PWD/eval
./run_q.sh flat2 flat2_L0_all16 all16
. /mnt/storage/lib_project/hailo8-exact-qat/tools/env.sh; export CUDA_VISIBLE_DEVICES=
$HPY emu.py flat2_L0_int8.har $E/flat_frames.npy $E/mesh_fp.npy fp > logs/emuf_fp.log 2>&1
$HPY emu.py flat2_L0_int8.har $E/flat_frames.npy $E/mesh_int8.npy q > logs/emuf_int8.log 2>&1
[ -f flat2_L0_all16.har ] && $HPY emu.py flat2_L0_all16.har $E/flat_frames.npy $E/mesh_all16.npy q > logs/emuf_all16.log 2>&1
source /mnt/storage/anaconda3/etc/profile.d/conda.sh; conda activate general; export PATH="$CONDA_PREFIX/bin:$PATH" LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" PYTHONNOUSERSITE=1
/mnt/storage/lib_project/models/layout_teacher/venv_hf/bin/python flat_gate.py score $E/mesh_fp.npy $E/mesh_int8.npy $(ls $E/mesh_all16.npy 2>/dev/null) > logs/flat_gate.txt 2>&1
echo DONE >> logs/flat_gate.txt
