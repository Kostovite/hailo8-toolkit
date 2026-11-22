#!/bin/bash
cd /mnt/storage/lib_project/models/Hailo_model/preproc_hailo
until grep -q EXIT eval_build.log; do sleep 20; done
. /mnt/storage/lib_project/hailo8-exact-qat/tools/env.sh; export CUDA_VISIBLE_DEVICES=
$HPY emu.py orient2_L0_int8.har eval/orient_canvas.npy eval/feat_fp.npy fp > logs/emu_fp.log 2>&1
$HPY emu.py orient2_L0_int8.har eval/orient_canvas.npy eval/feat_int8.npy q > logs/emu_int8.log 2>&1
$HPY emu.py orient2_L0_all16.har eval/orient_canvas.npy eval/feat_all16.npy q > logs/emu_all16.log 2>&1
/mnt/storage/lib_project/models/layout_teacher/venv_hf/bin/python orient_gate.py score eval/feat_fp.npy eval/feat_int8.npy eval/feat_all16.npy > logs/orient_gate.txt 2>&1
echo DONE >> logs/orient_gate.txt
