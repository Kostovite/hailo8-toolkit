#!/bin/bash
source /mnt/storage/lib_project/models/Hailo_model/convert/lib/env.sh; henv
export CUDA_VISIBLE_DEVICES=99 HAILO_CPSERVER= HAILO_INIT_LOG=1 HAILO_INIT_UNITS=11 OMP_NUM_THREADS=4
D=/mnt/storage/lib_project/models/Hailo_model/hailo_re/probe/runs/slotcount
VIRTUAL_ENV=$D/seed11/sdk CONDA_PREFIX=$D/seed11/sdk systemd-run --user --scope -q --unit=slotcount-seed11 -p MemoryMax=16G \
  timeout -s KILL 7200 nice -n 5 /mnt/storage/anaconda3/envs/hailo_env/bin/python3.10 -u $D/../../sweep.py run "conv:c=256,o=256,q=16,h=56,w=56,k=3,n=100" $D/seed11 > $D/seed11.out 2>&1
