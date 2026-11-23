#!/bin/bash
# Slotter node-count probe (RULES 29G): same node count, different node size/precision -> start budget, search effort
source /mnt/storage/lib_project/models/Hailo_model/convert/lib/env.sh; henv
export CUDA_VISIBLE_DEVICES=99 HAILO_CPSERVER= HAILO_INIT_LOG=1 OMP_NUM_THREADS=4
D=/mnt/storage/lib_project/models/Hailo_model/hailo_re/probe/runs/slotcount
PY=/mnt/storage/anaconda3/envs/hailo_env/bin/python3.10
for c in "small8:c=64,o=64,q=8" "small16:c=64,o=64,q=16" "big8:c=256,o=256,q=8" "big16:c=256,o=256,q=16"; do
  t=${c%%:*}; p=${c#*:}; mkdir -p $D/$t/sdk/etc/hailo/dfc
  VIRTUAL_ENV=$D/$t/sdk CONDA_PREFIX=$D/$t/sdk systemd-run --user --scope -q --unit=slotcount-$t -p MemoryMax=16G \
    timeout -s KILL 5400 nice -n 5 $PY -u $D/../../sweep.py run "conv:$p,h=56,w=56,k=3,n=24" $D/$t > $D/$t.out 2>&1 &
done
wait
