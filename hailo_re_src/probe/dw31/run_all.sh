#!/bin/bash
# all dw31 experiments in parallel, each in its own memory-capped scope with its own SDK home
D=/mnt/storage/lib_project/models/Hailo_model/hailo_re/probe/dw31
. /mnt/storage/lib_project/models/Hailo_model/hailo_re/env.sh
export CUDA_VISIBLE_DEVICES=99 OMP_NUM_THREADS=4 HAILO_CPCACHE=/mnt/storage/lib_project/models/Hailo_model/convert/compile_cache/cpsat_20261123
go() { local n=$1; shift; mkdir -p $D/sdk/$n/etc/hailo/dfc
  VIRTUAL_ENV=$D/sdk/$n CONDA_PREFIX=$D/sdk/$n timeout 2400 systemd-run --user --scope -q --unit=dw31-$n-$$ -p MemoryMax=20G \
    nice -n 5 $HPY -u "$@" >> $D/results.jsonl 2> $D/log_$n.txt & }
for H in 112 224; do for A in 0 1; do for Q in 8 16; do go iso_h${H}_a${A}_q${Q} $D/iso.py h${H}_a${A}_q${Q} $H $A $Q $D/iso; done; done; done
go v_base   $D/var.py base   "" $D/var
go v_f2     $D/var.py f2     "dw31_d0, dw31_d1, dw31_dc = defuse(dw31, 2)" $D/var
go v_f4     $D/var.py f4     "dw31_d0, dw31_d1, dw31_d2, dw31_d3, dw31_dc = defuse(dw31, 4)" $D/var
go v_f8     $D/var.py f8     "dw31_d0, dw31_d1, dw31_d2, dw31_d3, dw31_d4, dw31_d5, dw31_d6, dw31_d7, dw31_dc = defuse(dw31, 8)" $D/var
go v_spw    $D/var.py spw    "compilation_param(dw31, defuse_spatial_w=True)" $D/var
go v_sc8    $D/var.py sc8    "compilation_param(dw31, resources_allocation_strategy=manual_scs_selection, number_of_subclusters=8)" $D/var
go v_sc16   $D/var.py sc16   "compilation_param(dw31, resources_allocation_strategy=manual_scs_selection, number_of_subclusters=16)" $D/var
wait
echo ALLDONE >> $D/results.jsonl
