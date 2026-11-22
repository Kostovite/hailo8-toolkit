#!/bin/bash
# zoo/ops/run3z.sh CASE: 16-bit max-precision probe of a CMP_ZOOM build (c3z/)
R=/mnt/storage/lib_project/models/Hailo_model/hailo_re; cd $R && . ./env.sh >/dev/null 2>&1; cd $R/zoo/ops
export TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=2
PREC=max16 timeout 3h $HPY probe.py run $1 c3z > c3z/$1/run16.log 2>&1; echo "$1 done"
