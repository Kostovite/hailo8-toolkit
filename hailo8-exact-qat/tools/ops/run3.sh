#!/bin/bash
# zoo/ops/run3.sh CASE: DFC probe of a cases3.py case at 8 bit (c3/) and 16 bit max precision (c3_16/), CPU only.
R=/mnt/storage/lib_project/models/Hailo_model/hailo_re; cd $R && . ./env.sh >/dev/null 2>&1; cd $R/zoo/ops
c=$1
export TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=2
PREC=a8_w8 timeout 3h $HPY probe.py run $c c3 > c3/$c/run8.log 2>&1
rm -rf c3_16/$c; mkdir -p c3_16/$c; cp c3/$c/{m.onnx,ref.onnx,dom.txt} c3_16/$c/; [ -f c3/$c/fit.json ] && cp c3/$c/fit.json c3_16/$c/
PREC=max16 timeout 3h $HPY probe.py run $c c3_16 > c3_16/$c/run16.log 2>&1
echo "$c done"
