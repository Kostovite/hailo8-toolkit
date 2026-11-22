#!/bin/bash
source /mnt/storage/lib_project/models/Hailo_model/hailo_re/env.sh
R=/mnt/storage/lib_project/models/Hailo_model/hailo_re; S=$R/probe/scan/gn; mkdir -p $S/src $S/out; cd $R/probe
GP=/mnt/storage/anaconda3/envs/general; export DIST=normal CUDA_VISIBLE_DEVICES=
for c in 8 16 32 64 128; do for h in 8 12 16 20 24 32 40; do
  d=$S/src/gn1_C${c}_H${h}; mkdir -p $d
  (cd $R/zoo/ops && TC_C=$c TC_H=$h TC_W=$h env -u LD_PRELOAD LD_LIBRARY_PATH=$GP/lib $GP/bin/python -c "
import os, torch, torch_cases as T
m = T.L(T.CASES['GroupNorm1'][0], T.C).eval()
torch.onnx.export(m, torch.randn(1, T.C, T.H, T.W), '$d/m.onnx', opset_version=17, input_names=['x'], output_names=['y'], dynamo=False)" > /dev/null 2>&1)
  echo $d
done; done > $S/src.txt
xargs -P 6 -I{} sh -c "$HPY -u sweep.py run @{} $S/out > /dev/null 2>&1" < $S/src.txt
ls -d $S/out/* | while read d; do [ -s $d/m.hef ] && echo $d; done > $S/dirs.txt
$HPY -u silent/chip_vs_emu.py emu 8 $S/dirs.txt > $S/emu.log 2>&1
DEVICE=0000:03:00.0 $HPY -u silent/chip_vs_emu.py chip $S/res.jsonl $S/dirs.txt > $S/chip.log 2>&1
echo DONE
