#!/bin/bash
# pwl/run_all.sh CASE OP PIECES MODE: fit + mark + DFC test of one probe case through pwl_inject; one result line
cd /mnt/storage/lib_project/models/Hailo_model/hailo_re/zoo/ops
G=/mnt/storage/anaconda3/envs/general/bin/python; c=$1; op=$2; P=$3; M=$4
dom=$(cat res/$c/dom.txt 2>/dev/null || $G -c "import sys; sys.path.insert(0,'.'); from probe import CASES; print(CASES['$c'][1])" 2>/dev/null)
o=pwl/out/${c}_$P; mkdir -p pwl/out
fitl=$($G pwl_inject.py fit res/$c/m.onnx pwl/calib_$dom.npy $o.json --ops $op --pieces $P 2>&1 | tail -1)
$G pwl_inject.py mark res/$c/m.onnx $o.json $o.onnx > /dev/null 2>&1
res=$(CUDA_VISIBLE_DEVICES=99 $HPY pwl_test.py res/$c/m.onnx $o.onnx $o.json pwl/calib_$dom.npy pwl/test_$dom.npy $([ "$M" != none ] && echo --mode=$M) --compile 2>&1 | grep -E "SQNR hw|SQNR float|^compiled|Error" | tr '\n' ' ')
echo "$c	$op	$P	$M	$(echo $fitl | grep -oE 'fit [^,]*, [0-9]+ APU layers')	$res"
