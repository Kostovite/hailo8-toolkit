#!/bin/bash
set -e
cd /mnt/storage/lib_project/models/Hailo_model/hailo_re
R=$PWD; P=/mnt/storage/anaconda3/envs/general
CK=$R/runs/final1_L2_r4_3ep/ints_step9201.pt
(cd hailoqat && env -u LD_PRELOAD LD_LIBRARY_PATH=$P/lib CUDA_VISIBLE_DEVICES="" $P/bin/python ints2npz.py $CK ../capture_L2a16 ../out/final_ints.npz)
. env.sh
CUDA_VISIBLE_DEVICES=99 $HPY hailoqat/writeback.py ../yolo26m_v14sec_L2a16.har out/final_ints.npz ../yolo26m_v14sec_final.har 2>&1 | grep -E "pass|wrote|Error|SystemExit"
echo "DELIVER writeback done"
CUDA_VISIBLE_DEVICES=99 nohup $HPY compile_hef.py ../yolo26m_v14sec_final.har ../yolo26m_v14sec_final_max.hef out/alls_perf_max.txt > out/compile_final_max.log 2>&1 &
echo "DELIVER compile started $!"
EVAL_GPUS="0 1 2 3" ./har_pipeline.sh ../yolo26m_v14sec_final.har final_har 0 2>&1 | grep -E "exact|tag"
VAL_DIR=$R/data_v14sec/bench_test/images GPUS="0 1 2 3" ./evalgpu.sh TEST_final_har hailo $R/final_har.hn $R/capture_final_har --pt $R/models/v14sec/best.pt --bs 4
echo "DELIVER DONE"
