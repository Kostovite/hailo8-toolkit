#!/bin/bash
set -e
cd /mnt/storage/lib_project/models/Hailo_model/hailo_re
R=$PWD; P=/mnt/storage/anaconda3/envs/general
(cd hailoqat && env -u LD_PRELOAD LD_LIBRARY_PATH=$P/lib CUDA_VISIBLE_DEVICES="" $P/bin/python ints2npz.py ../runs/q4_fromL2_r2/ints_step3000.pt ../capture_L2a16 ../out/q4_s3000_ints.npz)
. env.sh
CUDA_VISIBLE_DEVICES=99 $HPY hailoqat/writeback.py ../yolo26m_v14sec_L2a16.har out/q4_s3000_ints.npz ../yolo26m_v14sec_q4_s3000.har 2>&1 | grep -E "pass|wrote|Error|SystemExit"
EVAL_GPUS="3" ./har_pipeline.sh ../yolo26m_v14sec_q4_s3000.har q4_s3000_har 3 2>&1 | grep -E "exact|tag"
VAL_DIR=$R/data_v14sec/bench/images GPUS="3" ./evalgpu.sh bench_q4_s3000_har hailo $R/q4_s3000_har.hn $R/capture_q4_s3000_har --pt $R/models/v14sec/best.pt --bs 4
echo "Q4HAR DONE"
