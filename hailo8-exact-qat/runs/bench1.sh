#!/bin/bash
cd /mnt/storage/lib_project/models/Hailo_model/hailo_re
export VAL_DIR=$PWD/data_v14sec/bench/images
R=$PWD
GPUS="3" ./evalgpu.sh bench_fp fp $R/models/v14sec/best.pt 896 896 --bs 8
GPUS="3" ./evalgpu.sh bench_q3a_s2000_har hailo $R/q3a_s2000_har.hn $R/capture_q3a_s2000_har --pt $R/models/v14sec/best.pt --bs 4
echo "BENCH1 DONE"
