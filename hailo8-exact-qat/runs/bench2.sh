#!/bin/bash
cd /mnt/storage/lib_project/models/Hailo_model/hailo_re
export VAL_DIR=$PWD/data_v14sec/bench/images
R=$PWD
GPUS="3" ./evalgpu.sh bench_L0a16 hailo $R/yolo26m_v14sec_L0a16.hn $R/capture_y26a16 --pt $R/models/v14sec/best.pt --bs 4
GPUS="3" ./evalgpu.sh bench_L2a16 hailo $R/L2a16.hn $R/capture_L2a16 --pt $R/models/v14sec/best.pt --bs 4
echo "BENCH2 DONE"
