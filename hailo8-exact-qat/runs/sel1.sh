#!/bin/bash
cd /mnt/storage/lib_project/models/Hailo_model/hailo_re
R=$PWD
export VAL_DIR=$R/data_v14sec/bench_sel/images
for s in 4000 2000; do
  GPUS="3" ./evalgpu.sh sel_final1_s$s hailo $R/L2a16.hn $R/capture_L2a16 --pt $R/models/v14sec/best.pt --bs 4 --ints $R/runs/final1_L2_r4_3ep/ints_step$s.pt
done
GPUS="3" ./evalgpu.sh sel_q4_s3000 hailo $R/L2a16.hn $R/capture_L2a16 --pt $R/models/v14sec/best.pt --bs 4 --ints $R/runs/q4_fromL2_r2/ints_step3000.pt
GPUS="3" ./evalgpu.sh sel_q3a_s2000 hailo $R/yolo26m_v14sec_L0a16.hn $R/capture_y26a16 --pt $R/models/v14sec/best.pt --bs 4 --ints $R/runs/q3a_r4/ints_step2000.pt
echo "SEL1 DONE"
