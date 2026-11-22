#!/bin/bash
cd /mnt/storage/lib_project/models/Hailo_model/hailo_re
R=$PWD
export VAL_DIR=$R/data_v14sec/bench_test/images
./evalgpu.sh TEST_fp fp $R/models/v14sec/best.pt 896 896 --bs 8
./evalgpu.sh TEST_L0a16 hailo $R/yolo26m_v14sec_L0a16.hn $R/capture_y26a16 --pt $R/models/v14sec/best.pt --bs 4
./evalgpu.sh TEST_L2a16 hailo $R/L2a16.hn $R/capture_L2a16 --pt $R/models/v14sec/best.pt --bs 4
./evalgpu.sh TEST_q3a_har hailo $R/q3a_s2000_har.hn $R/capture_q3a_s2000_har --pt $R/models/v14sec/best.pt --bs 4
echo "TEST TABLE DONE"
