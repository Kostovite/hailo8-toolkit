#!/bin/bash
cd /mnt/storage/lib_project/models/Hailo_model/hailo_re
for t in q3a_base:capture_q3a_s2000_har q3a_sat:capture_q3a_s2000_har_sat; do
  GPUS="0 1 2 3" ./evalgpu.sh smx_${t%%:*} hailo $PWD/q3a_s2000_har.hn $PWD/${t#*:} --pt $PWD/models/v14sec/best.pt --bs 4 | tail -1
done
echo SMXEVALDONE
