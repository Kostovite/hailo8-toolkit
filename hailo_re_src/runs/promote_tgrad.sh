#!/bin/bash
# promote the register-accumulator APU-gradient kernel once no job has the live extension loaded
while kill -0 3894436 2>/dev/null; do sleep 20; done
cd /mnt/storage/lib_project/models/Hailo_model/hailo_re/hailoqat/hq_ext
cp hq.cu hq.cu.r9 && cp hq.cu.new hq.cu && echo "PROMOTED tgrad2 $(date)"
