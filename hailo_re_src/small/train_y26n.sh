#!/bin/bash
# YOLO26n (COCO) fine-tuned on the 14-class sec-merged layout set at 640, for the fast PTQ/QAT head-to-head
P=/mnt/storage/anaconda3/envs/general
cd /mnt/storage/lib_project/models/Hailo_model/hailo_re/small
env -u LD_PRELOAD PYTHONNOUSERSITE=1 LD_LIBRARY_PATH=$P/lib PATH=$P/bin:/usr/bin:/bin $P/bin/yolo detect train \
  model=/mnt/storage/lib_project/models/Hailo_model/yolo26n.pt data=/mnt/storage/lib_project/models/Hailo_model/hailo_re/data_v14sec/data.yaml \
  imgsz=640 epochs=15 batch=64 device=0,1 workers=8 project=/mnt/storage/lib_project/models/Hailo_model/hailo_re/small name=y26n_v14sec exist_ok=True \
  cos_lr=True close_mosaic=3 plots=False amp=False
