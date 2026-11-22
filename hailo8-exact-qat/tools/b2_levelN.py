"""B2: fast level-0 Hailo quantization (calibration only, no AdaRound/QFT) with Ultralytics-identical letterbox.

Usage: python b2_level0.py PARSED_HAR OUT_HAR H W [N_CALIB] [EXTRA_ALLS_FILE]
"""
import glob
import os
import random
import sys
import time

import cv2
import numpy as np

from hailo_sdk_client import ClientRunner

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from letterbox import letterbox  # noqa: E402

src, dst, H, W = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
n_calib = int(sys.argv[5]) if len(sys.argv) > 5 else 1024
extra = open(sys.argv[6]).read() if len(sys.argv) > 6 else ""

# TRAIN_GLOB selects calibration pages; NMS="" skips Hailo NMS (YOLO26 is end2end: top-k runs on the host)
TRAIN_GLOB = os.environ.get("TRAIN_GLOB", "/mnt/storage/lib_project/models/Hailo_model/yolo_dataset_clean/images/train/*")
NMS = os.environ.get("NMS", "/mnt/storage/lib_project/models/Hailo_model/custom_nms_config_896.json")

files = sorted(glob.glob(TRAIN_GLOB))
random.Random(0).shuffle(files)
calib = np.zeros((n_calib, H, W, 3), np.float32)
for i, f in enumerate(files[:n_calib]):
    calib[i] = letterbox(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB), (H, W))[0]

runner = ClientRunner(har=src)
in_layer = runner.get_hn_model().get_input_layers()[0].name
alls = f"""normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])
model_optimization_flavor(optimization_level={os.environ.get("OPT_LEVEL", "0")}, compression_level=0)
model_optimization_config(calibration, batch_size=8, calibset_size={n_calib})
{f'nms_postprocess("{NMS}", meta_arch=yolov8, engine=cpu)' if NMS else ""}
{extra}"""
print(alls)
runner.load_model_script(alls)
t = time.time()
runner.optimize({in_layer: calib})
print(f"optimize took {time.time() - t:.0f} s")
runner.save_har(dst)
print("saved", dst)
