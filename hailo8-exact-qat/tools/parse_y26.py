"""Parse a YOLO26 (end2end, reg_max=1) ONNX into a Hailo HAR, cut before the host-side top-k selection.

Usage: python parse_y26.py ONNX OUT_HAR
"""
import collections
import sys

from hailo_sdk_client import ClientRunner

END_NODES = [f"/model.23/one2one_cv{b}.{i}/one2one_cv{b}.{i}.2/Conv" for i in range(3) for b in (2, 3)]
runner = ClientRunner(hw_arch="hailo8")
runner.translate_onnx_model(sys.argv[1], "best", start_node_names=["images"], end_node_names=END_NODES)
runner.save_har(sys.argv[2])
hn = runner.get_hn_dict()
print("HN layer types:", dict(collections.Counter(v["type"] for v in hn["layers"].values())))
