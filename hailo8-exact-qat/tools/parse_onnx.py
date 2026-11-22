"""Parse a YOLOv8 ONNX into a Hailo HAR, cutting before the Detect decode (same end nodes as export_and_parse_896.py).

Usage: python parse_onnx.py ONNX OUT_HAR
"""
import sys

from hailo_sdk_client import ClientRunner

END_NODES = [
    "/model.22/cv2.0/cv2.0.2/Conv", "/model.22/cv3.0/cv3.0.2/Conv",
    "/model.22/cv2.1/cv2.1.2/Conv", "/model.22/cv3.1/cv3.1.2/Conv",
    "/model.22/cv2.2/cv2.2.2/Conv", "/model.22/cv3.2/cv3.2.2/Conv",
]
runner = ClientRunner(hw_arch="hailo8")
runner.translate_onnx_model(sys.argv[1], "best", start_node_names=["images"], end_node_names=END_NODES)
runner.save_har(sys.argv[2])
print("saved", sys.argv[2])
