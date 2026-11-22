import sys
from hailo_sdk_client import ClientRunner
r = ClientRunner(hw_arch="hailo8")
r.translate_onnx_model(sys.argv[1], "gridnet")
r.save_har(sys.argv[2])
for l in r.get_hn_model().get_output_layers(): print("out", l.name, l.output_shape)
print("layers", len(list(r.get_hn_model().stable_toposort())))
