#!/usr/bin/env python
"""Chip parts of the page pre-processing models (2026-11-13) -> ONNX, fixed shapes.
  orient : OriNet stem+body on a 896x896 canvas (page scaled to fit, padded); host = 7x7 window mean, fc, softmax, valid windows
  flat   : UVDoc-type net (flat2.pt) at 712x488 -> 45x31x2 backward mesh (the 3-D head is dropped)
Input of both: RGB 0..1 (normalisation folded into the first conv by the parser via the model script)."""
import sys, torch, torch.nn as nn
sys.path.insert(0, "/mnt/storage/lib_project/models/layout_teacher/train"); sys.path.insert(0, "/mnt/storage/lib_project/models/Hailo_model/uvdoc/src")
from train_orient import OriNet
from model import UVDocnet


class OriChip(nn.Module):
    def __init__(self, net):
        super().__init__(); self.n = net

    def forward(self, x):                      # x: RGB 0..1
        return self.n.body(self.n.stem(x - .5))


class FlatChip(nn.Module):
    def __init__(self, net):
        super().__init__(); self.n = net

    def forward(self, x):
        return self.n(x)[0]


o = OriNet(); o.load_state_dict(torch.load("/mnt/storage/lib_project/models/layout_teacher/train/runs/orient2/ori_5000.pt", map_location="cpu")); o.eval()
torch.onnx.export(OriChip(o), torch.rand(1, 3, 896, 896), "orient2_chip.onnx", opset_version=13, input_names=["image"], output_names=["feat"], dynamo=False)
f = UVDocnet(num_filter=32, kernel_size=5); sd = torch.load("/mnt/storage/lib_project/models/layout_teacher/models_flat/flat2.pt", map_location="cpu"); f.load_state_dict(sd.get("model_state", sd)); f.eval()
torch.onnx.export(FlatChip(f), torch.rand(1, 3, 712, 488), "flat2_chip.onnx", opset_version=13, input_names=["image"], output_names=["mesh"], dynamo=False)
import onnx, collections
for n in ("orient2_chip.onnx", "flat2_chip.onnx"):
    g = onnx.load(n).graph; print(n, collections.Counter(x.op_type for x in g.node), [d.dim_value for d in g.output[0].type.tensor_type.shape.dim])
    for x in g.node:
        if x.op_type == "Conv":
            a = {t.name: list(t.ints) for t in x.attribute}
            if a.get("dilations", [1, 1]) != [1, 1]: print("  dilated", x.name, a["dilations"])
