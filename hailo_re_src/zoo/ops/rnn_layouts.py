"""Which recurrent-layer layouts does the DFC accept? x (1,C,H,W) -> mean over H -> sequence over W, features C.

Usage (general env): python rnn_layouts.py OUT_DIR
"""
import os
import sys

import torch
import torch.nn as nn

from torch_cases import C, H, W, L


class Seq(nn.Module):
    def __init__(self, typ, batch_first=True, layout="squeeze", bidir=False, layers=1):
        super().__init__()
        self.r = typ(C, C // (2 if bidir else 1), batch_first=batch_first, bidirectional=bidir, num_layers=layers)
        self.bf, self.layout = batch_first, layout

    def forward(self, x):
        t = nn.functional.avg_pool2d(x, (x.shape[2], 1))  # (1, C, 1, W)
        if self.layout == "squeeze":
            s = t.squeeze(2).transpose(1, 2)  # (1, W, C)
        else:  # permute + flatten
            s = t.permute(0, 3, 1, 2).flatten(2)  # (1, W, C)
        if not self.bf:
            s = s.transpose(0, 1)
        y, _ = self.r(s)
        if not self.bf:
            y = y.transpose(0, 1)
        y = y.transpose(1, 2).unsqueeze(2)  # (1, C, 1, W)
        return y


class GRUCellUnrolled(nn.Module):
    """GRU written with Linear + sigmoid/tanh, unrolled over W (what a recurrent layer reduces to on a dataflow chip)."""

    def __init__(self):
        super().__init__()
        self.x2h, self.h2h = nn.Conv1d(C, 3 * C, 1), nn.Conv1d(C, 3 * C, 1)

    def forward(self, x):
        t = nn.functional.avg_pool2d(x, (x.shape[2], 1)).squeeze(2)  # (1, C, W)
        gx = self.x2h(t)
        h = torch.zeros_like(t[:, :, :1])
        outs = []
        for i in range(t.shape[2]):
            gh = self.h2h(h)
            r = torch.sigmoid(gx[:, :C, i:i + 1] + gh[:, :C])
            z = torch.sigmoid(gx[:, C:2 * C, i:i + 1] + gh[:, C:2 * C])
            n = torch.tanh(gx[:, 2 * C:, i:i + 1] + r * gh[:, 2 * C:])
            h = (1 - z) * n + z * h
            outs.append(h)
        return torch.cat(outs, 2).unsqueeze(2)


CASES = {
    "LSTM_bf_squeeze": Seq(nn.LSTM), "GRU_bf_squeeze": Seq(nn.GRU), "RNN_bf_squeeze": Seq(nn.RNN),
    "LSTM_sf_squeeze": Seq(nn.LSTM, batch_first=False), "GRU_sf": Seq(nn.GRU, batch_first=False),
    "LSTM_bf_permute": Seq(nn.LSTM, layout="permute"), "LSTM_bidir": Seq(nn.LSTM, bidir=True),
    "LSTM_2layer": Seq(nn.LSTM, layers=2), "GRU_unrolled": GRUCellUnrolled(),
}

if __name__ == "__main__":
    out = sys.argv[1]
    torch.manual_seed(0)
    for name, mod in CASES.items():
        d = os.path.join(out, "T_" + name)
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "dom.txt"), "w").write("normal")
        try:
            torch.onnx.export(L(mod.eval()).eval(), torch.randn(1, C, H, W), os.path.join(d, "m.onnx"), opset_version=17,
                              input_names=["x"], output_names=["y"], dynamo=False)
            print("T_" + name)
        except Exception as e:  # noqa: BLE001
            print(f"# {name}: {type(e).__name__} {str(e)[:100]}", file=sys.stderr)
