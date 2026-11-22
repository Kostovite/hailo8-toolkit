"""GridNet split for Hailo-8: CHIP = trunk + region classifier (input canvas -> p2 feature map, kind/body/lang logits);
HOST = row/col profiles, 1-D heads, band/strip segment pooling (positions change per page, see split_model.HostPart).
usage: python export_onnx.py CKPT OUT.onnx"""
import sys, torch, torch.nn as nn
sys.path.insert(0, "/mnt/storage/lib_project/models/Dolphin/wordlid_vert")
import grid_schema as S, gridnet as G


def _proj_forward(self, x):
    """ProjBlock with the row/column broadcast as a nearest resize: Hailo has no Expand, and resize is how its own
    LayerNorm lowering broadcasts a reduced tensor. Identical values."""
    import torch.nn.functional as F
    r = F.interpolate(x.mean(3, keepdim=True), size=x.shape[-2:], mode="nearest")
    c = F.interpolate(x.mean(2, keepdim=True), size=x.shape[-2:], mode="nearest")
    return self.fuse(torch.cat([x, r, c], 1))


G.ProjBlock.forward = _proj_forward


class ChipPart(nn.Module):
    def __init__(self, net):
        super().__init__(); self.net = net

    def forward(self, x):
        p2, p4 = self.net.trunk(x)
        h = self.net.region(p4.mean((2, 3)))
        return p2, self.net.kind_head(h)


def load(ck):
    ck = torch.load(ck, map_location="cpu", weights_only=False); ar = ck["arch"]
    S.SIDE = G.SIDE = ar["side"]
    net = G.GridNet(head_stride=ar["head_stride"], width=ar["width"]).eval()
    r = net.load_state_dict(ck["model"], strict=False)
    assert not r.unexpected_keys and all(k.startswith("merge_") for k in r.missing_keys), r
    return net, ar["side"]


if __name__ == "__main__":
    net, side = load(sys.argv[1])
    torch.onnx.export(ChipPart(net), torch.zeros(1, 1, side, side), sys.argv[2], opset_version=13,
                      input_names=["canvas"], output_names=["p2", "kind"], do_constant_folding=True, dynamo=False)
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        ChipPart(net)(torch.zeros(1, 1, side, side))
    print("chip GMAC", fc.get_total_flops() / 2e9)
