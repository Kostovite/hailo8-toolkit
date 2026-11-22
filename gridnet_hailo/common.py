"""Shared by the DFC-env scripts: torch without torchvision (the hailo_env build lacks its ops; GridNet only needs
roi_align for the unit heads, which the Hailo path does not use), the wordlid_vert trainer, and shard loading."""
import os, sys, types
WV = "/mnt/storage/lib_project/models/Dolphin/wordlid_vert"
CK = os.environ.get("CK", "/mnt/storage/lib_project/models/Dolphin/wordlid/data/grid/gridnet_vert1_best.pt")
_tv = types.ModuleType("torchvision"); _tv.ops = types.ModuleType("torchvision.ops")
_tv.ops.roi_align = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("roi_align unused on the Hailo path"))
sys.modules.setdefault("torchvision", _tv); sys.modules.setdefault("torchvision.ops", _tv.ops)
sys.path.insert(0, WV)
import numpy as np, torch
import grid_schema as S, gridnet as G, train_gridnet as T
S.SIDE = T.SIDE = G.SIDE = 896


def loader(tags, bs=8, shuffle=False, workers=4):
    from torch.utils.data import DataLoader
    ds = T.GridShards(tags, False)
    g = torch.Generator().manual_seed(0)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, generator=g, num_workers=workers,
                      collate_fn=lambda b: T.collate(b, False))


def load_net(ck=CK):
    ck = torch.load(ck, map_location="cpu", weights_only=False); ar = ck["arch"]
    net = G.GridNet(head_stride=ar["head_stride"], width=ar["width"]).eval()
    r = net.load_state_dict(ck["model"], strict=False)
    assert not r.unexpected_keys and all(k.startswith("merge_") for k in r.missing_keys), r
    return net
