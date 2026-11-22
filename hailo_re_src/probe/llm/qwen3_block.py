"""One Qwen3 decoder layer in a Hailo-friendly form (real weights), for measuring an LLM on Hailo-8 without
converting the whole model: contexts per layer, FPS, accuracy -> extrapolate to 28 layers.

Layout: tokens on a GH x GW grid, x = [1, hidden, GH, GW] (NCHW, N = GH*GW, row-major token order), so every
Linear is a 1x1 conv. (A 1 x N strip parses to a zero-height tensor at o_proj: use a 2-D grid like yolo PSA.) Exact rewrites:
  RMSNorm            x * rsqrt(mean_C(x^2) + eps) * w
  q_norm / k_norm    per-head RMSNorm: the per-head mean of x^2 is a block-diagonal 1x1 conv (weights 1/head_dim)
  RoPE               x*cos + R(x)*sin, R = rotate_half as a fixed signed-permutation 1x1 conv; cos/sin constants
  GQA                k/v heads repeated by a 0/1 1x1 conv; causal prefill attention with a constant mask
Prefill only (no KV cache): N fixed tokens per pass, positions 0..N-1.

Usage (general env): python qwen3_block.py OUT_DIR LAYER N   -> OUT_DIR/q3_l{LAYER}_n{N}/m.onnx (+ check vs HF)
"""
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file

SNAP = glob.glob("/mnt/storage/tmp/hf_tok/hub/models--Qwen--Qwen3-0.6B/snapshots/*/")[0]
H, NH, NKV, HD, FF, EPS, THETA = 1024, 16, 8, 128, 3072, 1e-6, 1e6


def conv1(w):  # Linear weight [out, in] -> 1x1 conv
    c = nn.Conv2d(w.shape[1], w.shape[0], 1, bias=False)
    c.weight.data = w.float().reshape(*w.shape, 1, 1).clone()
    return c


def fixed(w):
    c = conv1(torch.as_tensor(w))
    c.weight.requires_grad_(False)
    return c


class Block(nn.Module):
    def __init__(self, sd, i, n):
        super().__init__()
        self.gh = int(round(n ** 0.5))
        while n % self.gh:
            self.gh -= 1
        self.gw = n // self.gh
        p = f"model.layers.{i}."
        g = lambda k: sd[p + k]  # noqa: E731
        self.n = n
        self.ln1 = nn.Parameter(g("input_layernorm.weight").float().reshape(1, H, 1, 1))
        self.ln2 = nn.Parameter(g("post_attention_layernorm.weight").float().reshape(1, H, 1, 1))
        self.q, self.k, self.v = conv1(g("self_attn.q_proj.weight")), conv1(g("self_attn.k_proj.weight")), conv1(
            g("self_attn.v_proj.weight"))
        self.o = conv1(g("self_attn.o_proj.weight"))
        self.gate, self.up, self.down = conv1(g("mlp.gate_proj.weight")), conv1(g("mlp.up_proj.weight")), conv1(
            g("mlp.down_proj.weight"))
        self.qn = nn.Parameter(g("self_attn.q_norm.weight").float().repeat(NH).reshape(1, NH * HD, 1, 1))
        self.kn = nn.Parameter(g("self_attn.k_norm.weight").float().repeat(NKV).reshape(1, NKV * HD, 1, 1))
        blk = lambda nh: np.kron(np.eye(nh), np.full((HD, HD), 1.0 / HD)).astype(np.float32)  # noqa: E731
        self.qmean, self.kmean = fixed(blk(NH)), fixed(blk(NKV))
        rot = np.zeros((HD, HD), np.float32)  # rotate_half: [-x2, x1]
        rot[np.arange(HD // 2), np.arange(HD // 2) + HD // 2] = -1
        rot[np.arange(HD // 2) + HD // 2, np.arange(HD // 2)] = 1
        self.qrot, self.krot = fixed(np.kron(np.eye(NH), rot)), fixed(np.kron(np.eye(NKV), rot))
        rep = np.kron(np.kron(np.eye(NKV), np.ones((NH // NKV, 1))), np.eye(HD)).astype(np.float32)
        self.rep = fixed(rep)  # [NH*HD, NKV*HD]: kv head j -> q heads 2j, 2j+1
        inv = 1.0 / THETA ** (np.arange(0, HD, 2) / HD)
        ang = np.outer(np.arange(n), inv)  # [N, HD/2]
        emb = np.concatenate([ang, ang], 1).T  # [HD, N]
        self.register_buffer("cos", torch.tensor(np.cos(emb), dtype=torch.float32).reshape(1, 1, HD, n))
        self.register_buffer("sin", torch.tensor(np.sin(emb), dtype=torch.float32).reshape(1, 1, HD, n))
        m = np.triu(np.full((n, n), -1e4, np.float32), 1)
        self.register_buffer("mask", torch.tensor(m).reshape(1, 1, n, n))

    @staticmethod
    def rms(x, w, mean=None):
        m = mean(x.pow(2)) if mean is not None else x.pow(2).mean(1, keepdim=True)  # x*x: parser skips the Mul
        return x * torch.rsqrt(m + EPS) * w

    def rope(self, x, rot, nh):
        cos = self.cos.repeat(1, nh, 1, 1).reshape(1, nh * HD, self.gh, self.gw)
        sin = self.sin.repeat(1, nh, 1, 1).reshape(1, nh * HD, self.gh, self.gw)
        return x * cos + rot(x) * sin

    def forward(self, x):  # x [1, H, GH, GW]
        h = self.rms(x, self.ln1)
        q = self.rope(self.rms(self.q(h), self.qn, self.qmean), self.qrot, NH)
        k = self.rope(self.rms(self.k(h), self.kn, self.kmean), self.krot, NKV)
        v = self.v(h)
        k, v = self.rep(k), self.rep(v)
        n = self.n
        q, k, v = q.view(1, NH, HD, n), k.view(1, NH, HD, n), v.view(1, NH, HD, n)
        a = ((q.transpose(-2, -1) @ k) * HD ** -0.5 + self.mask).softmax(-1)  # [1, NH, N, N]
        o = (v @ a.transpose(-2, -1)).reshape(1, NH * HD, self.gh, self.gw)
        x = x + self.o(o)
        h = self.rms(x, self.ln2)
        return x + self.down(nn.functional.silu(self.gate(h)) * self.up(h))


def hf_ref(sd, i, x_tokens):
    from transformers import AutoConfig
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding
    cfg = AutoConfig.from_pretrained(SNAP)
    cfg._attn_implementation = "eager"
    layer = Qwen3DecoderLayer(cfg, i).float().eval()
    p = f"model.layers.{i}."
    layer.load_state_dict({k[len(p):]: v.float() for k, v in sd.items() if k.startswith(p)})
    n = x_tokens.shape[1]
    pos = torch.arange(n)[None]
    pe = Qwen3RotaryEmbedding(cfg)(x_tokens, pos)
    mask = torch.triu(torch.full((n, n), float("-inf")), 1)[None, None]
    out = layer(x_tokens, attention_mask=mask, position_ids=pos, position_embeddings=pe)
    return out[0] if isinstance(out, tuple) else out


if __name__ == "__main__":
    out, i, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    sd = load_file(SNAP + "model.safetensors")
    torch.manual_seed(0)
    blk = Block(sd, i, n).eval()
    emb = sd["model.embed_tokens.weight"].float()
    ids = torch.randint(0, emb.shape[0], (n,))
    xt = emb[ids][None]  # [1, N, H] realistic layer-0 input
    with torch.no_grad():
        ref = hf_ref(sd, i, xt)[0]  # [N, H]
        y = blk(xt[0].T.reshape(1, H, blk.gh, blk.gw))[0].reshape(H, n).T
    err = (y - ref).abs().max().item() / ref.abs().max().item()
    print(f"layer {i} N={n}: max relative diff vs HF {err:.2e}")
    d = f"{out}/q3_l{i}_n{n}"
    os.makedirs(d, exist_ok=True)
    cal = emb[torch.randint(0, emb.shape[0], (16, n))]  # 16 random token sequences as calibration [16, N, H]
    np.save(f"{d}/calib.npy", cal.permute(0, 2, 1).reshape(16, H, blk.gh, blk.gw).numpy())
    torch.onnx.export(blk, xt[0].T.reshape(1, H, blk.gh, blk.gw), f"{d}/m.onnx", input_names=["x"], output_names=["y"],
                      opset_version=17, dynamo=False)
    print("@" + os.path.abspath(d))
