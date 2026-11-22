#!/usr/bin/env python
"""Page straightening model (2026-11-13): find the page, remove perspective + curl, leave flat pages alone.

Architecture (researched 2026-11-13, RUNS.md): a grid-regression net of the UVDoc type -- 8M parameters, one pass at
488 x 712 -> a 45 x 31 backward mesh (for every point of the flat page: where it lies in the photo).  The mesh IS page
detection + straightening.  Stock UVDoc weights already lift MDPBench VI photos 70.7 -> 79.2 end to end, but they also
move flat scans (mean mesh move .051 on the T1 book vs .075 on photos) -- unusable on scans.  This trainer fine-tunes
them on OUR pages with EXACT meshes:
  photo : photo_data.photo() (bend, creases, wrinkles, perspective, desk/keyboard/screen backgrounds, hands, lighting);
          mesh = H(p + d(p)) on the 45 x 31 grid, exact by construction.
  flat  : the page as a scan -- fills the frame, skew ~N(0, 1.2 deg) (30% exactly 0), uniform border; mesh = that rotation
          (identity when 0).  The model must learn "do nothing".
Loss = L1 on the mesh (frame-normalised).  Val = held-out pages (md5 % 50 == 0), fixed seed.
  ../venv_hf/bin/python train_flat.py --out runs/flat1 --steps 20000 --device cuda:1
  ../venv_hf/bin/python train_flat.py --eval CKPT          (val mesh error photo / flat, in % of the frame)
"""
import argparse, hashlib, os, random, sys, time
import cv2, numpy as np, torch
import torch.nn.functional as F
import photo_data as P
from model import UVDocnet
STOCK = "/mnt/storage/lib_project/models/Hailo_model/uvdoc/src/model/best_model.pkl"
IW, IH, GW, GH = 488, 712, 31, 45
GU, GV = np.meshgrid(np.linspace(0, 1, GW), np.linspace(0, 1, GH))


def mesh_photo(m):
    dx, dy = m["d"](GU, GV); u, v = GU + dx, GV + dy; H = m["H"]
    den = H[2, 0] * u + H[2, 1] * v + H[2, 2]
    X, Y = (H[0, 0] * u + H[0, 1] * v + H[0, 2]) / den, (H[1, 0] * u + H[1, 1] * v + H[1, 2]) / den
    return np.stack([X / (m["fw"] - 1) * 2 - 1, Y / (m["fh"] - 1) * 2 - 1]).astype(np.float32)


def flat(img, r):
    """The page as a scan: fills the frame, small skew, uniform border -> (frame RGB, mesh)."""
    h, w = img.shape[:2]; s = 760 / max(h, w); w, h = max(8, round(w * s)), max(8, round(h * s))
    img = cv2.resize(np.ascontiguousarray(img), (w, h), interpolation=cv2.INTER_AREA)
    ang = 0.0 if r.random() < .3 else r.gauss(0, 1.2)
    sc = 1.0 if ang == 0 and r.random() < .7 else r.uniform(.94, 1.0)
    A = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc); A[:, 2] += [r.uniform(-.02, .02) * w * (sc < 1), r.uniform(-.02, .02) * h * (sc < 1)]
    c = r.choice([255, 255, 0, r.randint(0, 255)])
    out = cv2.warpAffine(img, A, (w, h), flags=cv2.INTER_LINEAR, borderValue=(c, c, c))
    if r.random() < .5:
        out = P.clean_jitter(out, None, r)
    px, py = GU * (w - 1), GV * (h - 1)
    X, Y = A[0, 0] * px + A[0, 1] * py + A[0, 2], A[1, 0] * px + A[1, 1] * py + A[1, 2]
    return out, np.stack([X / (w - 1) * 2 - 1, Y / (h - 1) * 2 - 1]).astype(np.float32)


def uv_list(root):
    """real captured pages (UVDoc, 2023): img/<id>.png + grid2d/<geom>.mat (89 x 61 points, pixel xy of the flat grid)"""
    ids = sorted(x[:-4] for x in os.listdir(f"{root}/img") if x.endswith(".png"))
    return [i for i in ids if int(hashlib.md5(i.encode()).hexdigest()[:6], 16) % 50], [i for i in ids if int(hashlib.md5(i.encode()).hexdigest()[:6], 16) % 50 == 0]


def uv_sample(root, sid, r):
    """-> (frame RGB, mesh).  The frame is a random crop around the page (tight .. wide margins), as a phone would frame it."""
    import json, h5py
    geom = json.load(open(f"{root}/metadata_sample/{sid}.json"))["geom_name"]
    img = cv2.imread(f"{root}/img/{sid}.png")[:, :, ::-1]; H, W = img.shape[:2]
    with h5py.File(f"{root}/grid2d/{geom}.mat", "r") as f:
        g = np.array(f["grid2d"][:].T, np.float32)[::2, ::2]                    # (45, 31, 2) xy pixels
    x0, y0, x1, y1 = g[..., 0].min(), g[..., 1].min(), g[..., 0].max(), g[..., 1].max(); bw, bh = x1 - x0, y1 - y0
    mg = (lambda: r.uniform(0, .03)) if r.random() < .4 else (lambda: r.uniform(0, .25))
    x0, y0 = int(max(0, x0 - mg() * bw)), int(max(0, y0 - mg() * bh)); x1, y1 = int(min(W, x1 + mg() * bw + 1)), int(min(H, y1 + mg() * bh + 1))
    out = np.ascontiguousarray(img[y0:y1, x0:x1])
    if r.random() < .5:
        out = P.clean_jitter(out, None, r)
    return out, np.stack([(g[..., 0] - x0) / (x1 - x0 - 1) * 2 - 1, (g[..., 1] - y0) / (y1 - y0 - 1) * 2 - 1]).astype(np.float32)


class DS(torch.utils.data.IterableDataset):
    def __init__(self, items, seed=None, p_flat=.3, p_hard=.6, uv=None, p_uv=0.0, uv_only=False):
        self.items, self.seed, self.p_flat, self.p_hard = items, seed, p_flat, p_hard
        self.uv, self.p_uv, self.uv_only = uv, p_uv, uv_only                      # uv = (root, ids)

    def __iter__(self):
        wi = torch.utils.data.get_worker_info(); k = wi.id if wi else 0
        r = random.Random((self.seed if self.seed is not None else int(time.time() * 1000)) + 7919 * k)
        np.random.seed(r.randint(0, 2**31))
        while True:
            dom = r.choice(list(self.items)); path, lab, tax = r.choice(self.items[dom])
            try:
                if self.uv and (self.uv_only or r.random() < self.p_uv):
                    out, g = uv_sample(self.uv[0], r.choice(self.uv[1]), r); kind = 1
                    x = cv2.resize(out, (IW, IH), interpolation=cv2.INTER_AREA)
                    yield torch.from_numpy(x).permute(2, 0, 1), torch.from_numpy(g), kind
                    continue
                img, _, _ = P.load_sample(path, "", tax)
                if r.random() < self.p_flat:
                    out, g = flat(img, r); kind = 0
                else:
                    m = {}; out, _, _ = P.photo(img, np.zeros((0, 4), np.float32), np.zeros(0, np.int64), r, frame_long=820, out_map=m, hard=r.random() < self.p_hard)
                    out = out[:, :, ::-1] if False else out; g = mesh_photo(m); kind = 1
            except Exception:  # noqa: BLE001
                continue
            x = cv2.resize(np.ascontiguousarray(out), (IW, IH), interpolation=cv2.INTER_AREA)
            yield torch.from_numpy(x).permute(2, 0, 1), torch.from_numpy(g), kind


def split():
    tr, va = {}, {}
    for dom, its in P.manifest().items():
        for it in its:
            (va if int(hashlib.md5(it[0].encode()).hexdigest()[:6], 16) % 50 == 0 else tr).setdefault(dom, []).append(it)
    return tr, va


def load(ck, dev):
    net = UVDocnet(num_filter=32, kernel_size=5); sd = torch.load(ck, map_location="cpu"); net.load_state_dict(sd.get("model_state", sd)); return net.to(dev)


def evaluate(net, va, dev, n=400, uv=None):
    """uv = (root, val ids): error on the real captured pages only (returned as the 'photo' numbers)"""
    net.eval(); it = iter(torch.utils.data.DataLoader(DS(va, seed=1234, uv=uv, uv_only=uv is not None), batch_size=20, num_workers=4)); e = {0: [0.0], 1: []}
    with torch.no_grad():
        for _ in range(n // 20):
            x, g, k = next(it)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.startswith("cuda")):
                p = net(x.to(dev).float() / 255)[0]
            d = (p.float().cpu() - g).abs().mean(dim=(1, 2, 3)) * 50          # % of the frame side ([-1, 1] = 100%)
            for a, b in zip(d.tolist(), k.tolist()):
                e[b].append(a)
    net.train(); return float(np.mean(e[1])), float(np.mean(e[0])), float(np.percentile(e[1], 95))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out"); ap.add_argument("--steps", type=int, default=20000); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--device", default="cuda:1"); ap.add_argument("--init", default=STOCK)
    ap.add_argument("--eval"); ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--uvdoc", help="UVDoc_final folder: real captured pages mixed in"); ap.add_argument("--p-uv", type=float, default=.3)
    ap.add_argument("--p-flat", type=float, default=.3, help="share of the NON-real samples that are flat scans (1.0 = no synthetic photos)")
    ap.add_argument("--teacher", help="2026-11-13 flat3: photo samples (real + synthetic) are trained toward THIS frozen net's mesh instead of their\n"
                    "exact mesh (keeps stock's real-photo skill: every exact-mesh fine-tune lost DocUNet MS-SSIM); flat scans keep the identity target")
    a = ap.parse_args()
    torch.set_num_threads(4); tr, va = split(); dev = a.device
    uvt, uvv = uv_list(a.uvdoc) if a.uvdoc else (None, None)
    def ev(net):
        s = "val photo %.3f flat %.3f p95 %.3f" % evaluate(net, va, dev)
        if a.uvdoc:
            u = evaluate(net, va, dev, n=200, uv=(a.uvdoc, uvv)); s += "  real(UVDoc) %.3f p95 %.3f" % (u[0], u[2])
        return s
    if a.eval:
        print(a.eval, ev(load(a.eval, dev)), "(% of frame)"); return
    os.makedirs(a.out, exist_ok=True); net = load(a.init, dev).train()
    print("train pages", {k: len(v) for k, v in tr.items()}, "val", sum(len(v) for v in va.values()), flush=True)
    print("step 0 (init)", ev(net), "| real train pages", len(uvt) if uvt else 0, flush=True)
    teacher = load(a.teacher, dev).eval() if a.teacher else None
    if teacher is not None:
        for q in teacher.parameters():
            q.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4); sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=.03)
    scaler = torch.amp.GradScaler("cuda"); it = iter(torch.utils.data.DataLoader(DS(tr, p_flat=a.p_flat, uv=(a.uvdoc, uvt) if a.uvdoc else None, p_uv=a.p_uv), batch_size=a.bs, num_workers=a.workers, prefetch_factor=4)); t0 = time.time(); run = 0
    for step in range(1, a.steps + 1):
        x, g, kind = next(it); x, g = x.to(dev).float() / 255, g.to(dev)
        if teacher is not None and bool((kind == 1).any()):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                tg = teacher(x)[0].float()
            ph = (kind == 1).to(dev)[:, None, None, None]; g = torch.where(ph, tg, g)
        with torch.autocast("cuda", dtype=torch.float16):
            p = net(x)[0]
        loss = (p.float() - g).abs().mean(); opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step(); run = .98 * run + .02 * loss.item() if step > 1 else loss.item()
        if step % 100 == 0:
            print(f"step {step} loss {run * 50:.3f}% {(time.time() - t0) / step:.2f}s/step", flush=True)
        if step % 2500 == 0 or step == a.steps:
            torch.save(net.state_dict(), f"{a.out}/flat_{step}.pt")
            print("EVAL step %d" % step, ev(net), flush=True)


if __name__ == "__main__":
    main()
