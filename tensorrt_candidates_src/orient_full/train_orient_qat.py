#!/usr/bin/env python
"""Our own page-orientation model (2026-11-12).

Why not PaddleOCR's PP-LCNet_x1_0_doc_ori: it judges a 224 px thumbnail of the whole page (glyphs are illegible), so it
goes by the page's look.  Measured (orient_eval.py / orient_rules.py, held-out pages): upright pages wrongly turned --
vertical Japanese 81/81 (the Han Nom layout), Bengali newspapers 21% (37% with crop voting), thesis figure pages 4.7%.

Design: a small ResNet reads TILES at glyph scale (a tile = 12-40% of the page's long side, resized to 224 px) and
answers one of 5: text upright / turned 90 ccw / 180 / 270 ccw / no text here.  A page is judged by all its tiles at
once: the trunk runs fully convolutionally on the page at 896 px (one pass = every 224 px window at stride 32), tiles
vote with their text probability, and the page is turned only when the vote is one-sided.  Training tiles come from the
layout sets' labelled text boxes (all scripts incl. vertical CJK), half through the phone-photo warp; "no text" tiles
are background, margins, desks.

  CUDA_VISIBLE_DEVICES=1 ../venv_hf/bin/python -u train_orient.py --out runs/orient1
"""
import argparse, json, math, os, random, time, copy
import modelopt.torch.quantization as mtq
import modelopt.torch.opt as mto
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import photo_data as P
import resource as _rs                      # 2026-11-13: 3090 soft limit 1024 -> 'Too many open files' in DataLoader
_rs.setrlimit(_rs.RLIMIT_NOFILE, (_rs.getrlimit(_rs.RLIMIT_NOFILE)[1],) * 2)

TILE = 224
PAGE = 896
FIG = P.CLASSES.index("fig")


class Block(nn.Module):
    def __init__(self, i, o, s):
        super().__init__()
        self.a = nn.Conv2d(i, o, 3, s, 1, bias=False); self.ba = nn.BatchNorm2d(o)
        self.b = nn.Conv2d(o, o, 3, 1, 1, bias=False); self.bb = nn.BatchNorm2d(o)
        self.sc = None if (i == o and s == 1) else nn.Sequential(nn.Conv2d(i, o, 1, s, bias=False), nn.BatchNorm2d(o))

    def forward(self, x):
        y = self.bb(self.b(F.relu(self.ba(self.a(x)), True)))
        return F.relu(y + (x if self.sc is None else self.sc(x)), True)


class OriNet(nn.Module):
    """ResNet-18-shaped, 5 outputs: 0..3 = text turned k x 90 degrees counter-clockwise, 4 = no text."""

    def __init__(self, w=(32, 64, 128, 256), n=5):          # 2.8M params; the 48-wide first try ran 100 tiles/s on an M40
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, w[0] // 2, 3, 2, 1, bias=False), nn.BatchNorm2d(w[0] // 2), nn.ReLU(True),
                                  nn.Conv2d(w[0] // 2, w[0], 3, 2, 1, bias=False), nn.BatchNorm2d(w[0]), nn.ReLU(True))
        L, i = [], w[0]
        for k, o in enumerate(w):
            L += [Block(i, o, 1 if k == 0 else 2), Block(o, o, 1)]; i = o
        self.body = nn.Sequential(*L); self.fc = nn.Linear(i, n)

    def forward(self, x):
        return self.fc(self.body(self.stem(x)).mean((2, 3)))

    def dense(self, x, win=7):
        """Every win*32 px window at stride 32 -> logits (B, 5, h, w)."""
        f = F.avg_pool2d(self.body(self.stem(x)), win, 1) if min(x.shape[-2:]) >= win * 32 else self.body(self.stem(x)).mean((2, 3), keepdim=True)
        return F.conv2d(f, self.fc.weight[:, :, None, None], self.fc.bias)


def page_vote(net, img_rgb, dev="cpu"):
    """-> (votes[4], text mass): votes[k] = sum over windows of P(text turned k), mass = mean P(text)."""
    h, w = img_rgb.shape[:2]; s = PAGE / max(h, w)
    nw, nh = max(TILE, int(round(w * s / 32)) * 32), max(TILE, int(round(h * s / 32)) * 32)
    x = cv2.resize(np.ascontiguousarray(img_rgb), (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    x = torch.from_numpy(x).permute(2, 0, 1)[None].float().to(dev) / 255 - .5
    with torch.no_grad():
        p = net.dense(x).softmax(1)[0]
    v = p[:4].flatten(1).sum(1)
    return v.cpu().numpy(), float(1 - p[4].mean())


def page_probs(net, img_rgb, side=PAGE, dev="cpu"):
    """-> (windows, 5) probabilities, every 224 px window at stride 32 of the page scaled to `side` on its long edge."""
    h, w = img_rgb.shape[:2]; s = side / max(h, w)
    nw, nh = max(TILE, int(round(w * s / 32)) * 32), max(TILE, int(round(h * s / 32)) * 32)
    x = cv2.resize(np.ascontiguousarray(img_rgb), (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    with torch.no_grad():
        p = net.dense(torch.from_numpy(x).permute(2, 0, 1)[None].float().to(dev) / 255 - .5).softmax(1)[0]
    return p.flatten(1).t().cpu().numpy()


def decide_ll(p, t=1.0, min_w=2.0):
    """The served rule (orient_own_eval.py `loglik`): windows are independent readers.  L_k = sum_w P(text) log P(k | text);
    turn to k only when it beats "upright" by >= t nats per unit of text weight.  -> (k, margin, text weight)"""
    w = 1 - p[:, 4]; q = p[:, :4] / np.maximum(w[:, None], 1e-6)
    L = (w[:, None] * np.log(q + 1e-4)).sum(0); k = int(L.argmax()); m = float((L[k] - L[0]) / max(w.sum(), 1e-6))
    return (k if k and m >= t and w.sum() >= min_w else 0), m, float(w.sum())


def decide(v, mass, conf=.8, min_mass=.03):
    """votes -> k (0 = leave; k = the page is turned k x 90 ccw, undo with k x 90 clockwise)."""
    k = int(np.argmax(v))
    return k if k and v[k] / max(v.sum(), 1e-9) >= conf and mass >= min_mass else 0


def jitter(t, r):
    t = t.astype(np.float32)
    if r.random() < .7:
        t = (t - 128) * r.uniform(.6, 1.3) + 128 + r.uniform(-30, 30)
    if r.random() < .25:
        t = cv2.GaussianBlur(t, (0, 0), r.uniform(.4, 1.4))
    if r.random() < .25:
        k = r.uniform(.45, .9); n = max(32, int(TILE * k))
        t = cv2.resize(cv2.resize(t, (n, n), interpolation=cv2.INTER_AREA), (TILE, TILE), interpolation=cv2.INTER_LINEAR)
    if r.random() < .3:
        t = t + np.random.standard_normal(t.shape).astype(np.float32) * r.uniform(2, 9)
    if r.random() < .04:
        t = 255 - t
    return np.clip(t, 0, 255).astype(np.uint8)


def tiles(img, b, c, r, T, aug=True):
    """T tiles from one page -> [(tile uint8 224x224x3, label)].  Text coverage from the labelled boxes (fig = unknown)."""
    H, W = img.shape[:2]; q = 8
    text = np.zeros((H // q + 1, W // q + 1), np.float32); fig = np.zeros_like(text)
    for (x0, y0, x1, y1), k in zip(b, c):
        (fig if k == FIG else text)[int(y0) // q:int(y1) // q + 1, int(x0) // q:int(x1) // q + 1] = 1
    out = []
    if min(H, W) < 16:                                   # 2026-11-13: a sliver page (< 48 px) crashed randint(0, W - s) 3x in orient3
        return out
    for _ in range(T):
        want_text = r.random() < .82
        for _try in range(14):
            s = min(max(48, int(math.exp(r.uniform(math.log(.12), math.log(.40))) * max(H, W))), H, W)
            x0, y0 = r.randint(0, W - s), r.randint(0, H - s)
            ct = float(text[y0 // q:(y0 + s) // q + 1, x0 // q:(x0 + s) // q + 1].mean()); cf = float(fig[y0 // q:(y0 + s) // q + 1, x0 // q:(x0 + s) // q + 1].mean())
            if want_text and ct >= .25 and cf <= .2:
                lab = r.randrange(4); break
            if not want_text and ct <= .02 and cf <= .02:
                lab = 4; break
        else:
            continue
        t = img[y0:y0 + s, x0:x0 + s]
        t = cv2.resize(np.ascontiguousarray(t), (TILE, TILE), interpolation=cv2.INTER_AREA if s > TILE else cv2.INTER_LINEAR)
        t = np.rot90(t, lab if lab < 4 else r.randrange(4))
        out.append((jitter(t, r) if aug else np.ascontiguousarray(t), lab))
    return out


def vert_page(r):
    """Synthetic vertical Han / Nom page (Dolphin/wordlid_vert/vertgen.py, tuned to the owner's T1 book scans): the real
    book showed orient1 calling EVERY upright Han Nom page sideways -- HJDataset's modern Japanese did not cover it.
    -> (RGB page with white margins, one text box, class para)"""
    import sys
    if "/mnt/storage/lib_project/models/Dolphin/wordlid_vert" not in sys.path:
        sys.path.insert(0, "/mnt/storage/lib_project/models/Dolphin/wordlid_vert")
    import vertgen
    d = None
    while d is None:
        d = (vertgen.make_vert_tight if r.random() < .6 else vertgen.make_vert)(r)
    im = np.array(d["image"].convert("L")); h, w = im.shape
    if r.random() < .5:                                    # some pages are set larger / smaller than the generator's range
        f = r.uniform(.6, 1.6); im = cv2.resize(im, (max(32, int(w * f)), max(32, int(h * f))), interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC); h, w = im.shape
    l, t, rr, b = [int(r.uniform(.03, .25) * m) for m in (w, h, w, h)]
    page = np.full((h + t + b, w + l + rr), r.randint(225, 255), np.uint8); page[t:t + h, l:l + w] = np.minimum(page[t:t + h, l:l + w], im)
    return np.stack([page] * 3, -1), np.array([[l, t, l + w, t + h]], np.float32), np.array([P.CLASSES.index("para")])


class Mix(Dataset):
    def __init__(self, length, weights, p_photo=.5, T=4):
        self.m = P.manifest(); self.length, self.dom, self.w, self.p, self.T = length, list(weights), list(weights.values()), p_photo, T

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        wi = torch.utils.data.get_worker_info()
        r = random.Random((i * 2654435761 + (wi.seed if wi else 0)) % 2**32)
        np.random.seed(r.randrange(2**31)); cv2.setRNGSeed(r.randrange(2**31))
        while True:
            dom = r.choices(self.dom, self.w)[0]
            img, b, c = vert_page(r) if dom == "vert" else P.load_sample(*r.choice(self.m[dom]))
            if not len(b):
                continue
            if r.random() < self.p:
                img, b, c = P.photo(img, b, c, r)
            ts = tiles(img, b, c, r, self.T)
            if ts:
                return torch.from_numpy(np.stack([t for t, _ in ts])), torch.tensor([l for _, l in ts])


def collate(batch):
    return torch.cat([b[0] for b in batch]), torch.cat([b[1] for b in batch])


class ValPages(Dataset):
    """Held-out pages: (image as the service would get it, truth k).  4 variants per page: clean/photo x upright/turned."""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items) * 4

    def __getitem__(self, j):
        i, var = divmod(j, 4)
        r = random.Random(77 + i); np.random.seed(77 + i); cv2.setRNGSeed(77 + i)
        img, b, c = P.load_sample(*self.items[i])
        ph = P.photo(img, b, c, r)[0] if var >= 2 else img
        k = r.choice([1, 2, 3]) if var % 2 else 0
        return np.ascontiguousarray(np.rot90(ph, k)), k, var >= 2


@torch.no_grad()
def evaluate(net, dev, sets, confs=(.6, .8, .9)):
    net.eval(); res = {}
    for name, items in sets.items():
        st = {c: {"clean": [0, 0, 0, 0, 0], "photo": [0, 0, 0, 0, 0]} for c in confs}      # false turns, upright n, right turns, turned n, wrong angle
        for img, k, photo in DataLoader(ValPages(items), batch_size=None, num_workers=8):
            v, mass = page_vote(net, np.asarray(img), dev)
            for c in confs:
                d = decide(v, mass, c); s = st[c]["photo" if photo else "clean"]
                if int(k) == 0:
                    s[0] += d != 0; s[1] += 1
                else:
                    s[2] += d == int(k); s[3] += 1; s[4] += d not in (0, int(k))
        res[name] = {str(c): v for c, v in st.items()}
    return res


def show(res, c):
    tot = [0] * 10
    for name, r in res.items():
        a, b = r[str(c)]["clean"], r[str(c)]["photo"]
        print(f"   {name:10s} conf {c}: clean false {a[0]}/{a[1]} recall {a[2]}/{a[3]} wrong {a[4]} | photo false {b[0]}/{b[1]} recall {b[2]}/{b[3]} wrong {b[4]}", flush=True)
        tot = [x + y for x, y in zip(tot, a + b)]
    print(f"   ALL        conf {c}: clean false {tot[0]}/{tot[1]} recall {tot[2]}/{tot[3]} wrong {tot[4]} | photo false {tot[5]}/{tot[6]} recall {tot[7]}/{tot[8]} wrong {tot[9]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--steps", type=int, default=12000); ap.add_argument("--pages", type=int, default=32)
    ap.add_argument("--workers", type=int, default=24); ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--init", default=""); ap.add_argument("--val-n", type=int, default=50); ap.add_argument("--tiles", type=int, default=12)
    ap.add_argument("--mix", default="thesis=.24,doclaynet=.10,d4la=.12,comics=.05,slides=.06,screens=.04,m6doc=.10,badlad=.08,indicdlp=.08,docsynth=.05,hj=.08")
    a = ap.parse_args()
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out, exist_ok=True)
    net = OriNet().to(dev); ema = OriNet().to(dev)
    if a.init:
        net.load_state_dict(torch.load(a.init, map_location=dev))
    ema.load_state_dict(net.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=.05)
    sets = {k: [tuple(y.replace("/mnt/storage/lib_project/models/", P.M + "/") if isinstance(y, str) else y for y in x) for x in v[:a.val_n]] for k, v in json.load(open(os.path.join(P.M, "layout_teacher/train/runs/orient/val_items.json"))).items()}
    ds = Mix(a.pages * (a.steps + 100), {k: float(v) for k, v in (kv.split("=") for kv in a.mix.split(","))}, T=a.tiles)
    it = iter(DataLoader(ds, batch_size=a.pages, shuffle=True, num_workers=a.workers, collate_fn=collate, persistent_workers=True, prefetch_factor=3, drop_last=True))
    print("QAT: calibrating from mixed training tiles", flush=True)
    def calibration(m):
        m.eval()
        with torch.no_grad():
            for j in range(12):
                x, _ = next(it)
                x = x.to(dev).permute(0, 3, 1, 2).float() / 255 - .5
                m(x)
                print("calibration", j + 1, "/12", flush=True)
    mtq.quantize(net, mtq.INT8_DEFAULT_CFG, calibration)
    ema = copy.deepcopy(net)
    for p in ema.parameters(): p.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=.01)
    print("QAT calibration complete; beginning training", flush=True)
    log = lambda **k: open(f"{a.out}/log.jsonl", "a").write(json.dumps(k) + "\n")
    t0 = time.time(); run = [0., 0, 0]
    AMP = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(os.environ.get("LAYOUT_AMP", ""))     # 2026-11-13: 3090 tensor cores
    scaler = torch.amp.GradScaler("cuda", enabled=AMP == torch.float16)
    for step in range(1, a.steps + 1):
        x, y = next(it)
        x = x.to(dev).permute(0, 3, 1, 2).float() / 255 - .5; y = y.to(dev)
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1, step / 300) * (.5 * (1 + math.cos(math.pi * step / a.steps)) * .98 + .02)
        net.train()
        with torch.autocast("cuda", dtype=AMP, enabled=AMP is not None):
            z = net(x)
        z = z.float(); loss = F.cross_entropy(z, y, label_smoothing=.05)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0); scaler.step(opt); scaler.update()
        with torch.no_grad():
            k = min(.999, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), net.parameters()):
                pe.mul_(k).add_(pm, alpha=1 - k)
            for be, bm in zip(ema.buffers(), net.buffers()):
                be.copy_(bm)
        run[0] += float(loss); run[1] += int((z.argmax(1) == y).sum()); run[2] += len(y)
        if step % 100 == 0:
            print(f"step {step} loss {run[0]/100:.4f} tile acc {run[1]/run[2]:.4f} {(time.time()-t0)/100:.2f}s/step", flush=True)
            log(step=step, loss=run[0] / 100, acc=run[1] / run[2]); run = [0., 0, 0]; t0 = time.time()
        if step % a.eval_every == 0 or step == a.steps:
            r = evaluate(ema, dev, sets); log(step=step, **r); print("eval", step, flush=True)
            for c in (.8, .9):
                show(r, c)
            mto.save(ema, f"{a.out}/ori_qat_{step}.pth"); t0 = time.time()
    print("ORI_DONE", flush=True)


if __name__ == "__main__":
    main()
