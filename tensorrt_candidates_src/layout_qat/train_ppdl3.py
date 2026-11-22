#!/usr/bin/env python
"""Fine-tune PP-DocLayoutV3 (transformers 5.22, layout_teacher/venv_hf) on the full mix, 12-class taxonomy, fp32, DDP on the M40s.

The HF wrapper refuses labels, so this calls the inner model and RTDetrForObjectDetectionLoss directly. Decoder queries are seeded from a
mask head (mask_enhanced), so that head is kept honest on a drifting encoder with a rectangle-mask loss on matched queries; the
reading-order head stays frozen. BatchNorm stats are frozen (batches are small); evaluation uses EMA weights.

  torchrun --nproc_per_node 2 train_ppdl3.py --out runs/r1 --steps 60000
"""
import argparse, hashlib, json, math, os, random, time, copy
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForObjectDetection
from transformers.loss.loss_rt_detr import RTDetrForObjectDetectionLoss, RTDetrHungarianMatcher
import photo_data as P

SIDE = 896                  # 2026-11-09 owner: 896 letterbox, black pad, as the yolo canvas and the readers (r1/r2: 800 squash)
HF = os.path.expanduser("~/codex_trt_qat_20261119/source/layout_teacher/weights/ppdl3_hf")
OLD = "abstract algorithm aside_text chart content display_formula doc_title figure_title footer footer_image footnote formula_number header header_image image inline_formula number paragraph_title reference reference_content seal table text vertical_text vision_footnote".split()
FROM = {"cap": "figure_title", "catalogue": "content", "code": "algorithm", "equ": "display_formula", "fig": "image", "foot": "number",
        "header": "header", "list": "text", "para": "text", "reference": "reference_content", "sec": "paragraph_title", "tab": "table"}


def build_model():
    m = AutoModelForObjectDetection.from_pretrained(HF)
    idx = [OLD.index(FROM[c]) for c in P.CLASSES]
    n, core = len(P.CLASSES), m.model
    def head(old):
        new = nn.Linear(old.in_features, n)
        new.weight.data = old.weight.data[idx].clone(); new.bias.data = old.bias.data[idx].clone()
        return new
    core.enc_score_head = head(core.enc_score_head)
    core.decoder.class_embed = head(core.decoder.class_embed)
    old = core.denoising_class_embed
    new = nn.Embedding(n + 1, old.embedding_dim, padding_idx=n)
    new.weight.data[:n] = old.weight.data[idx].clone(); new.weight.data[n] = 0
    core.denoising_class_embed = new
    m.config.num_labels = n
    for k, v in dict(auxiliary_loss=True, weight_loss_vfl=1.0, weight_loss_bbox=5.0, weight_loss_giou=2.0, focal_loss_alpha=.75,
                     focal_loss_gamma=2.0, use_focal_loss=True).items():      # RT-DETR defaults; this config omits the training keys
        setattr(m.config, k, v)
    m.config.id2label = dict(enumerate(P.CLASSES)); m.config.label2id = {c: i for i, c in enumerate(P.CLASSES)}
    for p in core.decoder_order_head.parameters():
        p.requires_grad_(False)
    for p in core.decoder_global_pointer.parameters():
        p.requires_grad_(False)
    return m


# Sources whose taxonomy has no code class: code-looking text there is labelled para, so a code prediction on their
# pages must not be punished (r1 lost thesis code word acc .546 -> .258 to exactly that). Gradient into the code
# logit is cut for these pages; every other class is trained as before.
MAX_GT = 150               # r3: 567-box BaDLAD pages OOM (dn queries + rectangle-mask loss grow with GT count); 300 queries anyway
NOCODE = {"doclaynet", "d4la", "badlad", "indicdlp", "hj"}
CODE = P.CLASSES.index("code")


_FLAT = None


def photo_served(img, b, c, r):
    """A phone photo as the detector is SERVED it.  LAYOUT_FLAT_CKPT set (2026-11-13, r5): the photo goes through the page
    straightening step (flat_apply.py, same code path as docparse) and the boxes are carried along; else the raw photo."""
    global _FLAT
    ck = os.environ.get("LAYOUT_FLAT_CKPT")
    if not ck:
        return P.photo(img, b, c, r)
    if _FLAT is None:
        from flat_apply import Flat
        _FLAT = Flat(ck)
    om = {}; hard = r.random() < float(os.environ.get("LAYOUT_FLAT_HARD", ".3"))
    ph, pb, pc = P.photo(img, b, c, r, out_map=om, hard=hard)
    if not len(pb):
        return ph, pb, pc
    out, fb, keep = _FLAT.apply(ph, om["poly"])
    return out, fb[keep], pc[keep]


class Mix(Dataset):
    def __init__(self, length, p_photo, weights, seed=0):
        self.m = P.manifest(); self.length, self.p, self.dom, self.w, self.seed = length, p_photo, list(weights), list(weights.values()), seed

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        wi = torch.utils.data.get_worker_info()
        r = random.Random((i * 2654435761 + (wi.seed if wi else self.seed)) % 2**32)
        np.random.seed(r.randrange(2**31)); cv2.setRNGSeed(r.randrange(2**31))
        for _ in range(20):
            dom = r.choices(self.dom, self.w)[0]
            p, l, t = r.choice(self.m[dom])
            img, b, c = P.load_sample(p, l, t)
            if 0 < len(b) <= MAX_GT:
                break
        if r.random() < self.p:
            img, b, c = photo_served(img, b, c, r)
        else:
            img = P.clean_jitter(img, b, r)
        if not len(b):
            return self.__getitem__(i + 1)
        x, sc, ox, oy = P.letterbox(img, SIDE)
        b = (b * sc + np.array([ox, oy, ox, oy], np.float32)) / SIDE
        box = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)
        return torch.from_numpy(x).permute(2, 0, 1).float() / 255, {"class_labels": torch.from_numpy(c), "boxes": torch.from_numpy(box).float(),
                                                                    "nocode": torch.tensor(dom in NOCODE)}


def collate(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


class Net(nn.Module):
    """Forward = total loss. Inside DDP this is what gets wrapped."""

    def __init__(self, m, mask_w):
        super().__init__()
        self.m, self.mask_w = m, mask_w
        self.matcher = RTDetrHungarianMatcher(m.config)

    def forward(self, x, targets):
        nocode = torch.stack([t.pop("nocode") for t in targets]) if "nocode" in targets[0] else None
        o = self.m.model(x, labels=targets)
        il, ir, el = o.intermediate_logits, o.intermediate_reference_points, o.enc_topk_logits
        if nocode is not None and bool(nocode.any()):
            col = torch.arange(il.shape[-1], device=x.device) == CODE
            il = torch.where(nocode[:, None, None, None] & col, il.detach(), il)
            el = torch.where(nocode[:, None, None] & col, el.detach(), el)
        loss, ld, _ = RTDetrForObjectDetectionLoss(il[:, -1], targets, x.device, ir[:, -1], self.m.config, il, ir,
                                                   enc_topk_logits=el, enc_topk_bboxes=o.enc_topk_bboxes,
                                                   denoising_meta_values=o.denoising_meta_values)
        mk = o.out_masks[:, -1]
        dn = o.denoising_meta_values["dn_num_split"][0] if o.denoising_meta_values else 0
        idx = self.matcher({"logits": il[:, -1, dn:], "pred_boxes": ir[:, -1, dn:]}, targets)
        H, W = mk.shape[-2:]; bce, dice, n = 0., 0., 0
        for bi, (q, t) in enumerate(idx):
            if not len(q):
                continue
            pred = mk[bi, dn:][q.to(x.device)]
            b = targets[bi]["boxes"][t.to(x.device)]
            yy = (torch.arange(H, device=x.device) + .5) / H; xx = (torch.arange(W, device=x.device) + .5) / W
            gx = ((xx[None] >= b[:, 0, None] - b[:, 2, None] / 2) & (xx[None] <= b[:, 0, None] + b[:, 2, None] / 2)).float()
            gy = ((yy[None] >= b[:, 1, None] - b[:, 3, None] / 2) & (yy[None] <= b[:, 1, None] + b[:, 3, None] / 2)).float()
            gt = gy[:, :, None] * gx[:, None, :]
            bce = bce + F.binary_cross_entropy_with_logits(pred, gt, reduction="sum") / (H * W)
            p = pred.sigmoid().flatten(1); g = gt.flatten(1)
            dice = dice + (1 - (2 * (p * g).sum(1) + 1) / (p.sum(1) + g.sum(1) + 1)).sum(); n += len(q)
        ml = (bce + dice) / max(n, 1)
        return loss + self.mask_w * ml, {"det": loss.detach(), "mask": ml.detach() if torch.is_tensor(ml) else torch.tensor(0.)}


@torch.no_grad()
def predict(m, x, top=300):
    o = m(pixel_values=x)
    prob = o.logits.sigmoid(); B, Q, C = prob.shape
    s, i = prob.flatten(1).topk(top, 1)
    q = i // C; c = i % C
    b = o.pred_boxes.gather(1, q[..., None].expand(-1, -1, 4))
    return s, c, torch.stack([b[..., 0] - b[..., 2] / 2, b[..., 1] - b[..., 3] / 2, b[..., 0] + b[..., 2] / 2, b[..., 1] + b[..., 3] / 2], -1)


def val_sets():
    """Fixed-seed sets: clean thesis val, clean D4LA test sample, and the same thesis/DLN/D4LA pages photo-warped."""
    M = P.M; r = random.Random(7)
    d = f"{M}/yolo26m_merged_v14"; th = [(f"{d}/images/val/{f}", f"{d}/labels/val/{f[:-4]}.txt", 14) for f in sorted(os.listdir(f"{d}/images/val"))]
    d = f"{M}/layout_teacher/data_ext/d4la/test"; dl = [(f"{d}/images/{f}", f"{d}/labels/{f[:-4]}.txt", 12) for f in sorted(os.listdir(f"{d}/images"))]
    r.shuffle(dl); r.shuffle(th)
    out = {"thesis": (th[:500], False), "d4la": (dl[:400], False), "photo_thesis": (th[500:800], True), "photo_d4la": (dl[400:700], True)}
    for k, v in P.web_split().items():          # agreement with the stock teacher on held-out books/decks
        it = v["val"][:]; r.shuffle(it)
        seen, uniq = set(), []                  # r4: the comics val held one newspaper page 6x under different names
        for x in it:
            h = hashlib.md5(open(x[0], "rb").read()).hexdigest()
            if h not in seen:
                seen.add(h); uniq.append(x)
        it = uniq
        if it:
            out[k] = (it[:300], False); out["photo_" + k] = (it[:300], True)
    for k, v in P.ext_split().items():          # r3 external sets: held-out pages, clean + photo-warped
        it = v["val"][:]; r.shuffle(it)
        if it:
            out[k] = (it[:300], False); out["photo_" + k] = (it[:200], True)
    return out


class ValData(Dataset):
    def __init__(self, items, warp):
        self.items, self.warp = items, warp

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        r = random.Random(1000 + i); np.random.seed(1000 + i); cv2.setRNGSeed(1000 + i)
        p, l, t = self.items[i]
        img, b, c = P.load_sample(p, l, t)
        if self.warp:
            img, b, c = photo_served(img, b, c, r)
        x, sc, ox, oy = P.letterbox(img, SIDE)
        b = (b * sc + np.array([ox, oy, ox, oy], np.float32)) / SIDE if len(b) else b
        return torch.from_numpy(x).permute(2, 0, 1).float() / 255, b, c


def coco_ap(gts, dts):
    import contextlib, io
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    images, anns = [], []
    for i, (b, c) in enumerate(gts, 1):
        images.append({"id": i, "width": 1000, "height": 1000})
        for q, k in zip(b, c):
            x, y, X, Y = [float(v) * 1000 for v in q]
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": int(k) + 1, "bbox": [x, y, X - x, Y - y], "area": (X - x) * (Y - y), "iscrowd": 0})
    g = COCO(); g.dataset = {"images": images, "annotations": anns, "categories": [{"id": j + 1, "name": n} for j, n in enumerate(P.CLASSES)]}
    with contextlib.redirect_stdout(io.StringIO()):
        g.createIndex(); e = COCOeval(g, g.loadRes(dts), "bbox"); e.params.maxDets = [1, 10, 300]; e.evaluate(); e.accumulate()
    p = e.eval["precision"]
    per = {n: (float(p[:, :, j, 0, 2][p[:, :, j, 0, 2] > -1].mean()) if (p[:, :, j, 0, 2] > -1).any() else None) for j, n in enumerate(P.CLASSES)}
    # box-first (owner 2026-11-10): class-agnostic AP50:95 and AP75 -- are the regions found and tight, whatever the class?
    for a in anns:
        a["category_id"] = 1
    ga = COCO(); ga.dataset = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "region"}]}
    da = [dict(d, category_id=1) for d in dts]
    with contextlib.redirect_stdout(io.StringIO()):
        ga.createIndex(); ea = COCOeval(ga, ga.loadRes(da), "bbox"); ea.params.maxDets = [1, 10, 300]; ea.evaluate(); ea.accumulate()
    q = ea.eval["precision"][:, :, 0, 0, 2]
    per["_box"] = float(q[q > -1].mean()); per["_box75"] = float(q[5][q[5] > -1].mean())
    return float(p[:, :, :, 0, 2][p[:, :, :, 0, 2] > -1].mean()), per


@torch.no_grad()
def evaluate(m, device, sets, bs=8):
    m.eval(); out = {}
    for name, (items, warp) in sets.items():
        dl = DataLoader(ValData(items, warp), batch_size=bs, num_workers=8, collate_fn=lambda z: z)
        gts, dts = [], []
        for batch in dl:
            x = torch.stack([z[0] for z in batch]).to(device)
            s, c, b = predict(m, x)
            for j, z in enumerate(batch):
                gts.append((z[1], z[2])); iid = len(gts)
                keep = s[j] > .01
                for sc, k, q in zip(s[j][keep].tolist(), c[j][keep].tolist(), b[j][keep].tolist()):
                    dts.append({"image_id": iid, "category_id": k + 1, "bbox": [q[0] * 1000, q[1] * 1000, (q[2] - q[0]) * 1000, (q[3] - q[1]) * 1000], "score": sc})
        out[name], per = coco_ap(gts, dts)
        out[name + "_box"], out[name + "_box75"] = per.pop("_box"), per.pop("_box75")
        out[name + "_per"] = per
    return out


def freeze_bn(m):
    for x in m.modules():
        if isinstance(x, nn.modules.batchnorm._BatchNorm):
            x.eval()


def main():
    # NCCL timeout 3 h: rank 0 evaluates alone while the other ranks wait in a barrier; the r3 eval passes the 10-min default
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--mask-w", type=float, default=1.0)
    ap.add_argument("--p-photo", type=float, default=.5)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--probe", type=int, default=0, help="run N steps, print timing and memory, no eval")
    ap.add_argument("--mix", default="thesis=.4,doclaynet=.3,d4la=.3", help="domain=weight,...")
    ap.add_argument("--init", default="", help="EMA state dict to start from when there is no latest.pt")
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--accum", type=int, default=1, help="micro-batches per optimiser step (3090: --batch 6 --accum 2 = M40 DDP 2x6)")
    a = ap.parse_args()
    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl", timeout=__import__("datetime").timedelta(hours=3)); rank = dist.get_rank(); torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        rank = 0
    dev = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(rank + 1)
    os.makedirs(a.out, exist_ok=True)
    log = (lambda **k: open(f"{a.out}/log.jsonl", "a").write(json.dumps(k) + "\n")) if rank == 0 else (lambda **k: None)
    m = build_model().to(dev)
    net = Net(m, a.mask_w).to(dev)
    ema = copy.deepcopy(m).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    start = 0
    if os.path.exists(f"{a.out}/latest.pt"):
        ck = torch.load(f"{a.out}/latest.pt", map_location=dev); m.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); start = ck["step"]
    elif a.init:
        sd = torch.load(a.init, map_location=dev); m.load_state_dict(sd); ema.load_state_dict(sd)
    bb = [p for n, p in m.named_parameters() if p.requires_grad and "backbone" in n]
    rest = [p for n, p in m.named_parameters() if p.requires_grad and "backbone" not in n]
    opt = torch.optim.AdamW([{"params": bb, "lr": a.lr_backbone}, {"params": rest, "lr": a.lr}], weight_decay=1e-4)
    if start:
        opt.load_state_dict(ck["opt"])
    wrap = torch.nn.parallel.DistributedDataParallel(net, device_ids=[dev.index], find_unused_parameters=True) if ddp else net
    ds = Mix(a.batch * a.accum * (a.steps + 1000), a.p_photo, {k: float(v) for k, v in (kv.split("=") for kv in a.mix.split(","))}, seed=rank)
    if rank == 0 and start == 0:
        print({k: len(v) for k, v in ds.m.items()}, flush=True)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers, collate_fn=collate, drop_last=True, persistent_workers=True, prefetch_factor=3)
    sets = val_sets() if rank == 0 and not a.probe else None
    if rank == 0 and start == 0 and not a.probe:
        r0 = evaluate(ema, dev, sets); log(step=0, **r0); print("step0", {k: round(v, 3) for k, v in r0.items() if not k.endswith("_per")}, flush=True)
        torch.cuda.empty_cache()
    AMP = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(os.environ.get("LAYOUT_AMP", ""))
    scaler = torch.amp.GradScaler("cuda", enabled=AMP == torch.float16)
    if AMP is not None:
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    best = -1; step = start; t0 = time.time(); run = {"det": 0., "mask": 0.}; cnt = 0
    it = iter(dl)
    while step < a.steps:
        lr_scale = min(1, (step + 1) / a.warmup) * (.5 * (1 + math.cos(math.pi * step / a.steps)) * .95 + .05)
        opt.param_groups[0]["lr"], opt.param_groups[1]["lr"] = a.lr_backbone * lr_scale, a.lr * lr_scale
        m.train(); freeze_bn(m)
        opt.zero_grad(set_to_none=True); dsum = {"det": 0., "mask": 0.}
        for _ in range(a.accum):                                             # --accum: mean of micro-batch losses = DDP grad average
            x, tg = next(it)
            x = x.to(dev, non_blocking=True); tg = [{k: v.to(dev) for k, v in t.items()} for t in tg]
            with torch.autocast("cuda", dtype=AMP, enabled=AMP is not None):    # LAYOUT_AMP=fp16|bf16: tensor cores (3090); unset = fp32 (M40 runs)
                loss, d = wrap(x, tg)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            scaler.scale(loss / a.accum).backward()
            dsum["det"] += float(d["det"]) / a.accum; dsum["mask"] += float(d["mask"]) / a.accum
        d = dsum; scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(bb + rest, .1)
        scaler.step(opt); scaler.update()
        with torch.no_grad():
            k = min(.9997, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), m.parameters()):
                pe.mul_(k).add_(pm, alpha=1 - k)
            for be, bm in zip(ema.buffers(), m.buffers()):
                be.copy_(bm)
        step += 1; run["det"] += float(d["det"]); run["mask"] += float(d["mask"]); cnt += 1
        if rank == 0 and (step % 100 == 0 or (a.probe and step % 10 == 0)):
            sp = (time.time() - t0) / cnt
            print(f"step {step} det {run['det']/cnt:.2f} mask {run['mask']/cnt:.3f} grad {float(gn):.1f} {sp:.2f}s/step mem {torch.cuda.max_memory_allocated()/2**30:.1f}G", flush=True)
            log(step=step, det=run["det"] / cnt, mask=run["mask"] / cnt, sp=sp); run = {"det": 0., "mask": 0.}; cnt = 0; t0 = time.time()
        if a.probe and step >= a.probe:
            break
        if step % a.eval_every == 0 or step == a.steps:
            if ddp:
                dist.barrier()
            if rank == 0:
                torch.cuda.empty_cache()
                r = evaluate(ema, dev, sets); log(step=step, **r); torch.cuda.empty_cache()
                print("eval", step, {k: round(v, 3) for k, v in r.items() if not k.endswith("_per")}, flush=True)
                ck = {"model": m.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step}
                torch.save(ck, f"{a.out}/latest.tmp"); os.replace(f"{a.out}/latest.tmp", f"{a.out}/latest.pt")
                torch.save(ema.state_dict(), f"{a.out}/ema_{step}.pt")
                t0 = time.time(); run = {"det": 0., "mask": 0.}; cnt = 0
            if ddp:
                dist.barrier()
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
