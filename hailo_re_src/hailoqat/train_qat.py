"""Exact-numerics QAT for Hailo-8 (YOLO26 one2one head), DDP on all visible GPUs.

Student: QATNet (Hailo integer forward, STE backward) initialised from a level-0 HAR capture.
Teacher: the FP checkpoint; its one2one head outputs are distilled per anchor.
Loss = gt * one2one v8DetectionLoss + kd_cls * BCE(student logits, teacher probs)
     + kd_box * teacher-confidence-weighted L1 on box distances + ovf * accumulator-overflow penalty.

Launch: torchrun --nproc_per_node=2 train_qat.py --out RUN_DIR [opts]
"""
import argparse
import glob
import json
import math
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib  # noqa: E402

import graph26  # noqa: E402

ROOT = "/mnt/storage/lib_project/models/Hailo_model/hailo_re"


def head_names(hn):
    layers = hn["layers"]

    def find(b, i):
        tag = f"/model.23/one2one_cv{b}.{i}/one2one_cv{b}.{i}.2/Conv"
        return [k for k, v in layers.items() if tag in v.get("original_names", [])][0]

    return [(find(2, i), find(3, i)) for i in range(3)]


class Pages(torch.utils.data.Dataset):
    """Pages + YOLO labels; with `cache` (teacher_cache.py output) also the teacher's one2one boxes/score logits."""

    def __init__(self, img_dir, n=0, seed=0, cache="", imgsz=0):
        self.files = sorted(glob.glob(os.path.join(img_dir, "*")))
        if n:
            self.files = sorted(random.Random(seed).sample(self.files, n))
        self.cache, self._mm, self.imgsz = cache, None, imgsz
        if cache:
            self.meta = json.load(open(os.path.join(cache, "meta.json")))
            pos = {f: i for i, f in enumerate(open(os.path.join(cache, "files.txt")).read().splitlines())}
            missing = [f for f in self.files if f not in pos]
            assert not missing, f"{len(missing)} pages not in the teacher cache, e.g. {missing[0]}"
            self.cidx = [pos[f] for f in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        if self.imgsz and img.shape[:2] != (self.imgsz, self.imgsz):  # square pages: == evaluate.py's letterbox
            img = cv2.resize(img, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        lp = f.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        lab = np.loadtxt(lp, ndmin=2).astype(np.float32) if os.path.exists(lp) and os.path.getsize(lp) else np.zeros((0, 5), np.float32)
        if not self.cache:
            return torch.from_numpy(img), torch.from_numpy(lab)
        if self._mm is None:  # opened lazily so each DataLoader worker has its own maps
            from teacher_cache import open_cache
            m = self.meta
            self._mm = open_cache(self.cache, m["n"], m["nc"], m["anchors"], "r")
        j = self.cidx[i]
        return (torch.from_numpy(img), torch.from_numpy(lab), torch.from_numpy(np.array(self._mm[0][j])),
                torch.from_numpy(np.array(self._mm[1][j]).astype(np.float32)))


def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    cls, box, idx = [], [], []
    for i, b in enumerate(batch):
        lab = b[1]
        cls.append(lab[:, :1])
        box.append(lab[:, 1:5])
        idx.append(torch.full((len(lab),), i, dtype=torch.float32))
    out = {"cls": torch.cat(cls), "bboxes": torch.cat(box), "batch_idx": torch.cat(idx)}
    if len(batch[0]) == 4:
        out["t_boxes"] = torch.stack([b[2] for b in batch])
        out["t_scores"] = torch.stack([b[3] for b in batch])
    return imgs, out


def dequant(P, name, q, dev):
    r = P.conv[name]
    sc = torch.as_tensor(np.asarray(r["out_scale"], np.float32), device=dev).reshape(1, -1, 1, 1)
    zp = torch.as_tensor(np.asarray(r["out_zp"], np.float32), device=dev).reshape(1, -1, 1, 1)
    return (q - zp) * sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hn", default=f"{ROOT}/yolo26m_v14sec_L0.hn")
    ap.add_argument("--cap", default=f"{ROOT}/capture_y26g")
    ap.add_argument("--pt", default=f"{ROOT}/models/v14sec/best.pt")
    ap.add_argument("--train", default=f"{ROOT}/data_v14sec/train/images")
    ap.add_argument("--val", default=f"{ROOT}/data_v14sec/val/images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=4, help="per GPU")
    ap.add_argument("--lr_k", type=float, default=5e-3)
    ap.add_argument("--lr_b", type=float, default=0.5)
    ap.add_argument("--gt", type=float, default=1.0)
    ap.add_argument("--kd_cls", type=float, default=1.0)
    ap.add_argument("--kd_box", type=float, default=1.0)
    ap.add_argument("--ovf", type=float, default=1e-3)
    ap.add_argument("--k_radius", type=float, default=0.0,
                    help=">0: keep each latent kernel within +-R of its level-0 integer (AdaRound-like neighbourhood)")
    ap.add_argument("--b_radius", type=float, default=0.0, help=">0: same bound for the integer biases")
    ap.add_argument("--warmup", type=int, default=0, help="linear LR warmup steps")
    ap.add_argument("--apu_train", action="store_true", help="train the APU piecewise tables (mantissa + offset)")
    ap.add_argument("--lr_apu_m", type=float, default=0.05)
    ap.add_argument("--lr_apu_o", type=float, default=0.5)
    ap.add_argument("--apu_m_radius", type=float, default=32.0)
    ap.add_argument("--apu_o_radius", type=float, default=256.0)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--eval_n", type=int, default=200)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--n_train", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=0, help="resize the (square) pages to the model input size")
    ap.add_argument("--teacher_cache", default="", help="teacher_cache.py dir: skip the teacher forward")
    ap.add_argument("--impl", default="qatfast", help="qatfast (fused sm_52 kernels) or qatnet (reference)")
    ap.add_argument("--profile_steps", type=int, default=0, help="time the phases of N steps, then exit")
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = True  # forward exactness is guarded by qatfast's flags
    QATNet = importlib.import_module(args.impl).QATNet

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        rank, world = 0, 1
    dev = torch.device("cuda", torch.cuda.current_device())
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(rank)

    hn = json.load(open(args.hn))
    P = graph26.Params(args.cap, dev)
    student = QATNet(hn, P, dev).to(dev)
    heads = head_names(hn)

    from ultralytics import YOLO
    from ultralytics.utils.loss import v8DetectionLoss
    tmodel = YOLO(args.pt).model.to(dev).float().eval()
    for p_ in tmodel.parameters():
        p_.requires_grad_(False)
    det = tmodel.model[-1]
    feats_box = {}
    det.register_forward_pre_hook(lambda m, inp: feats_box.__setitem__("x", inp[0]))
    if not hasattr(tmodel, "args") or isinstance(tmodel.args, dict):
        from types import SimpleNamespace
        a = tmodel.args if isinstance(getattr(tmodel, "args", None), dict) else {}
        tmodel.args = SimpleNamespace(box=a.get("box", 7.5), cls=a.get("cls", 0.5), dfl=a.get("dfl", 1.5),
                                      epochs=a.get("epochs", 40))
    crit = v8DetectionLoss(tmodel, tal_topk=7, tal_topk2=1)

    if args.apu_train:
        student.enable_apu_training(args.apu_m_radius, args.apu_o_radius)
    apu_m = [a.mant_lat for a in student.apu_modules().values() if a.trainable]
    apu_o = [a.offs_lat for a in student.apu_modules().values() if a.trainable]
    params_k = [m.kernel for m in student.mac.values()]
    params_b = [m.bias for m in student.mac.values()]
    groups = [{"params": params_k, "lr": args.lr_k}, {"params": params_b, "lr": args.lr_b}]
    if apu_m:
        groups += [{"params": apu_m, "lr": args.lr_apu_m}, {"params": apu_o, "lr": args.lr_apu_o}]
    opt = torch.optim.Adam(groups)
    k0 = [p_.detach().clone() for p_ in params_k]
    b0 = [p_.detach().clone() for p_ in params_b]

    ds = Pages(args.train, args.n_train, cache=args.teacher_cache, imgsz=args.imgsz)
    sampler = torch.utils.data.distributed.DistributedSampler(ds, world, rank, shuffle=True) if ddp else None
    dl = torch.utils.data.DataLoader(ds, args.bs, sampler=sampler, shuffle=sampler is None, num_workers=6,
                                     collate_fn=collate, drop_last=True, persistent_workers=True)
    total = int(args.epochs * len(dl)) if not args.max_steps else args.max_steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup if args.warmup else 1.0) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    log = open(os.path.join(args.out, f"train_rank{rank}.log"), "a")

    step, t0 = 0, time.time()
    epoch = 0
    phases = {}

    def mark(name, t_prev):
        if args.profile_steps:
            torch.cuda.synchronize()
            t = time.time()
            phases[name] = phases.get(name, 0.0) + t - t_prev
            return t
        return t_prev
    while step < total:
        if sampler:
            sampler.set_epoch(epoch)
        tp = time.time()
        for imgs, batch in dl:
            tp = mark("data", tp)
            imgs = imgs.to(dev, non_blocking=True).float()
            batch = {k: v.to(dev) for k, v in batch.items()}
            if "t_scores" in batch:
                t1 = {"boxes": batch.pop("t_boxes"), "scores": batch.pop("t_scores")}
            else:
                with torch.no_grad():
                    tmodel(imgs.permute(0, 3, 1, 2) / 255.0)
                    t1 = det.forward_head(feats_box["x"], **det.one2one)
            tp = mark("teacher", tp)
            out = student(imgs)
            tp = mark("student_fwd", tp)
            bx = [dequant(P, b, out[b], dev) for b, _ in heads]
            cl = [dequant(P, c, out[c], dev) for _, c in heads]
            bs = imgs.shape[0]
            s_boxes = torch.cat([t.reshape(bs, 4, -1) for t in bx], -1)
            s_scores = torch.cat([t.reshape(bs, det.nc, -1) for t in cl], -1)
            sdict = {"boxes": s_boxes, "scores": s_scores, "feats": bx}
            l_gt, items = crit.loss(sdict, batch)
            l_gt = l_gt.sum() / bs if l_gt.dim() else l_gt
            t_prob = t1["scores"].sigmoid()
            l_cls = F.binary_cross_entropy_with_logits(s_scores, t_prob, reduction="mean") * det.nc
            w = t_prob.amax(1, keepdim=True)
            l_box = ((s_boxes - t1["boxes"]).abs() * w).sum() / (w.sum() * 4 + 1e-6)
            l_ovf = student.overflow_penalty()
            loss = args.gt * l_gt + args.kd_cls * l_cls + args.kd_box * l_box + args.ovf * l_ovf
            tp = mark("loss", tp)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            tp = mark("backward", tp)
            if ddp:  # one flattened all-reduce instead of 252 small ones
                allp = params_k + params_b + apu_m + apu_o
                grads = [p_.grad if p_.grad is not None else torch.zeros_like(p_) for p_ in allp]
                flat = torch.cat([g.reshape(-1) for g in grads])
                dist.all_reduce(flat)
                flat /= world
                off = 0
                for p_, g in zip(allp, grads):
                    p_.grad = flat[off:off + g.numel()].view_as(p_)
                    off += g.numel()
            opt.step()
            with torch.no_grad():
                for rad, ps, p0s in ((args.k_radius, params_k, k0), (args.b_radius, params_b, b0)):
                    if rad > 0:
                        for p_, q_ in zip(ps, p0s):
                            p_.copy_(torch.minimum(torch.maximum(p_, q_ - rad), q_ + rad))
            sched.step()
            tp = mark("opt", tp)
            step += 1
            if args.profile_steps and step == 2:
                phases.clear()
            if args.profile_steps and step == args.profile_steps + 2:
                if rank == 0:
                    n = args.profile_steps
                    print("PHASES s/step " + " ".join(f"{k} {v / n:.3f}" for k, v in phases.items())
                          + f" total {sum(phases.values()) / n:.3f}", flush=True)
                return
            if rank == 0 and (step % 20 == 0 or step == 1):
                msg = (f"step {step}/{total} loss {loss.item():.4f} gt {l_gt.item():.4f} kd_cls {l_cls.item():.4f} "
                       f"kd_box {l_box.item():.4f} ovf {float(l_ovf):.3g} {step * bs * world / (time.time() - t0):.2f} img/s")
                print(msg, flush=True)
                log.write(msg + "\n")
                log.flush()
            if rank == 0 and (step % args.eval_every == 0 or step == total):
                ints = student.export_ints()
                torch.save({k: (a.cpu(), b.cpu()) for k, (a, b) in ints.items()}, os.path.join(args.out, f"ints_step{step}.pt"))
                torch.save(student.state_dict(), os.path.join(args.out, "last_student.pt"))
                if apu_m:
                    torch.save(student.export_apu(), os.path.join(args.out, f"apu_step{step}.pt"))
            if step >= total:
                break
        epoch += 1
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
