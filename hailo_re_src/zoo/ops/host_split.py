"""Host-side ops in the middle of a network: split, then run in STAGED mode on one chip.

A mid-network op with no on-chip form (TopK, Scatter, GridSample, RoiAlign, ...) splits the model in two HEFs.
Running them frame by frame makes HailoRT switch network groups every frame (+7.6 ms/frame, HOST_VS_CHIP.md).
STAGED mode instead runs a chunk of K frames through part1, keeps the boundary tensors in host RAM as raw
integers, runs the host op on the whole chunk (onnxruntime, float32, batched), and runs the chunk through part2:
one network switch per chunk. The host op of chunk k overlaps the chip running part1 of chunk k+1.

Accuracy: the boundary keeps the chip's own integers (no float round trip inside HailoRT); the host op runs in
float32 on exactly dequantized values (Q/DQ fused into the host graph); build part1 with a 16-bit output and part2 with a 16-bit input
(PREC=max16 / quantization_param a16_w16) and the boundary costs nothing measurable (HOST_VS_CHIP.md).

  plan  (general env):  python host_split.py plan IN.onnx   -> host-only regions, cut tensors, head/tail/mid mode
  wrap  (general env):  python host_split.py wrap OP.onnx host.onnx   (an op model already cut out, NCHW)
  split (general env):  python host_split.py split IN.onnx OUT_DIR --cut-in T1 --cut-out T2
        -> part1.onnx (inputs -> T1), host.onnx (T1 -> T2, NHWC in/out, dynamic batch), part2.onnx (T2 -> outputs)
  run   (hailo env):    StagedRunner(part1.hef, host.onnx, part2.hef, chunk=256).run(frames_nhwc)
                        StagedRunner(..., ram_mb=64).run_iter(camera_frames)   (edge: RAM budget, streaming)

Host device: device="cpu" (onnxruntime CPU), "cuda:N" (onnxruntime CUDA EP, needs onnxruntime-gpu on the path:
gpu_env.sh), or host=<torch callable> (NHWC float tensor -> NHWC float tensor, runs on `device`; Q/DQ in torch).
"""
import argparse
import queue
import threading

import numpy as np


def nhwc_wrap(src, dst):
    """Rank-4 inputs/outputs NCHW -> NHWC (the chip side's layout) with a dynamic batch dimension."""
    import onnx
    from onnx import helper
    h = onnx.load(src)  # the chip side is NHWC: wrap the host op in NCHW<->NHWC transposes
    for vi, is_in in [(v, True) for v in h.graph.input] + [(v, False) for v in h.graph.output]:
        dims = vi.type.tensor_type.shape.dim
        if len(dims) != 4:
            continue
        name = vi.name
        if is_in:
            t = f"{name}__nchw"
            for n in h.graph.node:
                n.input[:] = [t if i == name else i for i in n.input]
            h.graph.node.insert(0, helper.make_node("Transpose", [name], [t], perm=[0, 3, 1, 2]))
        else:
            t = f"{name}__nchw"
            for n in h.graph.node:
                n.output[:] = [t if o == name else o for o in n.output]
            h.graph.node.append(helper.make_node("Transpose", [t], [name], perm=[0, 2, 3, 1]))
        c, hh, w = (d.dim_value for d in dims[1:])
        for d, v in zip(dims, [0, hh, w, c]):
            d.Clear()
            if v:
                d.dim_value = v
            else:
                d.dim_param = "N"  # dynamic batch: one onnxruntime call per chunk
    onnx.save(h, dst)


HOST_ONLY = {"TopK", "ScatterElements", "ScatterND", "GridSample", "RoiAlign", "MaxRoiPool", "NonZero",
             "NonMaxSuppression", "GatherND", "Unique", "GRU", "Loop", "Scan", "If", "Det", "STFT"}


def plan(src, lstm_steps=16):
    """Host-only nodes after onnx_hailo_compat: where each one sits and how to run it.
    head/tail -> parse with start/end nodes, host op before/after the HEF (costs the op time only);
    mid -> split + StagedRunner (one network switch per chunk). Adjacent host nodes merge into one region."""
    import onnx
    m = onnx.load(src)
    g = m.graph
    consts = {i.name for i in g.initializer}
    for n in g.node:  # topological order: a node fed only by constants yields constants (a fixed sampling grid)
        if all(i in consts for i in n.input if i):
            consts.update(n.output)
    prod = {o: n for n in g.node for o in n.output}
    host = []
    for n in g.node:
        hit = n.op_type in HOST_ONLY or (n.op_type == "Resize" and any(
            a.name == "mode" and a.s == b"cubic" for a in n.attribute))
        if n.op_type == "LSTM":
            sh = {v.name: v for v in list(g.value_info) + list(g.input)}.get(n.input[0])
            T = sh.type.tensor_type.shape.dim[0].dim_value if sh else 0
            hit = T > lstm_steps or T == 0
        if hit:
            host.append(n)
    hnames = {n.name for n in host}
    regions = []  # union adjacent host nodes
    for n in host:
        ups = {prod[i].name for i in n.input if i in prod and prod[i].name in hnames}
        r = next((r for r in regions if ups & {x.name for x in r}), None)
        (r.append(n) if r is not None else regions.append([n]))
    ins = {i.name for i in g.input} - consts
    outs = {o.name for o in g.output}
    res = []
    for r in regions:
        made = {o for n in r for o in n.output}
        cut_in = sorted({i for n in r for i in n.input if i and i not in made and i not in consts})
        used_after = {i for n in g.node if n not in r for i in n.input}
        cut_out = sorted(o for o in made if o in used_after or o in outs)

        head = all(t in ins for t in cut_in)
        tail = all(o in outs for o in cut_out)
        mode = "head: host op before the HEF" if head else "tail: host op after the HEF" if tail else \
            "mid: split + StagedRunner (chunk 64-256)"
        res.append({"ops": [n.op_type for n in r], "nodes": [n.name for n in r], "cut_in": cut_in,
                    "cut_out": cut_out, "mode": mode})
    return res


def split(src, out, cut_in, cut_out):
    import os

    import onnx
    from onnx import utils
    os.makedirs(out, exist_ok=True)
    m = onnx.load(src)
    ins = [i.name for i in m.graph.input if i.name not in {x.name for x in m.graph.initializer}]
    outs = [o.name for o in m.graph.output]
    utils.extract_model(src, f"{out}/part1.onnx", ins, cut_in)
    utils.extract_model(src, f"{out}/part2.onnx", cut_out, outs)
    utils.extract_model(src, f"{out}/host.onnx", cut_in, cut_out)
    nhwc_wrap(f"{out}/host.onnx", f"{out}/host.onnx")
    print(f"split {src}: part1 -> {cut_in} | host | {cut_out} -> part2 in {out}")


class _RefSession:
    """onnxruntime-like wrapper of onnx.reference.ReferenceEvaluator: every default-domain op, slow, CPU."""

    def __init__(self, m):
        from onnx.reference import ReferenceEvaluator

        class _In:
            def __init__(self, v):
                self.name = v.name
                self.shape = [d.dim_value if d.HasField("dim_value") else d.dim_param for d in
                              v.type.tensor_type.shape.dim]
        self.ev = ReferenceEvaluator(m)
        self.ins = [_In(v) for v in m.graph.input]

    def get_inputs(self):
        return self.ins

    def run(self, outs, feeds):
        return self.ev.run(outs, feeds)


class StagedRunner:
    """part1.hef -> host.onnx -> part2.hef on ONE chip, a chunk at a time, boundaries as raw chip integers."""

    def __init__(self, hef1, host, hef2, chunk=None, device_id=None, batch=8, device="cpu", ram_mb=None,
                 queue_depth=2):
        """chunk: frames per chunk (default 256), or ram_mb: a host-RAM budget for everything the pipeline holds
        (boundary caches in flight, host-op temporaries, the input and output chunk) -> chunk derived from the
        per-frame bytes of the two HEFs (self.mem_plan). queue_depth: chunks queued between chip and host (>= 1)."""
        from hailo_platform import (HEF, ConfigureParams, Device, FormatType, HailoStreamInterface,
                                    InputVStreamParams, OutputVStreamParams, VDevice)
        p = VDevice.create_params()
        p.device_ids = [device_id or str(Device.scan()[0])]
        self.vd = VDevice(p)
        raw = {8: FormatType.UINT8, 16: FormatType.UINT16}

        def setup(path, in_raw, out_raw):
            hef = HEF(path)
            cp = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
            for k in cp:
                cp[k].batch_size = batch
            ng = self.vd.configure(hef, cp)[0]
            ii, oi = hef.get_input_vstream_infos()[0], hef.get_output_vstream_infos()[0]
            bits = lambda vi: 16 if "16" in str(vi.format.type) else 8  # noqa: E731
            ip = InputVStreamParams.make(ng, format_type=raw[bits(ii)] if in_raw else FormatType.FLOAT32)
            op = OutputVStreamParams.make(ng, format_type=raw[bits(oi)] if out_raw else FormatType.FLOAT32)
            return dict(ng=ng, ip=ip, op=op, iname=ii.name, iq=ii.quant_info, oq=oi.quant_info, ibits=bits(ii),
                        odt=bits(oi), idt=np.uint16 if bits(ii) == 16 else np.uint8,
                        in_elems=int(np.prod(ii.shape)), out_elems=int(np.prod(oi.shape)))
        self.p1 = setup(hef1, False, True)
        self.p2 = setup(hef2, True, False)
        self.device = device
        self.p1_dt = np.uint16 if self.p1["odt"] == 16 else np.uint8
        if callable(host):  # torch function on `device`
            self.fn, self.batched = host, True
        else:
            import onnxruntime as ort
            self.host, self.hin = self._fused_host(host, ort)
            shp = [d if isinstance(d, int) else 2 for d in self.host.get_inputs()[0].shape]
            try:  # exported graphs often bake batch 1 into constants (zeros_like, Reshape): then one call per frame
                self.host.run(None, {self.hin: np.zeros(shp, self.p1_dt)})
                self.batched = True
            except Exception:  # noqa: BLE001
                self.batched = False
        self.depth = max(1, int(queue_depth))
        self.mem_plan = self._plan(chunk, ram_mb)
        self.K = self.mem_plan["chunk"]
        from concurrent.futures import ThreadPoolExecutor
        self.pool = ThreadPoolExecutor(8)  # per-frame host ops: frames in parallel (onnxruntime releases the GIL)

    def _plan(self, chunk, ram_mb):
        """Host bytes per frame of everything in flight:
          boundary caches: (depth + 1) part1 outputs (raw ints) + (depth + 1) host outputs (raw ints)
          host temporaries: dequantized input + float op output (CPU host only; on CUDA they live on the GPU)
          the input chunk (float32, being fed to part1) and the output chunk (float32, from part2)"""
        e1, b1 = self.p1["out_elems"], self.p1["odt"] // 8
        e2, b2 = self.p2["in_elems"], self.p2["ibits"] // 8
        cache = (self.depth + 1) * (e1 * b1 + e2 * b2)
        temps = 0 if self.device.startswith("cuda") else (e1 + e2) * 4
        io = (self.p1["in_elems"] + self.p2["out_elems"]) * 4
        per = cache + temps + io
        if ram_mb is not None:
            k = int(ram_mb * 2 ** 20 // per)
            if k < 1:
                raise ValueError(f"ram_mb={ram_mb}: one frame needs {per / 2 ** 20:.2f} MB (lower queue_depth?)")
            chunk = min(chunk or k, k)
        chunk = chunk or 256
        return {"chunk": chunk, "bytes_per_frame": per, "cache_bytes_per_frame": cache, "temp_bytes_per_frame": temps,
                "io_bytes_per_frame": io, "planned_mb": round(chunk * per / 2 ** 20, 1), "queue_depth": self.depth}

    def _fused_host(self, host_onnx, ort):
        """host.onnx with the boundary (de)quantization inside: DequantizeLinear(chip int) -> op -> QuantizeLinear
        (round half to even, saturate). onnxruntime runs it multithreaded; numpy did it on one core (1.3 ms/frame).
        16-bit boundaries use the com.microsoft Q/DQ (uint16) so the op's own opset stays untouched."""
        import onnx
        from onnx import TensorProto, helper, numpy_helper
        m = onnx.load(host_onnx)
        g = m.graph
        oq, iq = self.p1["oq"], self.p2["iq"]
        dt = {8: TensorProto.UINT8, 16: TensorProto.UINT16}
        x, y = g.input[0], g.output[0]
        xin, yout = f"{x.name}__q", f"{y.name}__q"
        ms = any(b == 16 for b in (self.p1["odt"], self.p2["ibits"]))
        dom = "com.microsoft" if ms else ""
        g.initializer.extend([
            numpy_helper.from_array(np.array(oq.qp_scale, np.float32), "__s1"),
            numpy_helper.from_array(np.array(oq.qp_zp, self.p1_dt), "__z1"),
            numpy_helper.from_array(np.array(iq.qp_scale, np.float32), "__s2"),
            numpy_helper.from_array(np.array(iq.qp_zp, self.p2["idt"]), "__z2")])
        g.node.insert(0, helper.make_node("DequantizeLinear", [xin, "__s1", "__z1"], [x.name], domain=dom))
        g.node.append(helper.make_node("QuantizeLinear", [y.name, "__s2", "__z2"], [yout], domain=dom))
        xi = helper.make_tensor_value_info(xin, dt[self.p1["odt"]], None)
        xi.type.tensor_type.shape.CopyFrom(x.type.tensor_type.shape)
        yo = helper.make_tensor_value_info(yout, dt[self.p2["ibits"]], None)
        yo.type.tensor_type.shape.CopyFrom(y.type.tensor_type.shape)
        del g.input[:]
        g.input.append(xi)
        del g.output[:]
        g.output.append(yo)
        if ms:
            m.opset_import.append(helper.make_opsetid("com.microsoft", 1))
        so = ort.SessionOptions()
        so.intra_op_num_threads = 8
        prov = ["CPUExecutionProvider"]
        if self.device.startswith("cuda"):
            assert "CUDAExecutionProvider" in ort.get_available_providers(), \
                f"{ort.__file__}: no CUDA EP (source gpu_env.sh: onnxruntime-gpu + CUDA 12 libs)"
            prov = [("CUDAExecutionProvider", {"device_id": int((self.device + ":0").split(":")[1])})] + prov
        try:
            sess = ort.InferenceSession(m.SerializeToString(), so, providers=prov)
        except Exception:  # noqa: BLE001  an op onnxruntime has no kernel for: onnx's reference evaluator (CPU)
            if self.device.startswith("cuda"):
                raise
            return _RefSession(m), xin
        if self.device.startswith("cuda"):
            assert sess.get_providers()[0] == "CUDAExecutionProvider", sess.get_providers()
        return sess, xin

    @staticmethod
    def _infer(p, x):
        from hailo_platform import InferVStreams
        with p["ng"].activate(p["ng"].create_params()), InferVStreams(p["ng"], p["ip"], p["op"]) as pipe:
            return list(pipe.infer({p["iname"]: x}).values())[0]

    def _host_torch(self, q):
        """Raw chip integers go to the GPU as-is (1-2 bytes/value), Q/DQ on the GPU, integers come back."""
        import torch
        oq, iq = self.p1["oq"], self.p2["iq"]
        with torch.no_grad():
            t = torch.from_numpy(q.view(np.int16) if q.dtype == np.uint16 else q).to(self.device)
            if q.dtype == np.uint16:
                t = t.to(torch.int32) & 0xFFFF  # torch has no uint16 arithmetic
            f = (t.float() - oq.qp_zp) * oq.qp_scale  # exact dequantization
            r = torch.round(self.fn(f) / iq.qp_scale + iq.qp_zp)  # round half to even, as QuantizeLinear
            if self.p2["idt"] == np.uint16:
                return r.clamp_(0, 65535).to(torch.int32).to(torch.int16).cpu().numpy().view(np.uint16)
            return r.clamp_(0, 255).to(torch.uint8).cpu().numpy()

    def _host(self, q):
        if callable(getattr(self, "fn", None)):
            return self._host_torch(q)
        if self.batched:
            return self.host.run(None, {self.hin: q})[0]
        return np.concatenate(list(self.pool.map(lambda i: self.host.run(None, {self.hin: q[i:i + 1]})[0],
                                                 range(len(q)))))

    def run(self, frames):
        """frames: (N, H, W, C) float32 -> part2 outputs (N, ...) float32."""
        return np.concatenate(list(self.run_iter(frames)))

    def run_iter(self, frames):
        """Streaming: frames = any iterable of (H, W, C) frames or (n, H, W, C) arrays (camera, video reader).
        Yields part2 outputs chunk by chunk; host RAM stays within mem_plan. Chip: P1(k+1) while host: op(k)."""
        def chunks():
            buf, have = [], 0
            for f in frames if not isinstance(frames, np.ndarray) else [frames]:
                f = np.asarray(f, np.float32)
                f = f[None] if f.ndim == 3 else f
                for i in range(0, len(f), self.K):
                    part = f[i:i + self.K]
                    buf.append(part)
                    have += len(part)
                    while have >= self.K:
                        cat = np.concatenate(buf)
                        yield cat[:self.K]
                        rest = cat[self.K:]
                        buf, have = ([rest] if len(rest) else []), len(rest)
            if have:
                yield np.concatenate(buf)

        todo, done = queue.Queue(maxsize=self.depth), queue.Queue(maxsize=self.depth)

        def host_worker():
            while (q := todo.get()) is not None:
                done.put(self._host(q))
            done.put(None)
        th = threading.Thread(target=host_worker)
        th.start()
        try:
            it = chunks()
            first = next(it, None)
            if first is not None:
                todo.put(self._infer(self.p1, first))
                for c in it:
                    todo.put(self._infer(self.p1, c))  # chip busy on chunk k+1 while the host works on chunk k
                    yield self._infer(self.p2, done.get())
            todo.put(None)
            while (g := done.get()) is not None:
                yield self._infer(self.p2, g)
        finally:
            th.join()

    def close(self):
        self.pool.shutdown()
        self.vd.release()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "split", "wrap"])
    ap.add_argument("src")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--cut-in", nargs="+")
    ap.add_argument("--cut-out", nargs="+")
    a = ap.parse_args()
    if a.cmd == "plan":
        import json
        for r in plan(a.src):
            print(json.dumps(r))
    elif a.cmd == "wrap":  # an existing NCHW op model -> host.onnx
        nhwc_wrap(a.src, a.out)
    else:
        split(a.src, a.out, a.cut_in, a.cut_out)
