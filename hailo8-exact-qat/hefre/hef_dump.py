"""Dump what a HEF does at every context switch: config bytes DMA'd per context, action mix, and the tensors that
leave / enter the chip (boundary I/O, inter-context activations, DDR buffers) with their per-frame bytes.

HEF layout (HailoRT 4.23 hef_internal.hpp): big-endian header {magic 0x01484546, version, proto_size, distinct},
then the ProtoHEFHef protobuf, then (v1+) the CCW blob that write_data_ccw_ptr actions point into.

Usage: python hef_dump.py FILE.hef [--json OUT.json]
"""
import collections
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hef_pb2 as H  # noqa: E402

DISTINCT = {0: 4 + 16, 1: 4 + 8 + 4, 2: 8 + 8 + 8 + 8, 3: 8 + 8 + 4 + 8 + 8 + 8}
CONN = {0: "boundary", 1: "inter_context", 2: "ddr", 3: "cache"}


def load(path):
    raw = open(path, "rb").read()
    magic, ver, psize = struct.unpack(">III", raw[:12])
    assert magic == 0x01484546, hex(magic)
    off = 12 + DISTINCT[ver]
    pad = struct.unpack(">I", raw[12 + 16:12 + 20])[0] if ver == 3 else 0
    hef = H.ProtoHEFHef()
    hef.ParseFromString(raw[off:off + psize])
    return hef, ver, len(raw), off + psize + pad


def action_bytes(a):
    k = a.WhichOneof("action")
    if k == "write_data":
        return len(a.write_data.data)
    if k == "write_data_ccw":
        return len(a.write_data_ccw.data)
    if k == "write_data_ccw_ptr":
        return a.write_data_ccw_ptr.size
    if k == "write_compressed_data":
        return len(a.write_compressed_data.data)
    return 0


def ops_summary(ops):
    kinds, nbytes = collections.Counter(), 0
    for op in ops:
        for a in op.actions:
            kinds[a.WhichOneof("action")] += 1
            nbytes += action_bytes(a)
    return kinds, nbytes


def edge_rows(ctx):
    rows = []
    for e in ctx.metadata.edge_layers:
        k = e.WhichOneof("edge")
        info = getattr(e, k)
        base = info.edge_layer_base
        cs = e.context_switch_info
        rows.append(dict(name=info.name, dir="in" if e.direction == 0 else "out", conn=CONN.get(cs.edge_connection_type),
                         shape=[base.height, base.width, base.features],
                         frame_bytes=base.core_bytes_per_buffer * base.core_buffers_per_frame,
                         buffers=cs.buffers, peer=cs.connected_context_index, engine=base.engine_id))
    return rows


def main():
    path = sys.argv[1]
    hef, ver, fsize, ccw_off = load(path)
    out = {"file": path, "version": ver, "size": fsize, "ccw_offset": ccw_off, "groups": []}
    for g in hef.network_groups:
        pk, pb = ops_summary(g.preliminary_config.operation)
        grp = {"name": g.network_group_name, "preliminary_bytes": pb, "preliminary_actions": dict(pk), "contexts": []}
        print(f"{path}: HEF v{ver}, {fsize / 2**20:.1f} MiB, group {g.network_group_name}, {len(g.contexts)} contexts")
        print(f"  preliminary config: {pb / 2**20:.2f} MiB  {dict(pk)}")
        for c in g.contexts:
            kinds, nb = ops_summary(c.operations)
            rows = edge_rows(c)
            io = collections.defaultdict(int)
            for r in rows:
                io[(r["conn"], r["dir"])] += r["frame_bytes"]
            grp["contexts"].append({"index": c.context_index, "name": c.metadata.name, "config_bytes": nb,
                                    "actions": dict(kinds), "edges": rows,
                                    "io_bytes": {f"{a}/{b}": v for (a, b), v in io.items()}})
            print(f"  ctx {c.context_index} {c.metadata.name}: config {nb / 2**20:.2f} MiB, "
                  f"{sum(kinds.values())} actions, batch-switch {kinds.get('switch_lcu_batch', 0)}")
            for (a, b), v in sorted(io.items()):
                n = sum(1 for r in rows if r["conn"] == a and r["dir"] == b)
                print(f"      {a:13s} {b:3s} {n:3d} tensors  {v / 2**20:7.2f} MiB/frame")
        out["groups"].append(grp)
    if "--json" in sys.argv:
        json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
