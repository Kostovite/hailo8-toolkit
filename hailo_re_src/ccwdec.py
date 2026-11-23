"""ccwdec.py: decode the config-channel writes (CCWs) of a HEF into (context, cfg_channel, address, data) records.

CCW format (HailoRT 4.23 hef.cpp parse_ccw_buffer, config_buffer.hpp):
  u32 words_count_minus_1, u32 address, then (words_count_minus_1 + 1) * 4 data bytes, padded to 8 bytes.
A write_data_ccw_ptr action points at a region of the CCW blob after the protobuf; that region is a back-to-back
stream of CCWs (the cfg DMA channel feeds it to the core). write_data_ccw carries the same stream inline.

Usage: python ccwdec.py FILE.hef [--sum] [--dump N]
  default: one line per CCW (ctx, ch, address, nbytes, first bytes); --sum: bytes per address block
As a module: records(path) -> list of dict(ctx, ch, addr, data)
"""
import collections
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hef_dump import load  # noqa: E402


def stream(buf):
    """Yield (address, data, data_offset_in_buf) for a back-to-back CCW stream; raise on a malformed stream."""
    i, n = 0, len(buf)
    while i + 8 <= n:
        wc, addr = struct.unpack_from("<II", buf, i)
        ln = (wc + 1) * 4
        if i + 8 + ln > n:
            raise ValueError(f"CCW at {i} wants {ln} bytes, {n - i - 8} left")
        yield addr, buf[i + 8:i + 8 + ln], i + 8
        i += 8 + ln
        i += (-i) % 8  # pad to 8
    if i < n and any(buf[i:]):
        raise ValueError(f"{n - i} trailing non-zero bytes")


def records(path):
    hef, _, _, blob_off = load(path)
    raw = open(path, "rb").read()
    out = []
    for g in hef.network_groups:
        ctxs = [("pre", g.preliminary_config.operation)] + [(c.context_index, c.operations) for c in g.contexts]
        for ci, ops in ctxs:
            for op in ops:
                for a in op.actions:
                    k = a.WhichOneof("action")
                    if k == "write_data_ccw_ptr":
                        p = a.write_data_ccw_ptr
                        buf, ch, base = raw[blob_off + p.offset: blob_off + p.offset + p.size], p.cfg_channel_index, blob_off + p.offset
                    elif k == "write_data_ccw":
                        buf, ch, base = a.write_data_ccw.data, getattr(a.write_data_ccw, "cfg_channel_index", 0), None
                    elif k == "write_data":
                        out.append(dict(ctx=ci, ch=-1, addr=a.write_data.address, data=bytes(a.write_data.data)))
                        continue
                    else:
                        continue
                    for addr, d, o in stream(buf):
                        out.append(dict(ctx=ci, ch=ch, addr=addr, data=bytes(d), foff=None if base is None else base + o))
    return out


def main():
    path = sys.argv[1]
    R = records(path)
    if "--sum" in sys.argv:
        blk = collections.defaultdict(lambda: [0, 0])
        for r in R:
            b = blk[(r["ctx"], r["addr"] >> 16)]
            b[0] += 1; b[1] += len(r["data"])
        print(f"{len(R)} CCWs, {sum(len(r['data']) for r in R)} bytes")
        for (c, hi), (n, nb) in sorted(blk.items(), key=lambda x: (str(x[0][0]), x[0][1])):
            print(f"ctx {c!s:>3} addr 0x{hi:04x}xxxx  {n:6d} writes {nb:9d} B")
        return
    lim = int(sys.argv[sys.argv.index("--dump") + 1]) if "--dump" in sys.argv else 10**9
    for r in R[:lim]:
        print(f"ctx {r['ctx']!s:>3} ch {r['ch']:2d} 0x{r['addr']:08x} {len(r['data']):7d} B  {r['data'][:16].hex()}")


if __name__ == "__main__":
    main()
