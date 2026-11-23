"""hefpatch.py IN.hef OUT.hef ADDR:OFFSET:HEXBYTES [...]: overwrite bytes inside the CCW whose target address is ADDR
(offset relative to that CCW's data), then fix the header hash (v2/v3 XXH3-64 over everything after the header).
Only CCWs stored in the blob (write_data_ccw_ptr) can be patched in place (same length)."""
import os
import struct
import sys

import xxhash

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ccwdec import records  # noqa: E402

HDR = {2: 12 + 32, 3: 12 + 44}


def rehash(raw):
    ver = struct.unpack(">I", raw[4:8])[0]
    h = HDR[ver]
    v = xxhash.xxh3_64_intdigest(bytes(raw[h:]))
    raw[12:20] = struct.pack(">Q", v)
    return raw


def main():
    src, dst, *edits = sys.argv[1:]
    raw = bytearray(open(src, "rb").read())
    ver = struct.unpack(">I", raw[4:8])[0]
    assert struct.unpack(">Q", raw[12:20])[0] == xxhash.xxh3_64_intdigest(bytes(raw[HDR[ver]:])), "hash layout check failed"
    R = records(src)
    for e in edits:
        a, o, hx = e.split(":")
        a, o, b = int(a, 16), int(o), bytes.fromhex(hx)
        r = [x for x in R if x["addr"] == a]
        assert len(r) == 1 and r[0]["foff"] is not None, f"need exactly one blob CCW at 0x{a:08x}, got {len(r)}"
        f = r[0]["foff"] + o
        print(f"0x{a:08x}+{o}: {raw[f:f + len(b)].hex()} -> {b.hex()}")
        raw[f:f + len(b)] = b
    open(dst, "wb").write(rehash(raw))


if __name__ == "__main__":
    main()
