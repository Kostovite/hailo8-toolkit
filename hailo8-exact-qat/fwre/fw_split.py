"""Split a Hailo-8 firmware file (hailo8_fw.X.bin from HailoRT) into its APP CPU image, secure-boot certificate and
CORE CPU image (layout from HailoRT common/include/firmware_header.h).

  [firmware_header_t app][app code][secure_boot_certificate_t][firmware_header_t core][core code]
  firmware_header_t = {magic 0x1DD89DE0, header_version, major, minor, revision, code_size}, little-endian u32
  secure_boot_certificate_t = {key_size, content_size, key_size + content_size bytes}

Both images are ARM Cortex-M Thumb: APP loads at 0x60000, CORE at 0xC0000 (load them in Ghidra as ARM:LE:32:Cortex;
ghidra_scripts/CortexMDump.java seeds functions from the vector table and decompiles everything).

Usage: python fw_split.py hailo8_fw.4.23.0.bin OUT_DIR
"""
import os
import struct
import sys

MAGIC = 0x1DD89DE0


def header(raw, off, what):
    magic, ver, major, minor, rev, size = struct.unpack_from("<6I", raw, off)
    assert magic == MAGIC, f"{what}: bad magic {magic:#x} at {off:#x}"
    print(f"{what}: firmware {major}.{minor}.{rev & 0xFFFFFF} (flags {rev >> 24:#x}) header v{ver}, code {size} B at {off + 24:#x}")
    return off + 24, size


src, out = sys.argv[1:3]
os.makedirs(out, exist_ok=True)
raw = open(src, "rb").read()
code, size = header(raw, 0, "APP CPU")
open(os.path.join(out, "app_cpu.bin"), "wb").write(raw[code:code + size])
cert = code + size
key, content = struct.unpack_from("<2I", raw, cert)
end = cert + 8 + key + content
open(os.path.join(out, "app_cert.bin"), "wb").write(raw[cert:end])
print(f"certificate: key {key} B, content {content} B")
code, size = header(raw, end, "CORE CPU")
open(os.path.join(out, "core_cpu.bin"), "wb").write(raw[code:code + size])
assert code + size == len(raw), f"{len(raw) - code - size} trailing bytes"
