"""Decode a Hailo-8 HEF (header v2) with HailoRT's public hef.proto and summarise the NPU schedule.

Usage: python hef_dump.py model.hef
"""
import collections
import os
import struct
import sys

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "proto"))
import hef_pb2  # noqa: E402

raw = open(sys.argv[1], "rb").read()
magic, version, proto_size = struct.unpack(">III", raw[:12])
assert magic == 0x01484546, hex(magic)
assert version == 2, f"header v{version} not handled"
xxh, ccws_size = struct.unpack(">QQ", raw[12:28])
hdr = 12 + 32
hef = hef_pb2.ProtoHEFHef()
hef.ParseFromString(raw[hdr:hdr + proto_size])
print(f"header v{version} proto={proto_size} ccws={ccws_size} total={hdr + proto_size + ccws_size} file={len(raw)}")
print("hw_arch:", hef_pb2.ProtoHEFHwArch.Name(hef.header.hw_arch))

groups = []
for ng in hef.network_groups:
    groups.append(ng)
    groups.extend(p.network_group for p in ng.partial_network_groups)
for ng in groups:
    if not ng.contexts:
        continue
    print("network group:", ng.network_group_metadata.network_group_name, "contexts:", len(ng.contexts),
          "partial layouts:", len(hef.network_groups[0].partial_network_groups))
    for ci, ctx in enumerate(ng.contexts):
        actions = collections.Counter()
        triggers = collections.Counter()
        lcus = set()
        for op in ctx.operations:
            triggers[op.trigger.WhichOneof("trigger")] += 1
            for a in op.actions:
                kind = a.WhichOneof("action")
                actions[kind] += 1
                if kind in ("enable_lcu", "disable_lcu"):
                    sub = getattr(a, kind)
                    lcus.add((sub.cluster_index, sub.lcu_index))
        edges = collections.Counter(
            hef_pb2.ProtoHEFEdgeConnectionType.Name(e.context_switch_info.edge_connection_type)
            if e.HasField("context_switch_info") else "boundary"
            for e in ctx.metadata.edge_layers)
        clusters = sorted({c for c, _ in lcus})
        print(f"  ctx {ci}: ops={len(ctx.operations)} clusters={clusters} lcus={len(lcus)}")
        print(f"    actions={dict(actions)}")
        print(f"    triggers={dict(triggers)} edges={dict(edges)}")
