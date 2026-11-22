"""Physical layout a HEF programs into the chip: per context, per cluster, which subclusters / L2 banks / APUs /
input aligners are switched on (enable_sequencer bitmaps), which LCUs run (enable_lcu), and the address ranges of
the plain register writes (write_data) and config-channel bursts (CCW).

Usage: python hwlayout.py X.hef [--raw]   (summary; --raw also prints every action)
       from hwlayout import layout; layout(path) -> dict
"""
import collections
import os
import sys

sys.path.insert(0, "/mnt/storage/lib_project/models/Hailo_model/hailo_re/hefre")
import hef_dump as D  # noqa: E402


def bits(x):
    return [i for i in range(64) if x >> i & 1]


def layout(path, raw=False):
    hef, ver, size, ccw_off = D.load(path)
    blob = open(path, "rb").read()[ccw_off:]
    out = []
    for g in hef.network_groups:
        for c in list(g.contexts) + [None]:
            ops = g.preliminary_config.operation if c is None else c.operations
            name = "preliminary" if c is None else f"ctx{c.context_index}"
            cl = collections.defaultdict(lambda: dict(sc=set(), l2=set(), apu=set(), ia=set(), lcu=set(), seq=0,
                                                       l3=set()))
            writes, ccw = collections.Counter(), collections.Counter()
            for op in ops:
                t = op.trigger.WhichOneof("trigger")
                for a in op.actions:
                    k = a.WhichOneof("action")
                    v = getattr(a, k)
                    if raw:
                        print(name, t, k, str(v).replace("\n", " ")[:200])
                    if k == "enable_sequencer":
                        d = cl[v.cluster_index]
                        d["sc"] |= set(bits(v.active_sc_bitmap))
                        d["l2"] |= set(bits(v.active_l2_bitmap))
                        d["apu"] |= set(bits(v.active_apu_bitmap))
                        d["ia"] |= set(bits(v.active_ia_bitmap))
                        d["seq"] += 1
                        if v.initial_l3_info.includes_initial_l3_info:
                            d["l3"].add((v.initial_l3_info.initial_l3_index, v.initial_l3_info.initial_l3_offset))
                    elif k == "enable_lcu":
                        cl[v.cluster_index]["lcu"].add(v.lcu_index)
                    elif k in ("write_data", "write_compressed_data"):
                        writes[v.address >> 16] += len(v.data)
                    elif k == "write_data_ccw":
                        ccw[v.cfg_channel_index] += len(v.data)
                    elif k == "write_data_ccw_ptr":
                        ccw[v.cfg_channel_index] += v.size
            out.append(dict(ctx=name, clusters={i: {k_: (sorted(s) if isinstance(s, set) else s) for k_, s in d.items()}
                                                for i, d in sorted(cl.items())},
                            writes={hex(k_ << 16): n for k_, n in sorted(writes.items())}, ccw=dict(ccw)))
    return out


def rng(s):
    return f"{len(s)}:{s[0]}-{s[-1]}" if s else "0"


if __name__ == "__main__":
    for c in layout(sys.argv[1], "--raw" in sys.argv):
        print(c["ctx"], "ccw bytes", c["ccw"], "writes", c["writes"])
        for i, d in c["clusters"].items():
            print(f"  cluster {i}: sc {rng(d['sc'])} l2 {rng(d['l2'])} apu {rng(d['apu'])} ia {rng(d['ia'])} "
                  f"lcu {d['lcu']} seq {d['seq']} l3 {d['l3']}")
