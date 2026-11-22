"""Per-context cost view of a HEF from the compiler's own HW graph: bottleneck node (frame_total_cycles), resource
totals, latency profile. python ctxcost.py A.hef [B.hef ...]"""
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from ighw import load  # noqa: E402

KEYS = ("subclusters", "apus", "input_aligners", "weights_blocks", "output_blocks", "l4_cuts", "l2_usage",
        "l3_weights_usage", "l3_output_usage")


def summary(path, top=4):
    g = load(path)
    print(f"== {path}  bottleneck_fps {g.bottleneck_fps:.1f}  clk {g.performance.clk_freq}")
    for c, prof in zip(g.contexts_graph.nodes, g.profiler_data.context):
        ng = c.network_graph
        names = {n.index: n.name for n in ng.nodes}
        rs = [r for r in ng.resources_list]
        tot = {k: round(sum(getattr(r, k) for r in rs), 2) for k in KEYS}
        cyc = sorted(((r.frame_total_cycles, names.get(r.layer_index, r.layer_index), r.clk_freq) for r in rs), reverse=True)
        lat = max((t.end_time for L in prof.layers for t in L.time), default=0)
        bw = prof.bandwidth
        print(f" {c.context.context_name}: nodes {len(ng.nodes)} latency_end {lat} bw in {bw.input_boundary:.0f}/{bw.input_inter_context:.0f}/{bw.input_ddr:.0f} "
              f"out {bw.output_boundary:.0f}/{bw.output_inter_context:.0f}/{bw.output_ddr:.0f}")
        print("   totals", tot)
        print("   top cycles", [(n, cy) for cy, n, _ in cyc[:top]], "clk", cyc[0][2] if cyc else None)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        summary(p)
