"""Load the compiler's integrated HW graph (HEF `mapping`) of a HEF. python ighw.py X.hef -> summary."""
import sys
sys.path.insert(0, "/mnt/storage/lib_project/models/Hailo_model/hailo_re/hefre")
import hef_dump as D  # noqa: E402
from hailo_sdk_client.allocator import integrated_hw_graph_pb2 as IG  # noqa: E402


def load(path):
    g = IG.ProtoIntegratedHWGraph()
    g.ParseFromString(D.load(path)[0].mapping)
    return g


if __name__ == "__main__":
    g = load(sys.argv[1])
    print("bottleneck_fps", g.bottleneck_fps, "se_formula", len(g.se_formula), "profiler ctx", len(g.profiler_data.context),
          "nodes", len(g.network_graph.nodes), "ctx nodes", len(g.contexts_graph.nodes), "script cmds", len(g.allocator_script.commands))
    for c in g.contexts_graph.nodes:
        ng = c.network_graph
        print(c.context.context_name, "nodes", len(ng.nodes), "res", len(ng.resources_list))
    if g.se_formula:
        print(g.se_formula[0])
