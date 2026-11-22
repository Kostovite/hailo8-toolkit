"""Dump the compiler's integrated HW graph (HEF `mapping` field) to JSON: nodes, edges, per-node resources, contexts.

Usage (DFC env): python hwgraph_json.py MODEL.hef OUT.json
"""
import json
import sys

from google.protobuf.json_format import MessageToDict

sys.path.insert(0, "/mnt/storage/lib_project/models/Hailo_model/hailo_re/hefre")
import hef_dump as D  # noqa: E402
from hailo_sdk_client.allocator import integrated_hw_graph_pb2 as IG  # noqa: E402

hef = D.load(sys.argv[1])[0]
g = IG.ProtoIntegratedHWGraph()
g.ParseFromString(hef.mapping)
nodes = [{"name": n.name, "type": n.type, "out": [[s.height, s.width, s.features] for s in n.output_shapes],
          "inp": [[s.height, s.width, s.features] for s in n.input_shapes], "succ": list(n.successors_indices),
          "kernel": [n.kernel_shape.height, n.kernel_shape.width, n.kernel_shape.features] if n.HasField("kernel_shape") else None}
         for n in g.network_graph.nodes]
res = [MessageToDict(r, preserving_proto_field_name=True) for r in g.network_graph.resources_list]
out = {"nodes": nodes, "resources": res, "edges": [[e.first_index, e.second_index] for e in g.network_graph.edges],
       "contexts_graph": MessageToDict(g.contexts_graph, preserving_proto_field_name=True),
       "bottleneck_fps": float(g.bottleneck_fps)}
json.dump(out, open(sys.argv[2], "w"))
print("nodes", len(nodes), "resource fields", sorted(res[1].keys()) if len(res) > 1 else None)
