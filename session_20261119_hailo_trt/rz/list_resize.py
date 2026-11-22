import onnx,sys
from onnx import shape_inference
m=shape_inference.infer_shapes(onnx.load(sys.argv[1]))
vi={v.name:[d.dim_value for d in v.type.tensor_type.shape.dim] for v in list(m.graph.value_info)+list(m.graph.input)+list(m.graph.output)}
for n in m.graph.node:
    if n.op_type=="Resize":
        a={x.name:onnx.helper.get_attribute_value(x) for x in n.attribute}
        print(n.name, list(n.input), vi.get(n.input[0]), "->", vi.get(n.output[0]), a)
print("opset", [o.version for o in m.opset_import])
