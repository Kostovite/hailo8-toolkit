"""Use FP32 QAT checkpoint weights on the unsupported mesh-output head; preserve Q/DQ elsewhere."""
import os,onnx
D=os.path.expanduser('~/codex_trt_qat_20261119');m=onnx.load(D+'/artifacts/trt/flat3d_qat500.onnx');g=m.graph
producers={o:n for n in g.node for o in n.output};aliases={}
for n in g.node:
 if n.op_type!='DequantizeLinear' or 'out_point_positions2D' not in n.name:continue
 q=producers.get(n.input[0]);
 if q and q.op_type=='QuantizeLinear':aliases[n.output[0]]=q.input[0];print('FP32 head',n.name,flush=True)
assert aliases,'no output-head QDQ nodes found'
for n in g.node:
 for i,x in enumerate(n.input):
  if x in aliases:n.input[i]=aliases[x]
onnx.checker.check_model(m);p=D+'/artifacts/trt/flat3d_qat500_headfp32.onnx';onnx.save(m,p);print('saved',p,flush=True)
