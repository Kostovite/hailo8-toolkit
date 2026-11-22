import os,onnx
from onnx import helper
D=os.path.expanduser('~/codex_trt_qat_20261119');p=D+'/artifacts/trt/flat3d_qat500.onnx';out=D+'/artifacts/trt/flat3d_qat500_prelu.onnx';m=onnx.load(p);nodes=[];count=0
for n in m.graph.node:
 if n.op_type!='PRelu':nodes.append(n);continue
 count+=1;tag='codex_prelu_%d'%count;x,a=n.input;positive=tag+'_pos';neg=tag+'_neg';negative=tag+'_relu_neg';scaled=tag+'_scaled'
 nodes.extend([helper.make_node('Relu',[x],[positive],name=tag+'_pos_relu'),helper.make_node('Neg',[x],[neg],name=tag+'_neg'),helper.make_node('Relu',[neg],[negative],name=tag+'_neg_relu'),helper.make_node('Mul',[negative,a],[scaled],name=tag+'_mul'),helper.make_node('Sub',[positive,scaled],list(n.output),name=tag+'_sub')])
del m.graph.node[:];m.graph.node.extend(nodes);onnx.checker.check_model(m);onnx.save(m,out);print('rewrote PRelu nodes',count,'output',out,flush=True)
