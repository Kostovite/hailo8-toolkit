import os,sys,numpy as np,torch,tensorrt as trt
D=os.path.expanduser('~/codex_trt_qat_20261119');A=D+'/aux_models';sys.path[:0]=[A,A+'/wordlid'];from region_router import RegionRouter
logger=trt.Logger(trt.Logger.WARNING);b=trt.Builder(logger);n=b.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH));p=trt.OnnxParser(n,logger)
if not p.parse_from_file(A+'/router_dynamic.onnx'):
 for i in range(p.num_errors):print(p.get_error(i),flush=True)
 raise RuntimeError('parse failed')
c=b.create_builder_config();c.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,1<<30);c.clear_flag(trt.BuilderFlag.TF32)
profile=b.create_optimization_profile();assert profile.set_shape('pooled_encoder',(1,1024),(32,1024),(256,1024));c.add_optimization_profile(profile)
e=b.build_serialized_network(n,c);assert e is not None
path=A+'/router/router_fp32.engine';open(path,'wb').write(e);print('built FP32 TensorRT',path,flush=True)
r=trt.Runtime(logger);engine=r.deserialize_cuda_engine(e);ctx=engine.create_execution_context();ck=torch.load(A+'/wordlid/data/grid/region_router.pt',map_location='cpu',weights_only=False);net=RegionRouter();net.load_state_dict(ck['router']);net.cuda().eval();x=torch.from_numpy(np.load(A+'/router/val.npy')).cuda().contiguous();assert ctx.set_input_shape('pooled_encoder',tuple(x.shape));ctx.set_tensor_address('pooled_encoder',x.data_ptr());ys=[]
for name in ('kind','body','lang'):
 y=torch.empty(tuple(ctx.get_tensor_shape(name)),dtype=torch.float32,device='cuda');ctx.set_tensor_address(name,y.data_ptr());ys.append(y)
assert ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream);torch.cuda.synchronize()
with torch.no_grad():
 ref=net(x);refs=[ref['kind'],ref['body'],ref['lang']];agree=[float((a.argmax(1)==z.argmax(1)).float().mean()) for a,z in zip(refs,ys)];err=[float((a-z).abs().max()) for a,z in zip(refs,ys)]
print('RESULT n',len(x),'argmax agreement',agree,'max_abs',err,flush=True);assert min(agree)==1.0
