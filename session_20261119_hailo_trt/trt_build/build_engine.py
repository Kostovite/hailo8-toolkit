import sys,os,tensorrt as trt
onnx,engine=sys.argv[1:3];logger=trt.Logger(trt.Logger.INFO);builder=trt.Builder(logger)
network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH));parser=trt.OnnxParser(network,logger)
if not parser.parse_from_file(onnx):
 for i in range(parser.num_errors):print(parser.get_error(i),flush=True)
 raise RuntimeError('ONNX parse failed')
config=builder.create_builder_config();config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,4<<30)
# Preserve FP32 for unquantized paths; Q/DQ layers explicitly specify quantization.
profile=builder.create_optimization_profile();profile.set_shape('image',(1,3,224,224),(1,3,896,672),(1,3,896,896));config.add_optimization_profile(profile)
print('building engine',flush=True);serialized=builder.build_serialized_network(network,config)
if serialized is None:raise RuntimeError('TensorRT build failed')
with open(engine,'wb') as f:f.write(serialized)
print('engine built',engine,os.path.getsize(engine),flush=True)
