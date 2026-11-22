import os,sys,tensorrt as trt
logger=trt.Logger(trt.Logger.INFO);b=trt.Builder(logger);n=b.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH));p=trt.OnnxParser(n,logger)
if not p.parse_from_file(sys.argv[1]):
 for i in range(p.num_errors):print(p.get_error(i),flush=True)
 raise RuntimeError('parse failed')
c=b.create_builder_config();c.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,4<<30)
print('building FP32 plus QDQ engine',flush=True);e=b.build_serialized_network(n,c)
if e is None:raise RuntimeError('build failed')
with open(sys.argv[2],'wb') as f:f.write(e)
print('built',sys.argv[2],os.path.getsize(sys.argv[2]),flush=True)
