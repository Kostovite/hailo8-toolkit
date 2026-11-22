import os,sys,numpy as np,torch
from hailo_platform import HEF,VDevice,HailoStreamInterface,InferVStreams,ConfigureParams,InputVStreamParams,OutputVStreamParams,FormatType
D=os.path.dirname(os.path.abspath(__file__));sys.path[:0]=['/mnt/storage/lib_project/models/Dolphin','/mnt/storage/lib_project/models/Dolphin/wordlid'];from region_router import RegionRouter
ck=torch.load('/mnt/storage/lib_project/models/Dolphin/wordlid/data/grid/region_router.pt',map_location='cpu',weights_only=False);net=RegionRouter();net.load_state_dict(ck['router']);net.eval()
x=np.load(D+'/router_val.npy').astype(np.float32);hef=HEF(D+'/router_all16.hef');dev=VDevice();ng=dev.configure(hef,ConfigureParams.create_from_hef(hef,interface=HailoStreamInterface.PCIe))[0];ip=InputVStreamParams.make(ng,format_type=FormatType.FLOAT32);op=OutputVStreamParams.make(ng,format_type=FormatType.FLOAT32);iname=hef.get_input_vstream_infos()[0].name;outputs=hef.get_output_vstream_infos();print('input',iname,'outputs',[(o.name,o.shape) for o in outputs],flush=True)
ret={o.name:[] for o in outputs}
with InferVStreams(ng,ip,op) as pipe,ng.activate(ng.create_params()):
 for i in range(0,len(x),8):
  r=pipe.infer({iname:x[i:i+8,None,None,:]})
  for name in ret:ret[name].append(r[name])
with torch.no_grad():ref=net(torch.from_numpy(x))
for name,values in ret.items():
 z=np.concatenate(values).reshape(len(x),-1);kind={6:'kind',3:'body',21:'lang'}[z.shape[1]];y=ref[kind].numpy();print('RESULT',kind,'n',len(x),'decision_agreement',np.mean(z.argmax(1)==y.argmax(1)),'mean_abs',np.abs(z-y).mean(),'max_abs',np.abs(z-y).max(),flush=True)
