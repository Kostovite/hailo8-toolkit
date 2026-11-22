import os,sys,pickle,numpy as np,torch
sys.path.insert(0,'/mnt/storage/lib_project/models/layout_teacher/train')
import train_ppdl3 as TP
D=os.path.dirname(os.path.abspath(__file__)); E='/mnt/storage/lib_project/models/Hailo_model/preproc_hailo/eval'
part=int(sys.argv[1]);torch.set_num_threads(2);torch.manual_seed(0)
m=TP.build_model();m.load_state_dict(torch.load('/mnt/storage/lib_project/models/layout_teacher/staged_v2_20261114/r5_ord5.pt',map_location='cpu',weights_only=True));m.eval().cuda()
x=np.load(E+'/layout_x.npy',mmap_mode='r');out={};
with torch.no_grad():
 for idx in range(part,len(x),2):
  a=torch.from_numpy(np.array(x[idx],copy=True)).permute(2,0,1)[None].float().cuda()/255
  s,c,b=TP.predict(m,a);out[idx]=(s[0].cpu().numpy(),c[0].cpu().numpy(),b[0].cpu().numpy())
  if len(out)%25==0:print('part',part,'pages',len(out),'/',(len(x)+1-part)//2,flush=True)
with open(D+f'/layout_r5_fp_preds_{part}.pkl','wb') as f:pickle.dump(out,f)
print('done',part,len(out),flush=True)
