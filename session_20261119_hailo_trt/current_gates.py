import os,sys,numpy as np,torch
import torch.nn.functional as F
D=os.path.dirname(os.path.abspath(__file__)); E='/mnt/storage/lib_project/models/Hailo_model/preproc_hailo/eval'
torch.set_num_threads(2)
if sys.argv[1]=='flat':
 d=np.load(E+'/flat_eval.npz');gt=d['gt'];groups={'synth_photo':(d['set']=='synth')&(d['kind']==1),'synth_flat':(d['set']=='synth')&(d['kind']==0),'uvdoc':d['set']=='uvdoc'}
 ref=np.load(D+'/flat3d_fp.npy',mmap_mode='r').transpose(0,3,1,2)
 for name in sys.argv[2:]:
  p=D+'/'+name
  if not os.path.isfile(p):print('missing',p,flush=True);continue
  m=np.load(p,mmap_mode='r');m=m.transpose(0,3,1,2) if m.shape[-1]==2 else m
  assert m.shape==gt.shape,(m.shape,gt.shape)
  e=np.abs(m-gt).mean((1,2,3))*50; df=np.abs(m-ref).mean((1,2,3))*50
  print(name,'|fp| mean/max %frame',df.mean(),df.max(),{k:{'n':int(s.sum()),'gt_mean':float(e[s].mean()),'p95':float(np.percentile(e[s],95))} for k,s in groups.items()},flush=True)
elif sys.argv[1]=='orient':
 sys.path.insert(0,'/mnt/storage/lib_project/models/layout_teacher/train');import train_orient as T
 net=T.OriNet();net.load_state_dict(torch.load('/mnt/storage/lib_project/models/layout_teacher/staged_v2_20261114/orient3_10000.pt',map_location='cpu',weights_only=True));net.eval(); d=np.load(E+'/orient_eval.npz');nh,nw,k=d['nh'],d['nw'],d['k'];ref=None
 for name in sys.argv[2:]:
  p=D+'/'+name
  if not os.path.isfile(p):print('missing',p,flush=True);continue
  f=np.load(p,mmap_mode='r');assert len(f)==len(k)
  dec=[]
  with torch.no_grad():
   for i in range(len(f)):
    x=torch.from_numpy(np.ascontiguousarray(f[i,:nh[i]//32,:nw[i]//32])).permute(2,0,1)[None].float()
    prob=F.conv2d(F.avg_pool2d(x,7,1),net.fc.weight[:,:,None,None],net.fc.bias).softmax(1)[0].flatten(1).t().numpy();dec.append(T.decide_ll(prob)[0])
  dec=np.asarray(dec);ref=dec if ref is None else ref; up=k==0
  print(name,'n',len(k),'diff_vs_current_canvas_fp',int((dec!=ref).sum()),'false_turn',int((dec[up]!=0).sum()),'upright_n',int(up.sum()),'correct_turn',int((dec[~up]==k[~up]).sum()),'turn_n',int((~up).sum()),flush=True)
else:raise ValueError(sys.argv[1])
