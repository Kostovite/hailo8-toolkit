import numpy as np,os,sys,glob
D=sys.argv[1];R='/mnt/storage/lib_project/models/Dolphin/wordlid/data/grid';rng=np.random.default_rng(617)
for tag,n in [('g5train',320),('g5val',100)]:
 files=sorted(glob.glob(f'{R}/{tag}_img_*.npy'));assert files,tag
 pools=[np.load(p,mmap_mode='r') for p in files];ix=rng.choice(sum(len(x) for x in pools),size=n,replace=False);ends=np.cumsum([len(x) for x in pools]);arr=np.empty((n,896,896),np.uint8)
 for j,i in enumerate(ix):
  k=int(np.searchsorted(ends,i,side='right'));o=0 if k==0 else ends[k-1];arr[j]=pools[k][i-o]
 np.savez_compressed(f'{D}/grid_{tag}_320.npz',images=arr);print(tag,n,flush=True)
