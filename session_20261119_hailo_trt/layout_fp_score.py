import os,sys,pickle,numpy as np
sys.path.insert(0,'/mnt/storage/lib_project/models/layout_teacher/train');import train_ppdl3 as TP
D=os.path.dirname(os.path.abspath(__file__));E='/mnt/storage/lib_project/models/Hailo_model/preproc_hailo/eval'
GT,NM=pickle.load(open(E+'/layout_gt.pkl','rb'));p={}
for i in (0,1):p.update(pickle.load(open(D+f'/layout_r5_fp_preds_{i}.pkl','rb')))
assert len(p)==len(NM)==len(GT)
for name in ('thesis','d4la','photo_thesis','photo_d4la'):
 G=[];det=[]
 for k in [j for j,n in enumerate(NM) if n==name]:
  G.append(GT[k]);s,c,b=p[k];keep=s>.01
  for sc,kk,q in zip(s[keep],c[keep],b[keep]):det.append({'image_id':len(G),'category_id':int(kk)+1,'bbox':[float(q[0]*1000),float(q[1]*1000),float((q[2]-q[0])*1000),float((q[3]-q[1])*1000)],'score':float(sc)})
 a,per=TP.coco_ap(G,det);print('RESULT',name,'n',len(G),'AP',a,'boxAP',per['_box'],'boxAP75',per['_box75'],flush=True)
