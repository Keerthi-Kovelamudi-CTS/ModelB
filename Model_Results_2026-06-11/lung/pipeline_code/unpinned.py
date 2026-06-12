import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
rng=np.random.RandomState(42)
def ci(fn,y,s,nb=500):
    v=[]; N=len(y); idx=np.arange(N)
    for _ in range(nb):
        b=rng.randint(0,N,N)
        if len(np.unique(y[b]))<2: continue
        v.append(fn(y[b],s[b]))
    lo,hi=np.percentile(v,[2.5,97.5]); return lo,hi
print(f"{'ratio':5s} {'win':9s} {'pt':7s} {'Sens':>5s} {'Spec':>5s} {'PPV':>5s} {'NPV':>5s} {'TP/FP/FN/TN':>16s} {'AUROC(CI)':>20s} {'AUPRC(CI)':>20s}")
for r in ["1","10"]:
  for w in ["5yr","10yr","20yr","lifetime"]:
    f=f"12mo_1:{r}/lookback/{w}/internal_scores_{w}.npz"
    try: d=np.load(f)
    except: print(f"1:{r} {w} MISSING"); continue
    y=d["y"].astype(int); s=d["s"].astype(float)
    au=roc_auc_score(y,s); al,ah=ci(roc_auc_score,y,s)
    ap=average_precision_score(y,s); pl,ph=ci(average_precision_score,y,s)
    fpr,tpr,th=roc_curve(y,s); T_y=th[np.argmax(tpr-fpr)]
    for nm,T in [("Youden",T_y),("0.5",0.5)]:
        pr=s>=T; tp=int((pr&(y==1)).sum());fp=int((pr&(y==0)).sum());fn=int((~pr&(y==1)).sum());tn=int((~pr&(y==0)).sum())
        se=tp/(tp+fn) if tp+fn else 0; sp=tn/(tn+fp) if tn+fp else 0; pv=tp/(tp+fp) if tp+fp else 0; nv=tn/(tn+fn) if tn+fn else 0
        print(f"1:{r:3s} {w:9s} {nm:7s} {se*100:5.1f} {sp*100:5.1f} {pv*100:5.1f} {nv*100:5.1f} {f'{tp}/{fp}/{fn}/{tn}':>16s} {f'{au:.3f}[{al:.3f}-{ah:.3f}]':>20s} {f'{ap:.3f}[{pl:.3f}-{ph:.3f}]':>20s}")
