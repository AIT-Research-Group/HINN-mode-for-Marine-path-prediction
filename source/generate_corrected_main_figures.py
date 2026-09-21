"""
Reproduce the corrected main comparison figures from saved predictions.

Main comparison: Random Forest, RNN, LSTM, HINN.
HINN spatiotemporal encoder-only is NOT a fifth main model; it is an ablation.
"""
import argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

def main(pred_path,outdir):
    D=np.load(pred_path); Y=D["Ytrue"]
    models={"Random Forest":D["RandomForest"],"RNN":D["RNN"],"LSTM":D["LSTM"],"HINN":D["HINN"]}
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    names=list(models); x=np.arange(4); w=.34
    ade=[np.linalg.norm(models[n]-Y,axis=2).mean() for n in names]
    fde=[np.linalg.norm(models[n]-Y,axis=2)[:,-1].mean() for n in names]
    fig,ax=plt.subplots(figsize=(8,4.8))
    ax.bar(x-w/2,ade,w,label="ADE"); ax.bar(x+w/2,fde,w,label="FDE")
    ax.set_xticks(x,names); ax.set_ylabel("Error (m)"); ax.grid(axis="y",alpha=.2); ax.legend()
    fig.savefig(out/"MainComparison.png",dpi=450,bbox_inches="tight"); plt.close(fig)
    t=np.arange(1,31)*10
    fig,ax=plt.subplots(figsize=(8,4.8))
    for n,P in models.items(): ax.plot(t,np.linalg.norm(P-Y,axis=2).mean(0),label=n)
    ax.set_xlabel("Prediction horizon (s)"); ax.set_ylabel("Mean position error (m)")
    ax.grid(alpha=.2); ax.legend(); fig.savefig(out/"HorizonError.png",dpi=450,bbox_inches="tight"); plt.close(fig)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--pred",required=True); ap.add_argument("--out",default="figures")
    a=ap.parse_args(); main(a.pred,a.out)
