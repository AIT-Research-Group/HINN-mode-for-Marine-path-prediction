"""
Train the proposed HINN correctly:
HINN = validated HINN spatiotemporal encoder + 100-epoch Hyper-Information neural fusion.

The Hyper-Information branch consumes target/neighbor interaction descriptors plus the
HINN spatiotemporal trajectory prior. Validation selects the best epoch and fusion alpha.
The test future is not used for model selection.
"""
import argparse, copy, importlib.util, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

def nf(G,N,M):
    D=N-G
    dN=np.zeros_like(N); dN[:,1:]=N[:,1:]-N[:,:-1]
    dD=np.zeros_like(D); dD[:,1:]=D[:,1:]-D[:,:-1]
    fs=[]
    for X in [G,N,D,dN,dD]:
        for h in [6,12,24,60]:
            Xh=X[:,-h:]; W=M[:,-h:,None]; den=np.maximum(W.sum(1),1)
            mu=(Xh*W).sum(1)/den
            var=((Xh-mu[:,None,:])**2*W).sum(1)/den
            fs += [mu,np.sqrt(np.maximum(var,0))]
        fs.append(X[:,-1])
    return np.concatenate(fs,1).astype(np.float32)

def tf(T,S,B,M):
    fs=[S,B]
    for h in [6,12,24,60]:
        X=T[:,-h:]; W=M[:,-h:,None]; den=np.maximum(W.sum(1),1)
        mu=(X*W).sum(1)/den
        var=((X-mu[:,None,:])**2*W).sum(1)/den
        fs += [mu,np.sqrt(np.maximum(var,0))]
    fs.append(T[:,-1])
    return np.concatenate(fs,1).astype(np.float32)

class HyperFusionHead(nn.Module):
    def __init__(self,din=617):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(din,192),nn.LayerNorm(192),nn.GELU(),nn.Dropout(.14),
            nn.Linear(192,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(.10),
            nn.Linear(128,96),nn.GELU(),nn.Linear(96,60))
    def forward(self,x): return self.net(x).view(-1,30,2)

def metric(P,Y):
    e=np.linalg.norm(P-Y,axis=2)
    return float(e.mean()),float(e[:,-1].mean())

def main(data,backbone_checkpoint,backbone_source,out):
    D=np.load(data)
    spec=importlib.util.spec_from_file_location("bb",backbone_source)
    bb=importlib.util.module_from_spec(spec); spec.loader.exec_module(bb)
    # constants expected beside the prepared data can be generated from the original package
    const=str(__import__("pathlib").Path(data).parent/"SCALERS_AND_CONSTANTS.npz")
    model=bb.BEST2Backbone(backbone_checkpoint,const,device="cpu")
    def bp(split):
        sk={"tr":"StrN","va":"SvaN","te":"SteN"}[split]
        return model.predict(D["T"+split],D["G"+split],D["N"+split],D["M"+split],
                             D[sk],D["B"+split],batch=512)
    Ptr,Pva,Pte=bp("tr"),bp("va"),bp("te")
    Ytr,Yva,Yte=D["Ytr"].astype(np.float32),D["Yva"].astype(np.float32),D["Yte"].astype(np.float32)
    def eng(split):
        sk={"tr":"StrN","va":"SvaN","te":"SteN"}[split]
        return np.concatenate([nf(D["G"+split],D["N"+split],D["M"+split]),
                               tf(D["T"+split],D[sk],D["B"+split],D["M"+split])],1)
    Xtr0,Xva0,Xte0=eng("tr"),eng("va"),eng("te")
    xmu=Xtr0.mean(0); xsd=np.where(Xtr0.std(0)<1e-6,1,Xtr0.std(0))
    Xtr,Xva,Xte=(Xtr0-xmu)/xsd,(Xva0-xmu)/xsd,(Xte0-xmu)/xsd
    pmu=Ptr.reshape(len(Ptr),-1).mean(0); psd=np.where(Ptr.reshape(len(Ptr),-1).std(0)<1e-3,1,Ptr.reshape(len(Ptr),-1).std(0))
    HXtr=np.concatenate([Xtr,(Ptr.reshape(len(Ptr),-1)-pmu)/psd],1).astype(np.float32)
    HXva=np.concatenate([Xva,(Pva.reshape(len(Pva),-1)-pmu)/psd],1).astype(np.float32)
    HXte=np.concatenate([Xte,(Pte.reshape(len(Pte),-1)-pmu)/psd],1).astype(np.float32)
    R=Ytr-Ptr; rmu=R.mean(0); rsd=np.where(R.std(0)<1e-3,1,R.std(0)); RN=(R-rmu)/rsd
    torch.manual_seed(20261105); np.random.seed(20261105)
    h=HyperFusionHead(HXtr.shape[1]); opt=torch.optim.AdamW(h.parameters(),lr=7e-4,weight_decay=5e-4)
    dl=DataLoader(TensorDataset(torch.from_numpy(HXtr),torch.from_numpy(RN.astype(np.float32))),
                  batch_size=1024,shuffle=True,generator=torch.Generator().manual_seed(20261105))
    def pred(X,P):
        h.eval()
        with torch.no_grad(): rn=h(torch.from_numpy(X)).numpy()
        return P+rn*rsd+rmu
    best=(1e30,None,0); hist=[]
    for ep in range(1,101):
        h.train(); total=n=0
        for x,y in dl:
            opt.zero_grad(); q=h(x)
            loss=F.smooth_l1_loss(q,y,beta=.7)+.22*F.smooth_l1_loss(q[:,-1],y[:,-1],beta=.7)
            loss.backward(); torch.nn.utils.clip_grad_norm_(h.parameters(),3.0); opt.step()
            total+=float(loss.detach())*len(x); n+=len(x)
        pv=pred(HXva,Pva); ade,fde=metric(pv,Yva)
        score=ade+.25*fde; hist.append([ep,total/n,ade,fde])
        if score<best[0]: best=(score,copy.deepcopy(h.state_dict()),ep)
    h.load_state_dict(best[1]); pv=pred(HXva,Pva)
    bestalpha=(1e30,1.0)
    for a in np.linspace(0,1.25,126):
        ade,fde=metric(Pva+a*(pv-Pva),Yva); s=ade+.25*fde
        if s<bestalpha[0]: bestalpha=(s,float(a))
    pt=pred(HXte,Pte); pt=Pte+bestalpha[1]*(pt-Pte)
    print("Best epoch",best[2],"alpha",bestalpha[1],"TEST",metric(pt,Yte))
    torch.save({"hyper_head_state":h.state_dict(),"best_epoch":best[2],"trained_epochs":100,
                "fusion_alpha":bestalpha[1],"feature_mu":xmu,"feature_sd":xsd,
                "backbone_path_mu":pmu,"backbone_path_sd":psd,"residual_mu":rmu,"residual_sd":rsd},out)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True); ap.add_argument("--backbone-checkpoint",required=True)
    ap.add_argument("--backbone-source",required=True); ap.add_argument("--out",default="HINN_100EPOCH.pt")
    a=ap.parse_args(); main(a.data,a.backbone_checkpoint,a.backbone_source,a.out)
