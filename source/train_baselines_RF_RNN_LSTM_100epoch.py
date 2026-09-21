"""
Train the SAME-AIS baselines used in the main paper comparison.
Main paper models: Random Forest, RNN, LSTM, HINN.
RNN and LSTM run exactly 100 epochs; best checkpoints are selected on validation.
"""
import argparse, copy, os, joblib, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestRegressor

def target_features(T,S,B,M):
    fs=[S,B,T[:,-1]]
    for h in [6,12,24,60]:
        X=T[:,-h:]; W=M[:,-h:,None]; den=np.maximum(W.sum(1),1)
        mu=(X*W).sum(1)/den
        var=((X-mu[:,None,:])**2*W).sum(1)/den
        fs += [mu,np.sqrt(np.maximum(var,0))]
    fs += [T[:,-1]-T[:,-6],T[:,-1]-T[:,-12],T[:,-1]-T[:,0]]
    return np.concatenate(fs,1).astype(np.float32)

class RNN(nn.Module):
    def __init__(self,h=40):
        super().__init__()
        self.rnn=nn.RNN(15,h,batch_first=True,nonlinearity="tanh")
        self.meta=nn.Sequential(nn.Linear(17,28),nn.GELU())
        self.head=nn.Sequential(nn.Linear(h+28,80),nn.GELU(),nn.Linear(80,60))
    def forward(self,T,M,S,B):
        _,h=self.rnn(T*M.unsqueeze(-1))
        return self.head(torch.cat([h[-1],self.meta(torch.cat([S,B],1))],1)).view(-1,30,2)

class LSTM(nn.Module):
    def __init__(self,h=40):
        super().__init__()
        self.rnn=nn.LSTM(15,h,batch_first=True)
        self.meta=nn.Sequential(nn.Linear(17,28),nn.GELU())
        self.head=nn.Sequential(nn.Linear(h+28,80),nn.GELU(),nn.Linear(80,60))
    def forward(self,T,M,S,B):
        _,(h,_)=self.rnn(T*M.unsqueeze(-1))
        return self.head(torch.cat([h[-1],self.meta(torch.cat([S,B],1))],1)).view(-1,30,2)

def metric(P,Y):
    e=np.linalg.norm(P-Y,axis=2); return float(e.mean()),float(e[:,-1].mean())

def train_seq(name,model,D,seed,out):
    Ttr=D["Ttr"].astype(np.float32); Mtr=D["Mtr"].astype(np.float32)
    Str=D["StrN"].astype(np.float32); Btr=D["Btr"].astype(np.float32)
    Tva=D["Tva"].astype(np.float32); Mva=D["Mva"].astype(np.float32)
    Sva=D["SvaN"].astype(np.float32); Bva=D["Bva"].astype(np.float32)
    Ytr=D["Ytr"].astype(np.float32); Yva=D["Yva"].astype(np.float32)
    ymu=Ytr.mean(0); ysd=np.where(Ytr.std(0)<1e-3,1,Ytr.std(0)); Yn=(Ytr-ymu)/ysd
    torch.manual_seed(seed); np.random.seed(seed)
    opt=torch.optim.AdamW(model.parameters(),lr=7e-4,weight_decay=2e-4)
    dl=DataLoader(TensorDataset(torch.from_numpy(Ttr),torch.from_numpy(Mtr),
        torch.from_numpy(Str),torch.from_numpy(Btr),torch.from_numpy(Yn.astype(np.float32))),
        batch_size=1024,shuffle=True,generator=torch.Generator().manual_seed(seed))
    def pred():
        model.eval(); outp=[]
        with torch.no_grad():
            for i in range(0,len(Tva),2048):
                outp.append(model(torch.from_numpy(Tva[i:i+2048]),torch.from_numpy(Mva[i:i+2048]),
                                  torch.from_numpy(Sva[i:i+2048]),torch.from_numpy(Bva[i:i+2048])).numpy())
        return np.concatenate(outp)*ysd+ymu
    best=(1e30,None,0); hist=[]
    for ep in range(1,101):
        model.train(); total=n=0
        for T,M,S,B,Y in dl:
            opt.zero_grad(); P=model(T,M,S,B)
            loss=F.smooth_l1_loss(P,Y,beta=.7)+.20*F.smooth_l1_loss(P[:,-1],Y[:,-1],beta=.7)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),3.0); opt.step()
            total+=float(loss.detach())*len(T); n+=len(T)
        Pv=pred(); ade,fde=metric(Pv,Yva); score=ade+.25*fde
        hist.append([ep,total/n,ade,fde])
        if score<best[0]: best=(score,copy.deepcopy(model.state_dict()),ep)
    model.load_state_dict(best[1])
    torch.save({"state":model.state_dict(),"ymu":ymu,"ysd":ysd,
                "best_epoch":best[2],"trained_epochs":100},os.path.join(out,f"{name}_100EPOCH.pt"))
    np.save(os.path.join(out,f"{name}_TRAIN_HISTORY.npy"),np.asarray(hist,float))

def main(data,out):
    os.makedirs(out,exist_ok=True); D=np.load(data)
    Xtr=target_features(D["Ttr"],D["StrN"],D["Btr"],D["Mtr"])
    rf=RandomForestRegressor(n_estimators=100,min_samples_leaf=3,max_features=.60,
                             n_jobs=-1,random_state=20260901)
    rf.fit(Xtr,D["Ytr"].reshape(len(D["Ytr"]),-1))
    joblib.dump(rf,os.path.join(out,"RandomForest.joblib"),compress=3)
    train_seq("RNN",RNN(40),D,20261031,out)
    train_seq("LSTM",LSTM(40),D,20261041,out)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--data",required=True); ap.add_argument("--out",default="baseline_models")
    a=ap.parse_args(); main(a.data,a.out)
