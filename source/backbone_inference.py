import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CorrectedGNNLSTM(nn.Module):
    def __init__(self,ft=15,fg=9,gdim=64,h=80,ctx=128,sdim=15):
        super().__init__()
        self.g_self=nn.Linear(fg,gdim)
        self.g_neigh=nn.Linear(fg,gdim,bias=False)
        self.g_norm=nn.LayerNorm(gdim)
        self.target_lstm=nn.LSTM(ft,h,batch_first=True)
        self.graph_lstm=nn.LSTM(gdim,h,batch_first=True)
        self.fuse=nn.Sequential(nn.Linear(4*h+sdim,ctx),nn.LayerNorm(ctx),nn.GELU(),nn.Dropout(.12))
        self.head=nn.Sequential(nn.Linear(ctx+sdim,96),nn.GELU(),nn.Dropout(.08),nn.Linear(96,2))
    def forward(self,xt,xself,xnbr,m,s):
        xt=xt*m.unsqueeze(-1); xself=xself*m.unsqueeze(-1); xnbr=xnbr*m.unsqueeze(-1)
        g=F.gelu(self.g_norm(self.g_self(xself)+self.g_neigh(xnbr))); g=g*m.unsqueeze(-1)
        ot,(ht,_)=self.target_lstm(xt); og,(hg,_)=self.graph_lstm(g)
        mr=m[:,-12:].unsqueeze(-1); den=mr.sum(1).clamp_min(1)
        pt=(ot[:,-12:]*mr).sum(1)/den; pg=(og[:,-12:]*mr).sum(1)/den
        c=self.fuse(torch.cat([ht[-1],hg[-1],pt,pg,s],1))
        return self.head(torch.cat([c,s],1))

class PolarAttnGNN(nn.Module):
    def __init__(self,ft=15,fg=9,gdim=64,h=80,ctx=128,sdim=15):
        super().__init__()
        self.g_self=nn.Linear(fg,gdim)
        self.g_neigh=nn.Linear(fg,gdim,bias=False)
        self.g_norm=nn.LayerNorm(gdim)
        self.target_lstm=nn.LSTM(ft,h,batch_first=True)
        self.graph_lstm=nn.LSTM(gdim,h,batch_first=True)
        self.att_t=nn.Linear(h,1); self.att_g=nn.Linear(h,1)
        self.fuse=nn.Sequential(nn.Linear(4*h+sdim,ctx),nn.LayerNorm(ctx),nn.GELU(),nn.Dropout(.12))
        self.head=nn.Sequential(nn.Linear(ctx+sdim,96),nn.GELU(),nn.Dropout(.08),nn.Linear(96,2))
    @staticmethod
    def pool(o,m,att):
        score=att(o).squeeze(-1).masked_fill(m<0.5,-1e4)
        a=torch.softmax(score,1)
        return (o*a.unsqueeze(-1)).sum(1)
    def forward(self,xt,xself,xnbr,m,s):
        xt=xt*m.unsqueeze(-1); xself=xself*m.unsqueeze(-1); xnbr=xnbr*m.unsqueeze(-1)
        g=F.gelu(self.g_norm(self.g_self(xself)+self.g_neigh(xnbr))); g=g*m.unsqueeze(-1)
        ot,(ht,_)=self.target_lstm(xt); og,(hg,_)=self.graph_lstm(g)
        pt=self.pool(ot,m,self.att_t); pg=self.pool(og,m,self.att_g)
        c=self.fuse(torch.cat([ht[-1],hg[-1],pt,pg,s],1))
        return self.head(torch.cat([c,s],1))

class BEST2Backbone:
    def __init__(self,checkpoint_path,constants_path,device=None):
        self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        C=np.load(constants_path)
        self.dvscale=C['dvscale'].astype(np.float32); self.pol_scale=C['pol_scale'].astype(np.float32)
        self.tt=C['tt'].astype(np.float32)
        ck=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        self.cart=CorrectedGNNLSTM(sdim=15).to(self.device); self.polar=PolarAttnGNN(sdim=15).to(self.device)
        self.cart.load_state_dict(ck['corrected_cartesian_state']); self.polar.load_state_dict(ck['polar_seed2_state'])
        self.cart.eval(); self.polar.eval()
        self.ac=float(ck['alpha_cartesian']); self.ap=float(ck['alpha_polar'])
        self.wp=float(ck['ensemble_polar_weight']); self.wc=float(ck['ensemble_cartesian_weight'])
    @torch.no_grad()
    def predict(self,T,G,N,M,S,B,batch=256):
        T=np.asarray(T,np.float32); G=np.asarray(G,np.float32); N=np.asarray(N,np.float32)
        M=np.asarray(M,np.float32); S=np.asarray(S,np.float32); B=np.asarray(B,np.float32)
        out=[]; tt=torch.from_numpy(self.tt).to(self.device).view(1,-1,1)
        dscale=torch.from_numpy(self.dvscale).to(self.device); pscale=torch.from_numpy(self.pol_scale).to(self.device)
        for i in range(0,len(T),batch):
            sl=slice(i,i+batch)
            xt=torch.from_numpy(T[sl]).to(self.device); xs=torch.from_numpy(G[sl]).to(self.device)
            xn=torch.from_numpy(N[sl]).to(self.device); mm=torch.from_numpy(M[sl]).to(self.device)
            ss=torch.from_numpy(S[sl]).to(self.device); bb=torch.from_numpy(B[sl]).to(self.device)
            zc=self.cart(xt,xs,xn,mm,ss); vc=bb+zc*dscale; C=tt*vc[:,None,:]
            C0=tt*bb[:,None,:]; C=C0+self.ac*(C-C0)
            zp=self.polar(xt,xs,xn,mm,ss); bsp=torch.sqrt((bb**2).sum(1)+1e-8); bang=torch.atan2(bb[:,1],bb[:,0])
            d=zp*pscale; sp=torch.clamp(bsp+d[:,0],min=0); ang=bang+d[:,1]
            vp=torch.stack([sp*torch.cos(ang),sp*torch.sin(ang)],1); P=tt*vp[:,None,:]
            P0=tt*bb[:,None,:]; P=P0+self.ap*(P-P0)
            out.append((self.wp*P+self.wc*C).cpu().numpy())
        return np.concatenate(out).astype(np.float32)
