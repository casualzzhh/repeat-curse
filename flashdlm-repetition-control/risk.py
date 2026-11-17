import torch
from torch import nn
from torch.nn import functional as F

class SpatiotemporalRisk(nn.Module):
    def __init__(self):
        super().__init__()
        self.input=nn.Sequential(nn.Linear(128,64),nn.LayerNorm(64))
        self.space=nn.ModuleList(nn.Conv1d(64,64,k,padding=k//2) for k in (3,5,7))
        self.spatial_fuse=nn.Linear(192,64)
        self.time1=nn.Conv1d(64,64,3);self.time2=nn.Conv1d(64,64,3)
        self.out=nn.Sequential(nn.Linear(129,64),nn.GELU(),nn.Linear(64,3))
    def forward(self,history,mask):
        b,k,l,d=history.shape;assert k==4
        x=self.input(history);flat=x.reshape(b*k,l,64).transpose(1,2)
        space=x+self.spatial_fuse(torch.cat([F.gelu(c(flat)).transpose(1,2) for c in self.space],-1)).reshape(b,k,l,64)
        t=space.permute(0,2,3,1).reshape(b*l,64,k)
        t=F.gelu(self.time1(F.pad(t,(2,0))));t=F.gelu(self.time2(F.pad(t,(2,0))))[:,:,-1].reshape(b,l,64)
        return self.out(torch.cat((space[:,-1],t,mask[:,:,None].float()),-1))
