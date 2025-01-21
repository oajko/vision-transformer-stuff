'''
Lin embed and pos done elsewhere

Normalization, multihead, ffn
Q,K,V

input -- [batch_size, num_patches, in_dims]
'''

import torch
import torch.nn as nn

class MultiheadAttention(nn.Module):
    def __init__(self,in_dims,mdl_dims,num_heads,*args,**kwargs):
        super(MultiheadAttention,self).__init__(*args,**kwargs)
        self.num_heads=num_heads
        self.mdl_dims=mdl_dims
        self.d_k=mdl_dims//num_heads

        self.lin_q=nn.Linear(in_dims,mdl_dims)
        self.lin_k=nn.Linear(in_dims,mdl_dims)
        self.lin_v=nn.Linear(in_dims,mdl_dims)
        self.fin_lin=nn.Linear(mdl_dims,in_dims)

        self.norm=nn.LayerNorm(in_dims)
    
    def split_heads(self,data):
        bs,chanels,_=data.size()
        return data.view(bs,chanels,self.num_heads,self.d_k).transpose(1, 2)
    
    def rejoin_heads(self,data):
        bs,num_heads,chanels,d_k=data.size()
        return data.transpose(1,2).contiguous().view(bs,chanels,num_heads*d_k)
    
    def forward(self,x):
        x=self.norm(x)
        q,k,v=self.split_heads(self.lin_q(x)),self.split_heads(self.lin_k(x)),self.split_heads(self.lin_v(x))
        x=torch.matmul(q,k.transpose(-2, -1))
        x=x/torch.sqrt(self.num_heads)
        x=nn.functional.softmax(x,dim=-1)
        x=torch.matmul(x,v)
        x=self.rejoin_heads(x)
        return self.fin_lin(x)

class FreeForward(nn.Module):
    def __init__(self,in_dims,ff_dims,*args,**kwargs):
        super(FreeForward,self).__init__(*args,**kwargs)
        self.norm=nn.LayerNorm(in_dims)
        self.ff1=nn.Linear(in_dims,ff_dims)
        self.gelu=nn.GELU()
        self.ff2=nn.Linear(ff_dims,in_dims)
        self.norm2=nn.LayerNorm(in_dims)
    
    def forward(self,x):
        x=self.norm(x)
        x=self.ff1(x)
        x=self.gelu(x)
        x=self.ff2(x)
        x=self.norm2(x)
        return x

class Transformer(nn.Module):
    def __init__(self,*args,**kwargs):
        super(Transformer,self).__init__(*args,**kwargs)
        self.mh=MultiheadAttention(in_dims=14**2,mdl_dims=768,num_heads=8)
        self.ff=FreeForward(in_dims=14**2,ff_dims=768)
    
    def forward(self,x):
        x=x+self.mh(x)
        x=x+self.ff(x)
        return x

