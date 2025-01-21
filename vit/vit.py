'''
Images made into patches.

- linear projection

Flatten images to linear
Apply position embedding
Input into transformer
MPL layer

'''

import torch
import transformer
import torch.nn as nn
import numpy as np

class VIT(nn.Module):
    def __init__(self,*args,**kwargs):
        super(VIT,self).__init__(*args,**kwargs)
        self.tf1=transformer.Transformer()

        self.mlp=nn.Sequential(
            nn.Linear(768,512),
            nn.ReLU(),
            nn.Linear(512,64),
            nn.ReLU(),
            nn.Linear(64,2),
        )
    
    def forward(self,x):
        x=x.view(-1,768)+self.position_encode(x)
        x=self.tf1(x)
        x=self.mlp(x)
        return x
    
    def position_encode(self,_):
        pos_arr=np.expand_dims(np.arange(768),1)
        dim_arr=np.expand_dims(np.arange(768),0)
        denom=np.power(10000,(2*dim_arr//2))/np.float32(768)
        return torch.tensor(pos_arr/denom,dtype=torch.float32)