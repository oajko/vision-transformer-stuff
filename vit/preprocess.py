'''
Image input
- Read image
- Resize to 224x224
- Transformations

from sklearn.cluster import KMeans
mdl=KMeans(n_clusters=4)
mdl.fit()

Class called cluster with method KMeans
fit is a call inside same class

In pytorch, modules to preprocess. by io, we use io library, else user defined.


Expect file names of class

-root
-|-0
-|-|-im1
-|-|-im2
-|-1
-|-|-im1
-|-|-im2
....

'''

import torch
from torch.utils.data import Dataset
import os
import cv2

class preprocess(Dataset):
    def __init__(self,xpath,transforms=None):
        self.xpath=xpath
        self.transforms=transforms
        self.location=[]
        self.class_to_idx={i:idx for idx, i in enumerate(os.listdir(self.xpath))}  
        for name,idx in self.class_to_idx.items():
            path=os.path.join(self.xpath,name)
            self.location.extend([(os.path.join(path,im),idx) for im in os.listdir(path)])
    
    def __len__(self):
        return len(self.location)

    def __getitem__(self,index):
        im,label=self.location[index]
        path=os.path.join(os.path.join(self.xpath,label),im)
        image=cv2.imread(path)
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image=self.resize_and_pad(image)
        if self.transforms:
            image=self.transforms(image)
        else:
            image=torch.tensor(image,dtype=torch.float32).permute(2, 0, 1)/255.0
        patches=image.unfold(1,16,16).unfold(2,16,16)
        patches=patches.permute(1,2,0,3,4).reshape(-1,3,16,16)
        patches = patches.flatten(start_dim=2)

        return patches,torch.tensor(label,dtype=torch.long)
    
    def resize_and_pad(self,im):
        h,w=im.shape[:2]
        scale_factor=min(224/h,224/w)
        neww,newh=int(w*scale_factor),int(h*scale_factor)
        resized=cv2.resize(im,(neww,newh))
        delta_w=224-neww
        delta_h=224-newh
        t,b=delta_h//2,delta_h-(delta_h//2)
        l,r=delta_w//2,delta_w-(delta_w//2)
        return cv2.copyMakeBorder(resized,t,b,l,r,cv2.BORDER_CONSTANT,value=[0,0,0])