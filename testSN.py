import Liblinks.utis as f_s
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import *
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as tf
import os

from torchvision.utils import make_grid
from pylab import plt
from Liblinks.sn_lib import *
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

#sn_linear=SNLinear(2,2,bias=False,Ip=30)
#gf_linear=spectral_norm(nn.Linear(2,2,bias=False),n_power_iterations=30)
#print(sn_linear.weight)
#print(gf_linear.weight)
#gf_linear.weight=sn_linear.weight
#print(sn_linear.weight)
#print(gf_linear.weight)

#input = torch.randn(10, 2)
#out=sn_linear(input)
#outgf=gf_linear(input)

#print(out)
#print(outgf)


class testnet(nn.Module):
    def __init__(self):
        super(testnet, self).__init__()
        self.l1=nn.Linear(2,3)
        self.l2=nn.Linear(1,3)

    def forward(self,x,y=None):
        if y is not None:
            return self.l1(x)+self.l2(y)
        else:
            return self.l1(x)

testn=testnet()
opter=torch.optim.Adam(testn.parameters())
print('------')
for para in testn.parameters():
    print(para)

data=torch.tensor([1.,2.]).type(torch.FloatTensor)
y=torch.tensor([1.]).type(torch.FloatTensor)
out_loss=testn(data).mean()
out_loss.backward()
opter.step()
print('------')
for para in testn.parameters():
    print(para)

out_loss=testn(data,y).mean()
out_loss.backward()
opter.step()

print('-----')
for para in testn.parameters():
    print(para)