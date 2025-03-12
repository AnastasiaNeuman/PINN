#here we attempt to replicate the model used in
# https://doi.org/10.1016/j.commatsci.2020.110224
# where the input is the 3D volume fraction data on each grid point from SCFT 
# and the labels are the phases  

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np

#here we are reading in a data file as a numpy array

data1 = np.loadtxt("fp5chi10p9rhoSpecies0.dat", delimiter=' ', usecols=(0,1,2,3))
data2 = np.loadtxt("fp5chi10p9rhoSpecies1.dat", delimiter=' ', usecols=(0,1,2,3))

#converting to tensor
tensor1 = torch.tensor(data1)
tensor2 = torch.tensor(data2)
print(tensor1)
size = tensor1.size()
print(size)
#transforming the data by exchanging axes to generate additional training data 
#xyz = (0, 1, 2, 3)
#zyx = (2, 1, 0, 3)
#zxy = (2, 0, 1, 3)
#xzy = (0, 2, 1, 3)
#yxz = (1, 0, 2, 3)
#yzx = (1, 2, 0, 3)
#this can definitely be made a loop but lets do it the easy way for now 

tensor11 = torch.permute(tensor1, (2, 1, 0, 3))
print(tensor11)
