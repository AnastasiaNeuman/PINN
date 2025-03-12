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
print(data1)
