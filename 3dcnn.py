#here we attempt to replicate the model used in
# https://doi.org/10.1016/j.commatsci.2020.110224
# where the input is the 3D volume fraction data on each grid point from SCFT 
# and the labels are the phases  

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import ToTensor
import numpy as np


#converting to tensor
tensor1 = torch.tensor(data1)
tensor2 = torch.tensor(data2)
#transforming the data by exchanging axes to generate additional training data 
#xyz = (0, 1, 2, 3)
#zyx = (2, 1, 0, 3)
#zxy = (2, 0, 1, 3)
#xzy = (0, 2, 1, 3)
#yxz = (1, 0, 2, 3)
#yzx = (1, 2, 0, 3)
#this can definitely be made a loop but lets do it the easy way for now 

tensor11 = torch.index_select(tensor1, 1, torch.LongTensor([2,1,0,3]))
tensor12 = torch.index_select(tensor1, 1, torch.LongTensor([2,0,1,3]))
tensor13 = torch.index_select(tensor1, 1, torch.LongTensor([0,2,1,3]))
tensor14 = torch.index_select(tensor1, 1, torch.LongTensor([1,0,2,3]))
tensor15 = torch.index_select(tensor1, 1, torch.LongTensor([1,2,0,3]))

tensor21 = torch.index_select(tensor2, 1, torch.LongTensor([2,1,0,3]))
tensor22 = torch.index_select(tensor2, 1, torch.LongTensor([2,0,1,3]))
tensor23 = torch.index_select(tensor2, 1, torch.LongTensor([0,2,1,3]))
tensor24 = torch.index_select(tensor2, 1, torch.LongTensor([1,0,2,3]))
tensor25 = torch.index_select(tensor2, 1, torch.LongTensor([1,2,0,3]))

#combine tensors into one data tensor
data = torch.stack((tensor1, tensor11, tensor12, tensor13, tensor14, tensor15, tensor2, tensor21, tensor22, tensor23, tensor24, tensor25))


#now we have a small set of data to start as a training set
#next we have to create our labels

labels_map = {
    0: "Lamellar",
    1: "Cylindrical",
    2: "Spherical",
    3: "Gyroid",
    4: "Disordered",
}

#all the data should be lamellar, so we'll start with a basic label tensor of zeros
labels = torch.zeros(12)

#Adam optimizer needs to be added in here when you have more data to categorize

#now let's try creating a custom dataset combining the tensors and labels
dataset = TensorDataset(data, labels)
#print("Dataset sample:", dataset[1])

#next we will set up the DataLoader
batch_size = 2
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

#set device 
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
#next, we can define our neural network class

class NeuralNetwork(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv_pool_stack = nn.Sequential(

#here were creating the sequential operation of 3D convolution layer -> 3D convolution layer -> pooling layer -> 3D layer -> 3D layer -> pooling layer
#in Conv3D
# in_channels = density, so 1		
#16 = # filters, each filter creates one output channel, so output channels = 16
#kernel size defined by depth, width, height
#unsure what to use here, so we will just start with numbers in image
			nn.Conv3d(1, 16, 30),
#next we add a ReLU for activation
			nn.ReLU(),
#next we want to add another 3D convolution layer, again with 16 filters
			nn.Conv3d(1, 16, (28, 28,28)),
#and another ReLU for activation
			nn.ReLU(),
#next a max pooling layer, also with 16 filters
			nn.MaxPool3d((14,14,14)),
			nn.ReLU(),
#apply a drop out w 0.4 drop rate
			nn.Dropout(p=0.4),
#convolution layer with 32 filters, 12x12x12
			nn.Conv3d(1, 32, (12,12,12)),
			nn.ReLU(),
#convolution layer with 32 filters, 10x10x10
			nn.Conv3d(1, 32, (10,10,10)),
			nn.ReLU(),
#max pooling layer, 32 filters, 5x5x5
			nn.MaxPool3d((5,5,5,)),
			nn.ReLU(),
#another drop out with 0.4 drop rate
			nn.Dropout(p=0.4),
		)
#next we define the flatten and dense portion
		self.flatten_dense_stack = nn.Sequential(
			nn.Flatten(),
			nn.ReLU(),
#512 linear number is from paper
			nn.Linear(4000,512),
			nn.ReLU(),
#add another dropout here
			nn.Dropout(p=0.4),
#5 number is from paper
			nn.Linear(512,5),
		)
#we don't end with another ReLU, we will use a softmax after we call the model to get the probabilities
		
	def forward(self, x):
		x = self.conv_pool_stack(x)
		logits = self.flatten_dense_stack(x)
		return logits

#print structure of model
model = NeuralNetwork().to(device)
print(model)

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()

# Initialize the Adam Optimizer
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#define training loop for model
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

#define test loop for model
def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

#running training and tetsing loops
epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(dataloader, model, loss_fn, optimizer)
    test_loop(dataloader, model, loss_fn)
print("Done!")

