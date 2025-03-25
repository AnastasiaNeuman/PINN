# Sources:

https://doi.org/10.1103/PhysRevE.106.014503 \
"Deep learning and inverse discovery of polymer self-consistent field theory inspired by physics-informed neural networks" \
Phys. Rev. E 106, 014503

https://doi.org/10.1016/j.jcp.2021.110519 \
"Deep learning and self-consistent field theory: A path towards accelerating polymer phase discovery" \
Journal of Computational Physics Volume 443, 15 October 2021, 110519

https://doi.org/10.1088/1367-2630/ab68fc \
"Phase diagrams of polymer-containing liquid mixtures with a theory-embedded neural network" \
Issei Nakamura 2020 New J. Phys. 22 015001

https://doi.org/10.1016/j.commatsci.2020.110224 \
"Deep learning model for predicting phase diagrams of block copolymers" \
Computational Materials Science Volume 188, 15 February 2021, 110224

# Notes:

## Deep learning model for predicting phase diagrams of block copolymers
 - Uses 3DCNN
 - Trained on 3000 metastable structures
 - Inputs were sets of volume fractions of each segment type in each 32 × 32 × 32 grid voxel (obtained from SCFT calculation) and...
 - a one-hot vector of the five types of stable phase labels for the corresponding set of polymer structures and χN values
 - Layers = 2 convolutional layers, a max-pooling layer, 2 more convolutional layers, another max-pooling layer, a flatten layer, 2 dense layers

   ![image](https://github.com/user-attachments/assets/c0a21083-7531-404f-a6a5-a970dedc99c3)

  - a ReLU function was used for activation of each layer except the last one
  - A softmax function used for last layer to output the probabilities of the five phases
  - dropout w/ 0.4 drop rate applied after each max-pooling layer + first dense layer to avoid overfitting
  - to increase amount of training data, they transform (xyz) data by exchanging axes i.e. (zxy), (yzx), etc
  - doubled data by using both A and B volume fraction from same metastable structure as separate data points
  - used Tensorflow
  - 80% of data = training; 20% = validation
  - Adam optimizer used to associate volume fraction data with the categorized stable phase labels
  - loss function = Categorical cross-entropy
  - batch size = 32
  - ran 1000 epochs, loss and accuracy were 8 × 10−7 and 1.000 for trained data and 0.0066 and 0.9982 for validation data
  
  ### PyTorch Implementation Notes
   - Adam Optimizer = torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False, *, foreach=None, maximize=False, capturable=False, differentiable=False, fused=None)
   - ReLU Function = torch.nn.ReLU(inplace=False)
   - Softmax Function = torch.nn.Softmax(dim=None)
   - Cross Entropy Loss = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0)
   - Dropout with 0.4 drop out rate = torch.nn.Dropout(p=0.4, inplace=False) 
   - 3D Convolutional Layer = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
   - Max-Pooling Layer = torch.nn.MaxPool3d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
   - Flatten Layer = torch.nn.Flatten(start_dim=1, end_dim=-1)
   - Dense Layer = torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
     
# Implementation

## 3D CNN

In 3dcnn.py, we will attempt to read in some 3D BCP volume fraction data we have generated previously through SCFT and use this as our training data.

We will use pytorch.  

We will then implement a model like that in "Deep learning model for predicting phase diagrams of block copolymers". 

## 3D CNN TF

The same as 3dcnn, but using tensorflow rather than pytorch. 
