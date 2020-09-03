import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import tools
import tests

class MaskedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, blind_center=False):
        """
        Args:
          in_channels (int): Number of input channels.
          out_channels (int): Number of output channels.
          kernel_size (int): Kernel size similar to nn.Conv2d layer.
          blind_center (bool): If True, the kernel has zero in the center.
        """
        super(MaskedConv2d,self).__init__()
        self.Conv2d = nn.Conv2d(in_channels, out_channels, kernel_size,padding=kernel_size//2,bias=False)
        mask = torch.ones(1,1,kernel_size,kernel_size)
        mask[:,:,kernel_size//2,kernel_size//2+(blind_center==False):] = 0
        mask[:,:,kernel_size//2+1:] = 0
        self.register_buffer('mask',mask)
       

    def forward(self, x):
        """
        Args:
          x of shape (batch_size, in_channels, height, width): Input images.
        
        Returns:
          y of shape (batch_size, out_channels, height, width): Output images.
        """
        self.Conv2d.weight.data *= self.mask
        
        return self.Conv2d.forward(x)

class PixelCNN(nn.Module):
    def __init__(self, n_channels=64, kernel_size=7):
        """PixelCNN model."""
        super(PixelCNN, self).__init__()
        self.layer1 = nn.Sequential(
        MaskedConv2d(1,n_channels,kernel_size,blind_center=True),
        nn.BatchNorm2d(n_channels),
        nn.ReLU(),
        )
        
        self.layer2 = nn.Sequential(
        MaskedConv2d(n_channels,n_channels,kernel_size,blind_center=False),
        nn.BatchNorm2d(n_channels),
        nn.ReLU(),
        )
        
        self.layer3 = nn.Sequential(
        MaskedConv2d(n_channels,n_channels,kernel_size,blind_center=False),
        nn.BatchNorm2d(n_channels),
        nn.ReLU(),
        )
        
        self.layer4 = nn.Sequential(
        MaskedConv2d(n_channels,n_channels,kernel_size,blind_center=False),
        nn.BatchNorm2d(n_channels),
        nn.ReLU(),
        )
        
        self.layer5 = nn.Sequential(
        MaskedConv2d(n_channels,n_channels,kernel_size,blind_center=False),
        nn.BatchNorm2d(n_channels),
        nn.ReLU(),
        )
        
        self.layer6 = nn.Sequential(
        MaskedConv2d(n_channels,n_channels,kernel_size,blind_center=False),
        nn.BatchNorm2d(n_channels),
        nn.ReLU(),
        )
        
        self.layer7 = nn.Sequential(
        MaskedConv2d(n_channels,n_channels,kernel_size,blind_center=False),
        nn.BatchNorm2d(n_channels),
        nn.ReLU(),
        )
        
        self.layer8 = nn.Sequential(
        MaskedConv2d(n_channels,n_channels,kernel_size,blind_center=False),
        nn.BatchNorm2d(n_channels),
        nn.ReLU(),
        )
        
        self.layer9 = nn.Conv2d(n_channels,256,1)


    def forward(self, x):
        """Compute logits of the conditional probabilities p(x_i|x_1, ..., x_{i-1}) of the PixelCNN model.
        
        Args:
          x of shape (batch_size, 1, 28, 28): Tensor of input images.
        
        Returns:
          logits of shape (batch_size, 256, 28, 28): Tensor of logits of the conditional probabilities
                                                      for each pixel.
        
        NB: Do not use softmax nonlinearity after the last layer.
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        return x