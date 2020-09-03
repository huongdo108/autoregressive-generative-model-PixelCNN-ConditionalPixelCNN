import os
import numpy as np

import copy
import torch
import matplotlib.pyplot as plt
from IPython import display

import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as utils


def get_binary_receptive_field(net, image_size, i, j):
    #inputs = torch.empty(32, 1, image_size[0], image_size[1], requires_grad=True)
    #inputs.data = torch.arange(32 * image_size[0] * image_size[1]).float().view(inputs.shape)
    inputs = torch.randn(32, 1, image_size[0], image_size[1], requires_grad=True)
    net.eval()
    net.to('cpu')
    outputs = net(inputs)
    loss = outputs[0,0,i,j]
    loss.backward()
    rfield = torch.abs(inputs.grad[0, 0]) > 0
    return rfield


# Test MaskedConv2d
def set_weights_MaskedConv2d(net):
    for param in net.parameters():
        if param.shape[:2] == torch.Size([2, 1]) and param.shape[3] == 5:
            param.data[:1] = 1/(5*5/2)
            param.data[1:] = -1/(5*5/2)


