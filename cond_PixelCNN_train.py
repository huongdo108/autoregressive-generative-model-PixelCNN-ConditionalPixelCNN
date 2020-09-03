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

from cond_PixelCNN_model import ConditionalPixelCNN

import argparse

parser = argparse.ArgumentParser(description='Conditional PixelCNN')
parser.add_argument('--n_epochs', default=11, type=int, metavar='N',
                    help='number of epochs to run training loop with default = 11')

parser.add_argument('-b', '--batch_size', default=32, type=int,
                    metavar='N', help='batch size training with default = 32')

parser.add_argument('--cuda', dest='cuda', action='store_false',
                    help='use cuda')

parser.add_argument('--skip_training', dest='skip_training', action='store_true',
                    help='skip training')

parser.add_argument('-lr', '--learning_rate', default=0.001, type=float,
                    metavar='LR', help='learning rate')

parser.add_argument('-nc', '--n_channels', default=64, type=int,
                    metavar='NC', help='n_channels in convolutional network')

parser.add_argument('-ks', '--kernel_size', default=7, type=int,
                    metavar='KS', help='kernel_size in convolutional network')



def loss_fn(logits, x):
    """Compute loss of the conditional PixelCNN model. Please see PixelCNN.loss for more details.

    Args:
      logits of shape (batch_size, 256, 28, 28): Logits of the conditional probabilities
                  p(x_i | x_1,...,x_{i-1}) of the 256 intensities of pixel x_i computed using all
                  previous pixel value x_1,...,x_{i-1}.
      x of shape (batch_size, 1, 28, 28): Images used to produce `generated_x`. The values of pixel
                  intensities in x are between 0 and 1.

    Returns:
      loss: Scalar tensor which contains the value of the loss.
    """
    loss = nn.CrossEntropyLoss()
    x = (x*255).reshape(x.shape[0],28,28).long()
    logits = logits.reshape(logits.shape[0],256,28,28)
    l = loss(logits,x)
    return l

def generate(net, labels, image_size=(28, 28), device='cpu'):
    """Generate samples using a trained conditional PixelCNN model.
    Note: use as device labels.device.

    Args:
      net: Conditional PixelCNN model.
      labels of shape (n_samples): Long tensor of the desired classes of the generated samples.
      image_size: Tuple of image size (height, width).
      device:     Device to use.
    
    Returns:
      samples of shape (n_samples, 1, height, width): Generated samples.
    """
    net.eval()
    samples = torch.zeros(labels.shape[0],1,image_size[0],image_size[1]).to(device)
    for i in range(image_size[0]):
        for j in range(image_size[0]):
            out = net(samples,labels)
            probs = F.softmax(out[:,:,i,j], dim=-1).data
            samples[:,:,i,j] = torch.multinomial(probs, 1).float() / 255.0
    return samples

def main():
    """
    train and test the quality of the produced encodings by training a classifier using the encoded images
    """
    args = parser.parse_args()
    data_dir = tools.select_data_dir()
    if args.cuda:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    transform = transforms.Compose([transforms.ToTensor(),])

    trainset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

    # Create network
    net = ConditionalPixelCNN(n_channels=args.n_channels, kernel_size=args.kernel_size)
    net.to(device)

    if not args.skip_training:
        optimizer = torch.optim.Adam(net.parameters(),lr=args.learning_rate)
        for epoch in range(args.n_epochs):
            for i, data in enumerate(trainloader, 0):
                images, labels= data
                images= images.to(device)
                labels = labels.to(device)
                net.train()
                optimizer.zero_grad()            
                y = net(images,labels)
                y= y.to(device)
                loss = loss_fn(y, images)
                loss= loss.to(device)
                loss.backward()
                optimizer.step()
                
            # Generate samples
            with torch.no_grad():
                net.eval()
                labels = torch.cat([torch.arange(10) for _ in range(12)], dim=0).to(device)
                samples = generate(net, labels, device=device)
                tools.plot_generated_samples(samples, ncol=10)

            print('Train Epoch {}: Loss: {:.6f}'.format(epoch +1, loss.item()))    

        # Save the model to disk
        tools.save_model(net, '10_cond_pixelcnn.pth')
    else:
        net = ConditionalPixelCNN(n_channels=args.n_channels, kernel_size=args.kernel_size)
        tools.load_model(net, '10_cond_pixelcnn.pth', device)

    print('Generate samples using the trained model')
    with torch.no_grad():
        net.eval()
        labels = torch.cat([torch.arange(10) for _ in range(12)], dim=0).to(device)
        samples = generate(net, labels, device=device)
        tools.plot_generated_samples(samples, ncol=10)

if __name__ == '__main__':
    main()