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

from PixelCNN_model import PixelCNN

import argparse

parser = argparse.ArgumentParser(description='PixelCNN')
parser.add_argument('--n_epochs', default=11, type=int, metavar='N',
                    help='number of epochs to run training loop with default = 11')

parser.add_argument('-b', '--batch_size', default=32, type=int,
                    metavar='N', help='batch size training with default = 32')

parser.add_argument('--cuda', dest='cuda', action='store_false',
                    help='use cuda')

parser.add_argument('--skip_training', dest='skip_training', action='store_true',
                    help='skip training')

parser.add_argument('-lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='learning rate')

parser.add_argument('-nc', '--n_channels', default=64, type=int,
                    metavar='NC', help='n_channels in convolutional network')

parser.add_argument('-ks', '--kernel_size', default=7, type=int,
                    metavar='KS', help='kernel_size in convolutional network')

parser.add_argument('-ns', '--n_samples', default=120, type=int,
                    metavar='NS', help='number of generated samples')

def loss_fn(logits, x):
    """Compute PixelCNN loss. The PixelCNN model uses conditional distributions $p(x_i|x_1,...,x_{i-1})$
    for pixel intensities x_i which are multinomial distributions over 256 possible values. Thus the loss
    function is the cross-entropy classification loss with 256 intensity values computed for each pixel x_i.

    NB: Our tests assume the cross-entropy loss function which has log_softmax implemented inside,
    such as `nn.CrossEntropyLoss`.

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

def generate(net, n_samples, image_size=(28, 28), device='cpu'):
    """Generate samples using a trained PixelCNN model.

    Args:
      net:        PixelCNN model.
      n_samples:  Number of samples to generate.
      image_size: Tuple of image size (height, width).
      device:     Device to use.
    
    Returns:
      samples of shape (n_samples, 1, height, width): Generated samples.
    """
    net.eval()
    samples = torch.zeros(n_samples,1,image_size[0],image_size[1]).to(device)
    for i in range(image_size[0]):
        for j in range(image_size[0]):
            out = net(samples)
            probs = F.softmax(out[:,:,i,j], dim=-1).data
            samples[:,:,i,j] = torch.multinomial(probs, 1).float() / 255.0
    return samples


def main():
    """
    train and test the quality of the produced encodings by training a classifier using the encoded images
    """
    args = parser.parse_args()
    if args.cuda:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    data_dir = tools.select_data_dir()

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    net = PixelCNN(n_channels=args.n_channels, kernel_size=args.kernel_size)
    net.to(device)

    if not args.skip_training:
        optimizer = torch.optim.Adam(net.parameters(),lr=args.learning_rate)
        
        for epoch in range(args.n_epochs):
            for i, data in enumerate(trainloader, 0):
                images, _= data
                images= images.to(device)
                net.train()
                optimizer.zero_grad()            
                y = net(images)
                y = y.to(device)
                loss = loss_fn(y, images)
                loss = loss.to(device)
                loss.backward()
                optimizer.step()
                
            with torch.no_grad():
                samples = generate(net, n_samples=args.n_samples, device=device)
                tools.plot_generated_samples(samples)
        

            print('Train Epoch {}: Loss: {:.6f}'.format(epoch +1, loss.item()))    

        # Save the model to disk 
        tools.save_model(net, '10_pixelcnn.pth')
    else:
        net = PixelCNN(n_channels=args.n_channels, kernel_size=args.kernel_size)
        tools.load_model(net, '10_pixelcnn.pth', device) 

    # Generate samples
    print('Generate samples with trained model')
    with torch.no_grad():
      samples = generate(net, n_samples=args.n_samples, device=device)
      tools.plot_generated_samples(samples)

if __name__ == '__main__':
    main()