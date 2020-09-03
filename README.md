# Autoregressive generative model with PixelCNN and Conditional PixelCNN

## Overview
The goal of this repository is to get familiar with autoregressive generative models using the PixelCNN model and Conditional Pixel CNN

## Data 

I used standard MNIST data from torchvision.datasets.MNIST

## PixelCNN

The model is decribed in Section 3.5 of [this paper](https://arxiv.org/pdf/1601.06759.pdf).

<img src="https://github.com/huongdo108/autoregressive-generative-model-PixelCNN-ConditionalPixelCNN/blob/master/images/pixelcnn_context.png" align="centre">

PixelCNN is an autoregressive model for the distribution of images.
The joint probability <img src="https://render.githubusercontent.com/render/math?math=p(x)"> of an <img src="https://render.githubusercontent.com/render/math?math=n \times n"> image **x** is written as a product
of the conditional distributions over the pixels:
<img src="https://render.githubusercontent.com/render/math?math=p(x) = \prod_{i=1}^{n^2} p(x_i|x_1,...,x_{i-1})">

The order of the pixels in the model is chosen arbitrarily. It is convenient to choose the first pixel <img src="https://render.githubusercontent.com/render/math?math=x_1"> to be in the top left corner and the last pixel <img src="https://render.githubusercontent.com/render/math?math=x_{n^2}"> in the bottom right corner. 

The conditional distribution <img src="https://render.githubusercontent.com/render/math?math=p(x_i|x_1,...,x_{i-1})"> is modeled using a deep convolutional neural network. This network is designed in the following way:
- The input and the output images have the same size.
- The value of pixel **i** in the output image is only affected by pixels of the input image that precede **i** (as shown on the figure). This can be achieved by a network which is a stack of masked convolutional layers.


**Masked convolutional layer**

- A masked convolutional layer is a standard convolutional layer whose kernel has zero values below and to the right of the central location. The remaining values of the kernel are the parameters of the layer which are trained in a standard way.

- A simple way to implement the masked convolutional layer is to use a standard `nn.Conv2d` module and multiply its kernel by a binary mask in the `forward()` function.

- The layer can have two kinds of binary masks:
  1. with zero in the center (`blind_center=True`):

  <img src="https://github.com/huongdo108/autoregressive-generative-model-PixelCNN-ConditionalPixelCNN/blob/master/images/masked_conv.png" align="centre">

  2. with one in the center (`blind_center=False`):

  <img src="https://github.com/huongdo108/autoregressive-generative-model-PixelCNN-ConditionalPixelCNN/blob/master/images/masked_conv_2.png" align="centre">

  The first type of mask is used in the first layer of our PixelCNN model and the second type of mask is used in the remaining layers. This kind of masking ensures that the output pixels are not affected by subsequent pixels of the input image.

- The binary mask can be created using function [`register_buffer`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module.register_buffer). This way the mask will be automatically transferred to the given device when calling `model.to(device)`.

- The convolutional layer should **not** have a bias term because the biases are not affected by the mask.

**Using 2d batch normalization in the model**

Using the batch normalization significantly improves the convergence of the training procedure. However, when the network is in the `train()` mode (that means that the batch norm uses statistics computed from the mini-batch), the batch norm breaks the required causality structure of the PixelCNN model. Since all the locations contribute to the batch statistics, the subsequent pixels affect the values of the previous pixels. Even though, the batch normalization represents the whole batch using only two statistics, the network seems to learn to make use of the information in the subsequent pixels. This is a possible explanation of the following observation: the loss computed in the `eval()` mode can be substantially larger compared to the loss computed in the `train()` mode.

When the network is used in the `eval()` model, the required causality structure is preserved. Even though using running statistics may result in larger loss values, it does not seem to affect significantly the quality of the generated images.

**Loss function for training PixelCNN**

In PixelCNN, the conditional distributions <img src="https://render.githubusercontent.com/render/math?math=p(x_i|x_1,...,x_{i-1})"> of pixel intensities <img src="https://render.githubusercontent.com/render/math?math=x_i"> are multinomial distributions over 256 possible values. Thus, the loss function is the mean of the cross-entropy classification losses with 256 classes computed for each pixel <img src="https://render.githubusercontent.com/render/math?math=x_i">.

**Generation procedure**

The procedure that generates samples using a trained PixelCNN model. The generation proceeds as follows:
* Initialize `samples` tensor as images with all zeros.
* Apply the PixelCNN model to `samples` tensor. The output will contain logits (probabilities before softmax) over 256 pixel intensity values for pixels in all locations. However, on the first iteration we are only interested in the pixel intensities at the first location (0,0) because <img src="https://render.githubusercontent.com/render/math?math=p(x_1)"> needs to be computed.
* Use computed probabilities to sample a pixel intensity value for the pixel at location (0, 0). Write the sampled value to location (0, 0) of the `samples` tensor.
* Apply the model to the `samples` tensor. Now the sampled value of <img src="https://render.githubusercontent.com/render/math?math=x_1"> is used by the model to generate the probabilities of pixel intensities for the pixel at location (0, 1), thus <img src="https://render.githubusercontent.com/render/math?math=p(x_2\mid x_1)"> is computed.

* a pixel intensity value is sampled for the second pixel and written to the corresponding location of `samples` tensor.
* The process is continued until all the values of the `samples` tensor are changed.

**Result**

 <img src="https://github.com/huongdo108/autoregressive-generative-model-PixelCNN-ConditionalPixelCNN/blob/master/images/pixelcnn_result.png" align="centre">

## Conditional generation with PixelCNN

The basic idea of the conditioning is described in Section 2.3 of [this paper](https://arxiv.org/pdf/1606.05328.pdf). However, a much simpler model is implemented in this repository.

**Masked convolutional layer**

Similar to PixelCNN.

**Conditional PixelCNN**

Conditional PixelCNN models allows to generate images of a desired class. This can be achieved by providing the desired class label to every layer of the PixelCNN model. In this repository, I do it in the following way: the input of each masked convolutional layer is:

<img src="https://render.githubusercontent.com/render/math?math=\mathbf{x} + \mathbf{W} \mathbf{h}">

where
  * <img src="https://render.githubusercontent.com/render/math?math=\mathbf{x}"> is the output of the previous layer
  * <img src="https://render.githubusercontent.com/render/math?math=\mathbf{h}"> is a 10-dimensional one-hot coded vector of the desired class
  * <img src="https://render.githubusercontent.com/render/math?math=\mathbf{W}"> is <img src="https://render.githubusercontent.com/render/math?math=c \times 10"> matrix (parameter of a fully-connected layer), where **c** is the number of input channels in the masked convolutional layer.

**Loss function for training conditional PixelCNN**

Similar to PixelCNN.

**Generation procedure**

The `generate()` function is *almost* identical to the `generate()` function from the PixelCNN. It additionally receives the labels of the desired classes so that they can be used in the forward computations of the conditional PixelCNN model.

**Result**

 <img src="https://github.com/huongdo108/autoregressive-generative-model-PixelCNN-ConditionalPixelCNN/blob/master/images/cond_pixelcnn_result.png" align="centre">

 <img src="https://github.com/huongdo108/autoregressive-generative-model-PixelCNN-ConditionalPixelCNN/blob/master/images/cond_pixelcnn_result2.png" align="centre">