# Foreword

This repo demonstrates implementations of some modern CNN networks to understand how it works. Some modifications have been made in order to fit with laptop GPUs for training.

# 0. LeNet
We will pratice first with the LeNet model. This network aims to recognize handwritten digits in images. In 1989, LeCun's team published the first study to successfully train CNNs via backpropagation. This model is also called LeNet-5 because of 5 layers, i.e., 2 convolutional layers and 3 full connected layers, are included in this network.

Link: http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf 

We applied here some advanced techniques such as ReLu rather than Sigmod, MaxPool rather than AvgPool, Dropout and BatchNorm.

NOTE: after 10 epochs, LR=0.001, BS = 64 then Train_Acc: 0.53, Val_Acc: 0.56

# 1. AlexNet (2012)

AlexNet is a deep convolutional neural network developed by Alex Krizhevsky and his colleagues in 2012. It was designed to classify images for the ImageNet LSVRC-2010 competition where it achieved state of the art results. Thus, it has been considered as the most popular and earliest breakthrough algorithm in Computer Vision.

The original paper: https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf 

There are some major key principles to design the AlexNet. First, it operated with 3-channel images having (224,224,3) in size. Second, it used max pooling along with ReLU activations for subsampling. Third, the kernels used for convolutions were randomly either 11x11, 5x5 or 3x3 while kernels used for max pooling were 3x3 in size. At last, it classified images into 1000 classes and was able to train on multiple GPUs.

NOTE: after 10 epochs, LR=0.001, BS = 64 then Train_Acc: 0.84, Val_Acc: 0.71

## Adaptation to train the AlexNet on the CIFAR10 dataset

In order to understand the inspiration of the AlexNet network and to accommodate with the limited memory GPUs fitted on a typical laptop, we will build the Alexnet to train on a small dataset. For that reason, we will be using the CIFAR10 dataset. The dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. We do not apply any more data augmentation methods, except for subtracting it by mean 0.5 in this dataset.

# 2. VGG (2014) 

Link: https://arxiv.org/abs/1409.1556 

The basic building block of CNNs (LeNet, AlexNet) is a sequence of the following: (i) a convolutional layer with padding to maintain the resolution, (ii) a nonlinearity such as a ReLU, (iii) a pooling layer such as max-pooling to reduce the resolution. This leads to a rapid decreases in the spatial resolution (e.g., log2(d) speed). 

To address this issue, Simonyan and Zisserman was to use multiple convolutions in between downsampling via max-pooling in the form of a block. That is, the pairs of a convolution layer and the ReLu function can be applied many times before a max-pooling appears (e.g., CONV x ReLU x CONV x ReLU x CONV x ReLU x POOL). It is a key idea that can create a network with over 100 layers. In particular, a VGG block consists of a sequence of convolutions with 3 × 3 kernels with padding of 1 (keeping height and width) followed by a 2 × 2 max-pooling layer with stride of 2 (halving height and width after each block - reduction by half).

With such an approach, VGG defines a family of networks rather than just a specific CNN model (e.g., VGG-11, 13, 16 , 19). To demonstrate, we just made a VGG-11 in this project. In particular, VGG-11 will have 11 weight layers (convolutional + fully connected). The convolutional layers will have a 3×3 kernel size with a stride of 1 and padding of 1. 2D max pooling in between the weight layers as explained in the paper. Not all the convolutional layers are followed by max-pooling layers. ReLU non-linearity as activation functions.

NOTE: after 10 epochs, LR=0.001, BS = 64 then Train_Acc: 0.85, Val_Acc: 0.78

# SUMMARY (LeNet, AlexNet, VGG)

They share a common design pattern: extract features exploting spatial structure via a sequence of CONV and POOL, and post-process the representations via FC layers. This could lead some issues. First, the FC layers consume tremendous numbers of parameters. For example, VGG-11 requires 25088 x 4096 matrix occupying almost 400 MB of RAM (float 32 bit = 4 byte, thus, ((25088 x 4096 * 4) / (1024 * 1024) = 392 MB). Second, it is equally impossible to add fully connected layers earlier in the network to increase the degree of nonlinearity: doing so would destroy the spatial structure and require potentially even more memory.



The parameters of the CNNs

https://towardsdatascience.com/the-w3h-of-alexnet-vggnet-resnet-and-inception-7baaaecccc96

TENSOR: (batch_size, height, width, depth)

INPUT LAYER: parameters = 0

CONV LAYER: (( kernel_width * kernel_height * num_kernel_prelayer + 1_bias ) * num_filters)

POOL LAYER: parameters = 0

FC LAYER: ((current layer neurons c * previous layer neurons p) + 1 * c)


For example:
IMPORTANT: The RGB image (width, height, 3), and bias = 1

CONV 1 paramters: (( kernel_width * kernel_height * 3 + 1 ) * num_filters)

Pre_layer: (1, 3, 3, 32) -> paramters: = 3*3*32

Curr_Layer: (1, 3, 3, 64)

CONV Parametters: (3*3*32 + 1) * 64

The shape of the output layer

H: Image size, K: Kernel, P: Padding, S: Stride

OUTPUT = ((H - K  + 2P) / S) + 1

# Multi-Branch Networks (GoogLeNet) 2014

Link: https://arxiv.org/pdf/1409.4842.pdf 

GoogLeNet was designed with an Inception block that is a key idea of the network. It solved the problem of selecting convolution kernel sizes among the network body in an ingenious way. In particular, it simply concatenated multi-branch convolutions. In general, GoogLeNet is inspired by AlexNet, VGG in lower layers while the Network in Network in the body for network design.

In this project, we do not use auxiliary classifiers for training.

NOTE: after 2 epochs, LR=0.001, BS = 8 then Train_Acc: 0.70, Val_Acc: 0.76

# Batch Normalization

It is a popular and effective technique that consistently accelerates the convergence of deep networks. Additionally, another benefit of Batch Normalization is its inherent regularization.

Together with residual blocks, they have made the training process possible to train networks with over 100 layers. 

# Residual Networks (ResNet) 2015

Link: https://arxiv.org/pdf/1512.03385.pdf 

As a general trend, stacking more convolution layers could produce deeper networks. However, the authors wanted to ask "is learning better networks as easy as stacking more layers?". They found that the vanishing / exploding gradients could make 
a hard for the training process and a saturation of the accuracy can be reached. Thus, such degradation is not caused by overfitting, and adding more layers leads to higher training errors.

Microsoft engineers created the ResNet to make the CNNs are able to go to a hundred or even a thousand of layers. The key contribution of the ResNet is introducing a shortcut connection that skips one or more layers.
This website will provide an intuition example for the ResNet.
https://towardsdatascience.com/understanding-and-visualizing-resnets-442284831be8

The nework also uses a pattern for downscaling the resolution instead of using pooling layers like normally CNNs do. This means that the down sampling of the resolution through the network is achieved by increasing the stride. In fact, only one max pooling operation is performed in our Conv1 layer, and one average pooling layer at the end of the ResNet, right before the fully connected dense layer.

Within the pattern, there are two kinds of the connection as Identity Shortcut and Projection Shortcut. The identity shortcut simply bypasses the input layer to the addition operator. The projection shortcut performs a convolution operation to ensure the resolution at this addition operation are the same size. 

The following paper will explain a compelling advantage of the skip connection. First, the skip connection will help the gradients to flow (a steady, continuous stream of something.) from later layers to earlier layers. Second, when we train the conv-layer block with backpropagation, so it must do something to minimize the loss. Then it doesn’t make anything messy, because each lower layer takes the skip-connection signal into account when updating the weights.

https://arxiv.org/pdf/1712.09913.pdf

NOTE: after 10 epochs, LR=0.001, BS = 16 then Train_Acc: 0.79, Val_Acc: 0.76

# Densely Connected Networks (DenseNet) 2017

Link: https://arxiv.org/pdf/1608.06993.pdf 

DenseNet brings a new idea for CNNs by exploiting feature reuses. That is, it concates (rather than the addition operation in ResNet) feature-maps learned by different layers to increase variation in the input of subsequent layers and to improve efficiency. Compare to traditional CNNs, the DenseNet introduces ( L*(L+1) ) / 2 connections in a L-layer network. It is also why we call it as a Dense CNN. Another benefit of DenseNet is easy to train that proven with empirical experiments. We will present an adaptive DenseNet for CIFAR-10.

DENSE BLOCK (DB)

Within the DenseNet, they define a hyperparameter, namely growth rate, for each block. For the l-th layer has k0 + k*(l-1) input feature-map. As a result, the DenseNet will be called as DenseNet - Layers - GrowthRate. For example, we will design a network DenseNet - 100 - 12, where each block constitues 16 dense layers. In addition, for block 1, the number of feature maps increase by the growth rate, 12, each time (24, 36, 48, ... 216). The first input has 24 due to the CONV 1. 32x32x3 * CONV(output_channel = 24, kernel = 3, stride = 1, padding = 1).

TRANSITION BLOCK (TB)

The transition blocks are put in between two DBs reducing in half the number of feature maps (theta = 0.5) and also reducing in half the feature size by pooling with stride = 2, a kernel = 2 and a padding = 1.

Although these concatenation operations reuse features to achieve computational efficiency, unfortunately they lead to heavy GPU memory consumption. As a result, applying DenseNet may require more complex memory-efficient implementations that may increase training time

NOTE: after 10 epochs, LR=0.001, BS = 16 then Train_Acc: 0.79, Val_Acc: 0.77

# MobileNetV1 (2017) Google

The MobileNet is mainly inspired by the Xception net using the depthwise separable convolution operation.

Xception: https://arxiv.org/abs/1610.02357

The MobileNet introduces a new appoarch to design a convolution operation. Basically, a standard convolution operation is in charge of filtering features based on the convolutional kernels and combining features in order to produce a new representation. In the MobileNet, the process is divided into two steps called depthwise separable convolutions for substantial reduction in computational cost.

For the standard convolution, it has a computational cost of:
D_f * D_f * M * N * D_k * D_k

where the number of the input channels and the output channels are M and N, respectively. The feature map F has a size of D_f x D_f and the kernel size D_k x D_k.

In contrary, a depthwise separable convolution is made of two layers: depthwise and pointwise convolutions. Depthwise convolutions apply a single filter per each input channel. Pointwise convolutions first apply a 1x1 convolution and then linearly combine the output of depthwise layers. The depthwise separable convolution has a computational cost of:

(D_f * D_f * M * D_k * D_k) + (M * N * D_f * D_f)

As a result, the reduction in computational is given as follows:

( (D_f * D_f * M * D_k * D_k) + (M * N * D_f * D_f) ) / (D_f * D_f * M * N * D_k * D_k)
= 1/N + 1 / (D_k * D_k)

The authors define two hyperparameters alpha and beta. The alpha is the multiplier to compress the model. This is introduced as follows:
M' = alpha * M ; N' = alpha * N where alpha \in (0, 1]. Similar to the alpha, the beta is the multiplier to reduce the resolution of feature maps. That is, the computational cost has the effect of reducing by beta*beta times as D_f' = beta * D_f.

NOTE: after 10 epochs, LR=0.001, BS = 16 then Train_Acc:0.83, Val_Acc: 0.81

# MobileNetV2 (2018) Google

Link: https://arxiv.org/abs/1704.04861 

The main contributions of the MobileNet V2 are two points: The Inverted residual block and the Linear bottlenecks. 

Basically, the residual blocks connect the begining and end of a convolution block with a skip connection. Thus, in an intuition, the traditional residual block follows a wide -> narrow -> wide approach concerning the number of the channels. In contrary, the MobileNet V2 follows a narrow -> wide -> narrow approach. In specifically, the skip connection will connect narrow layers while layers in between are wide. In other words, for the traditional residual blocks are used less convolutions 1x1 for compression, whereas inverted residual blocks will be used much more convolutions 1x1 for expansion within each residual block. By using the inverted residual block, it produces fewer parameters compared to the traditional one. As a result, the MobileNet V2 model is lighter than MobileNet V1. 

The linear bottlenecks are consituted by adding a linear convolution instead of applying a non-linear functions, e.g., ReLU, at the last step of bottlneck blocks. They have shown an evidence that, although the non-linear functions allow us to build the networks that have multiple layers, they destroy / lose information. Thus, with narrow appoarch within the inverted residual blocks, they apply a linear convolution to improve performance of the model.

NOTE: after 10 epochs, LR=0.001, BS = 16 then Train_Acc:0.81, Val_Acc: 0.77


# MobileNetV3 (2019) Google

Link: https://arxiv.org/pdf/1905.02244.pdf

After the success of MobileNet V1 and V2, the design trend of lightweight models attracts an active research attention on DNNs dedicated to embedded devices. There are several interesting works including but not limited to ShuffleNet (V1, V2), MNasNet, CondenseNet, EffNet. Among those networks, a third version of MobileNet, named MobileNetV3, was released by Google engineers. The MobileNet V3 contains some main contributions as follows:

- (1) Efficient Mobile Building Blocks,
- (2) Neural Architecture Search for Block-Wise Search,
- (3) NetAdapt for Layer wise search,
- (4) Network Improvements — Layer removal and H-swish.

Efficient Mobile Building Blocks: The blocks will added squeeze and excitation layers in the initial building block taken from V2. In particular, the squeeze and excitation layers are formatted as: Pool -> Dense -> ReLU -> Dense -> h-swish -> scale back.

Neural Architecture Search (NAS) for light models: is the process of trying to make a model (generally an RNN also called controller) output a thread of modules that can be put together to form a model that gives the best accuracy possible by searching among all the possible combinations. However, to be fitted with MobileNet, the NAS is tuned accordingly. We work on a new reward function: ACC(m) × [LAT(m)/TAR]^w, which considers both accuracy and latency (total inference time) for the model. ACC is accuracy, LAT is latency, TAR is target latency, and m is the model that resulted from the search. Here w is a constant. The authors also observe that for smaller models(that we are looking to search), w needs to be -0.15 (vs the original w = -0.07).

Network Improvements — Layer removal: In the last block, the 1x1 expansion layer taken from the Inverted Residual Unit from MobileNetV2 is moved past the pooling layer. This means the 1x1 layer works on feature maps of size 1x1 instead of 7x7 making it efficient in terms of computation and latency. On the other hand, using 16 filters in the initial 3x3 layer instead of 32, which is the default mobile models. As a result, these changes add up to save nine milliseconds of inference time.

Network Improvements — swish non-linearity instead of ReLU: 

h_swish [x] = x * ((ReLU6(x+3))/ 6)

