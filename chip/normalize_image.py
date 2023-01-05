from cProfile import label
import torch

# Tensor (batch_size, channels, width, height)
# a = torch.randn(5,3,32,32)
# Return the first image
# a[0]
# Return the channel B (due to the RGB reversed):
# a[0][0]
# Return the values of the first row of the first channel B in the first image:
# a[0][0][0]
# Return a value of the first pixel of the first row of the first channel B in the first image::
# a[0][0][0][0]

import os
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


os.environ['KMP_DUPLICATE_LIB_OK']='True'
data_path = './CIFAR-10/'

transform_img = transforms.Compose([
    # transforms.Resize(32),
    # transforms.CenterCrop(32),
    transforms.ToTensor(),
    # here do not use transforms.Normalize(mean, std)
])

image_data = torchvision.datasets.ImageFolder(
    root=data_path, transform=transform_img
)

def display_image(images):   
    image_data_loader = DataLoader(        
        image_data, 
        batch_size=len(image_data), 
        shuffle=False, 
        num_workers=0
    )
    images, labels = next(iter(image_data_loader)) 
    images_np = images.numpy()
    img_plt = images_np.transpose(0,2,3,1)
    # display 5th image from dataset
    plt.imshow(img_plt[0])
    plt.show()

# display_image(images)

# Calculate the mean and standard deviation of the dataset
# When the dataset is small and the batch size is the whole dataset. 
# Below is an easy way to calculate when we equate batch size to the whole dataset.

def get_normalization_ds():
    # batch size is whole dataset
    image_data_loader = DataLoader( image_data,  batch_size=len(image_data), shuffle=False, num_workers=0)
    images, lebels = next(iter(image_data_loader))
    print(lebels.shape)
    # shape of images = [b,c,w,h]
    mean, std = images.mean([0,2,3]), images.std([0,2,3])
    print("mean and std: \n", mean, std)

#Mean: tensor([0.4917, 0.4826, 0.4472]); STD: tensor([0.2457, 0.2429, 0.2611])
# get_normalization_ds()

# Calculate the mean and standard deviation by batches

def get_normalization_batch(batch_size):
    image_data_loader = DataLoader(image_data, batch_size=batch_size, shuffle=False, num_workers=0)
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in image_data_loader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)
    print("mean and std: \n", mean, std)

# tensor([0.4917, 0.4826, 0.4472]) tensor([0.2457, 0.2429, 0.2611])
# get_normalization_batch(batch_size=16)

def display_normalized():
    transform_img_normal = transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.4917, 0.4826, 0.4472], std= [0.2457, 0.2429, 0.2611])
    ])

    image_data_normal = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transform_img_normal
    )

    image_data_loader_normal = DataLoader(
        image_data_normal, 
        batch_size=len(image_data_normal), 
        shuffle=False, 
        num_workers=0
    )
    images_normal, labels = next(iter(image_data_loader_normal))
    mean, std = images_normal.mean([0,2,3]), images_normal.std([0,2,3])
    display_image(images_normal)
    print("Nomarlized mean and std: \n", mean, std)

# display_normalized()

image = torch.Tensor(
    [[48, 56, 20, 55, 112, 159],
    [212, 202, 55, 170, 247, 131],
    [173, 29, 149, 88, 116, 77],
    [217, 253, 8, 98, 111, 130],
    [208, 35, 251, 255, 235, 92],
    [53, 192, 82, 179, 64, 174]]
)

# mean, std = image.mean(), image.std()
mean, std = 131.5556, 75.3254
print("mean and std: \n", mean, std)

output= (image - mean) / std

print(output)



