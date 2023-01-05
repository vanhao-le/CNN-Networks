import torch
import os

# define image
IMAGE_SIZE = 32
# specify Imagenet mean and standard deviation for the RGB image.
# MEAN = [0.485, 0.456, 0.406]
# STD = [0.229, 0.224, 0.225]

# for CIFAR-10
MEAN = [0.4917, 0.4826, 0.4472]
STD = [0.2457, 0.2429, 0.2611]


# define training hyperparameters
INIT_LR = 0.001
BATCH_SIZE = 16
EPOCHS = 10

# define the root dataset path
BASE_PATH = "CIFAR-10"
TRAIN = os.path.join(BASE_PATH, "train")
VAL = os.path.join(BASE_PATH, "val")


# set the device we will be using to train the model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PLOT = os.path.join("output", "train.png")
MODEL_PATH = os.path.join("output", "model.pth")