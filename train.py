import torchsummary
from chip import config
from chip import create_dataloaders
from imutils import paths
import torch
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

import os

from model.VGG import VGGM
from model.LeNet import LeNet
from model.AlexNet import AlexNet
from model.GoogLeNet import GoogLeNet
from model.ResNet import resnet
from model.DenseNet import DenseNetBC_CIFAR
from model.MobileNetV1 import MobileNetV1
from model.MobileNetV2 import MobileNetV2


os.environ['KMP_DUPLICATE_LIB_OK']='True'

def main():

    train_transform = transforms.Compose([
        # Scale the image up to a square of 40 pixels in both height and width
        # transforms.Resize(40),
        # Randomly crop a square image of 40 pixels in both height and width to
        # produce a small square of 0.64 to 1 times the area of the original
        # image, and then scale it to a square of 32 pixels in both height and width
        # transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.64, 1.0), ratio=(1.0, 1.0)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(90),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])



    # Notice that we do not perform data augmentation inside the validation transformer
    val_transform = transforms.Compose([
        # transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])

    # create data loaders
    (train_ds, train_loader) = create_dataloaders.get_dataloader(
        config.TRAIN,
        transforms=train_transform,
        batch_size=config.BATCH_SIZE
    )

    (val_ds, val_loader) = create_dataloaders.get_dataloader(    
        config.VAL,
        transforms=val_transform,
        batch_size=config.BATCH_SIZE, shuffle=False
    )
    # input size
    img, label = train_ds[0]
    print("The input size", img.shape, label)    
    print("Length of the train / val set:", len(train_ds), "/", len(val_ds))
    num_classes = len(train_ds.classes)
    print("The number of classes:", num_classes)    

    # calculate steps per epoch for training and validation set
    train_steps = len(train_loader.dataset) // config.BATCH_SIZE
    val_steps = len(val_loader.dataset) // config.BATCH_SIZE   

    # initialize the LeNet model
    print("[INFO] initializing the model...")
    # model = GoogLeNet(input_dim=3, num_classes=10)
    # model = resnet()    
    # model = DenseNetBC_CIFAR()
    # model = MobileNetV1(in_channels=3, num_classes=num_classes)
    model = MobileNetV2(num_classes=10)
    model.to(config.DEVICE)

    torchsummary.summary(model, (3, 32, 32))
    # return
        
    # x = torch.randn(1, 3, 32, 32)
    # out = model(x)
    # print(out)
    
    # initialize our optimizer and loss function
    opt = optim.SGD(model.parameters(), lr=config.INIT_LR, weight_decay=0.005, momentum=0.9)
    # Learning rate schedule
    # scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[10, 50], gamma=0.5)
    loss_func = nn.CrossEntropyLoss()
    # initialize a dictionary to store training history
    H = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    # measure how long training is going to take
    print("[INFO] training the network...")
        # loop over our epochs
    for e in range(config.EPOCHS):
        # set model for training
        model.train()
        # initialize loss values for training / validation
        total_train_loss = 0.0
        total_val_loss = 0.0
        # initialize corrected predictions for training / validation
        train_correct = 0
        val_correct = 0
        # loop over training set
        for (imgs, lbs) in train_loader:
            # send input to gpu
            # print(imgs.shape, lbs.shape)
            (imgs, lbs) = (imgs.to(config.DEVICE) , lbs.to(config.DEVICE))
            # perform forward()
            preds = model(imgs)
            loss = loss_func(preds, lbs)
            # zero gradient 
            opt.zero_grad()
            # perform backpropagation step
            loss.backward()
            opt.step()
            # add the loss to the total training loss
            total_train_loss += loss
            train_correct += (preds.argmax(1) ==  lbs).type(torch.float).sum().item()

        # switch off autograd for evaluation
        with torch.no_grad():
            # set the model in evaluation mode
            model.eval()
            # loop over the validation set
            for (imgs, lbs) in val_loader:
                # send input to gpu
                (imgs, lbs) = (imgs.to(config.DEVICE) , lbs.to(config.DEVICE))
                # make the predictions and calculate the validation loss
                preds = model(imgs)
                loss = loss_func(preds, lbs)
                total_val_loss += loss
                # calculate the number of correct predictions
                val_correct += (preds.argmax(1) ==  lbs).type(torch.float).sum().item()
        
        # calculate the average training and validation loss
        avg_train_loss = total_train_loss / train_steps
        avg_val_loss = total_val_loss / val_steps
        # calculate the training and validation accuracy
        train_acc = train_correct / len(train_ds)
        val_acc = val_correct / len(val_ds)

        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, config.EPOCHS))
        print("Train loss: {:.6f}, Train accuracy: {:.2f}".format(avg_train_loss, train_acc))
        print("Val loss: {:.6f}, Val accuracy: {:.2f}".format(avg_val_loss, val_acc))

        
        # cpu_avg_train_loss = torch.tensor(avg_train_loss, dtype=torch.float32)
        # cpu_avg_val_loss = torch.tensor(avg_val_loss, dtype=torch.float32)
        cpu_avg_train_loss = avg_train_loss.clone().detach()
        cpu_avg_val_loss = avg_val_loss.clone().detach()
        # update our training history
        H["train_loss"].append(cpu_avg_train_loss.clone().detach().cpu().numpy())
        H["train_acc"].append(train_acc)
        H["val_loss"].append(cpu_avg_val_loss.clone().detach().cpu().numpy())
        H["val_acc"].append(val_acc)

    
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["val_loss"], label="val_loss")
    plt.plot(H["train_acc"], label="train_acc")
    plt.plot(H["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(config.MODEL_PLOT)

    # serialize the model to disk
    torch.save(model, config.MODEL_PATH)


    return


if __name__ == '__main__':
    main()
    