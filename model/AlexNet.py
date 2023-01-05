import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, input_dim, num_classes = 10):
        super(AlexNet, self).__init__()
        # Tensor (batch_size, channels, height, width)
        # INPUT (batch_size, 3, 32, 32)
        # LAYER 1: CONV => BATCHNORM => ReLU => MAXPOOL
        # 11x11 Conv(96), stride 4
        # 3x3 MaxPool, stride 2
        # LAYER 1: (1, 96, 27, 27)

        # Equation for output convolution operation
        # ((H - K  + 2P) / S) + 1
        # ((W - K  + 2P) / S) + 1

        # nn.Conv2d(in_channels= input_dim, out_channels = 96, kernel_size=11, stride=4, padding=0),
        # modified layer 1
        # Convolution layer 1, 3-channel input, 96 convolution kernels, kernel size 7 * 7, stride 2, padding 2
        # After this layer, the image size becomes ((32 - 7 + 2 * 2) / 2) + 1 => 15 * 15
        # After 3 * 3 Maximum pooling and 2 steps, the image becomes ((15 - 3) / 2) + 1 => 7 * 7

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels= input_dim, out_channels = 96, kernel_size=7, stride=2, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 3, stride= 2)
        )
        # LAYER 2: CONV => BATCHNORM => ReLU => MAXPOOL
        # 5x5 Conv(256), stride 1, pad 2
        # 3x3 MaxPool, stride 2
        #     
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels= 96, out_channels= 256, kernel_size= 5, stride = 1, padding= 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 3, stride= 2)
        )
        # LAYER 3: CONV => BATCHNORM => ReLU
        # 3x3 Conv(384), stride 1, pad 1
            
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU()
        )
        # LAYER 4: CONV => BATCHNORM => ReLU
        # 3x3 Conv(384), stride 1, pad 1 
     
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU()
        )
        # LAYER 5: CONV => BATCHNORM => ReLU => MAXPOOL
        # 3x3 Conv(256), stride 1, pad 1
        # 3x3 MaxPool, stride 2
        # DELETED  nn.MaxPool2d(kernel_size = 3, stride = 2)

        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()           
        )
        # LINEAR 1: FC => RELU
        # FC: 4096
        # Drop_out 0.5
        
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 3 * 3, 1024),
            nn.ReLU()
        )
        # LINEAR 1: FC => RELU
        # FC: 4096
        # Drop_out 0.5
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        self.fc2= nn.Sequential(
            nn.Linear(512, num_classes)
        )
    # forward
    def forward(self, x):            
        # print("0 ------------------------------>", x.shape)
        out = self.layer1(x)
        # print("1 ------------------------------>", out.shape)
        out = self.layer2(out)
        # print("2 ------------------------------>", out.shape)
        out = self.layer3(out)        
        # print("3 ------------------------------>", out.shape)
        out = self.layer4(out)
        # print("4 ------------------------------>", out.shape)
        out = self.layer5(out)        
        # print("5 ------------------------------>", out.shape)
        out = out.reshape(out.size(0), -1)        
        # print("6 ------------------------------>", out.shape)
        out = self.fc(out)        
        out = self.fc1(out)        
        out = self.fc2(out)
        
        return out

        



