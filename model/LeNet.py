import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary



class LeNet(nn.Module):

    def __init__(self, input_dim, num_classes = 10):
        super(LeNet, self).__init__()
        # Tensor (batch_size, channels, height, width)
        # INPUT (batch_size, 3, 32, 32)
        # CONV => AVG => CONV => AVG => FC => FC => FC
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels= input_dim, out_channels= 6, kernel_size=5, stride= 1, padding=0),
            nn.BatchNorm2d(num_features=6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels= 6, out_channels= 16, kernel_size=5, stride= 1, padding=0),
            nn.BatchNorm2d(num_features=16),         
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # fully connected linear layers
        self.linear_layers = nn.Sequential(            
            nn.Linear(in_features= 16*5*5, out_features=120),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=84, out_features= num_classes)
        )

       
    # forward
    def forward(self, x):       
        out = self.conv_layers(x)
        # print("------------------------------>", out.shape)
        # flatten to prepare for the fully connected layers
        out = out.view(out.size(0), -1)
        out = self.linear_layers(out)     
        return out

        
# def test():
#     net = LeNet(input_dim=3, num_classes=10)    
#     x = torch.randn(2,3,32,32)
#     y = net(x)
#     torchsummary.summary(net, x)
#     print(y.size())

# test()


