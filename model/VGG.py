import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary

# A-11, B-13, D-16, E-19
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'M': [64, 128, 256, 512, 512, 'M'],
}



class VGG(nn.Module):

    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        # Tensor (batch_size, channels, height, width)
        # INPUT (batch_size, 3, 32, 32)
        self.num_classes = 10        
        self.conv_layers = self._make_layers(cfg[vgg_name])

        # fully connected linear layers
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features= 512*16*16, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=64, out_features= self.num_classes)
        )
    
    # Define a block
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                            nn.BatchNorm2d(x),
                            nn.ReLU(inplace=True)]
                in_channels = x

        # layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)

       
    # forward
    def forward(self, x):
    
        out = self.conv_layers(x)
        # print("------------------------------>", out.shape)
        # flatten to prepare for the fully connected layers
        out = out.view(out.size(0), -1)
        out = self.linear_layers(out)     
        return out



def VGG11():
    return VGG('A')

def VGG13():
    return VGG('B')

def VGG16():
    return VGG('D')

def VGG19():
    return VGG('E')

def VGGM():
    return VGG('M')

        
# def test():
#     net = VGG(input_dim=3, num_classes=10)    
#     x = torch.randn(2,3,32,32)
#     y = net(x)
#     torchsummary.summary(net, x)
#     print(y.size())

# test()


