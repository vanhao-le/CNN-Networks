import torch
import torch.nn.functional as F
import torch.nn as nn

def conv_bn_relu(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True)
    )
        
def conv_dw(in_channels, out_channels, stride):
    return nn.Sequential(
        # dw
        # with group option, Pytorch will apply singel filter for each input channel
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
        nn.BatchNorm2d(num_features=in_channels),
        nn.ReLU(inplace=True),

        # pw
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
    )


class MobileNetV1(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(MobileNetV1, self).__init__()

        self.model = nn.Sequential(       
            #  default stride = 2
            # (3, 32, 32) -> (32, 32, 32)
            conv_bn_relu(in_channels=in_channels, out_channels= 32, stride= 1),
            # (32, 32, 32) - > (64, 32, 32)
            conv_dw(in_channels= 32, out_channels= 64, stride= 1),
            # (64, 32, 32) - > (128, 32, 32)
            conv_dw(in_channels= 64, out_channels= 128, stride= 1),
            # (128, 32, 32) -> (128, 32, 32)
            conv_dw(in_channels= 128, out_channels= 128, stride= 1),
            # (128, 32, 32) -> (256, 16, 16)
            conv_dw(in_channels= 128, out_channels= 256, stride= 2),
            # ((256, 16, 16) -> (256, 16, 16)
            conv_dw(in_channels= 256, out_channels= 256, stride= 1),
            # (256, 16, 16) -> (512, 16, 16)
            conv_dw(in_channels= 256, out_channels= 512, stride= 1),
            conv_dw(in_channels= 512, out_channels= 512, stride= 1),
            conv_dw(in_channels= 512, out_channels= 512, stride= 1),
            conv_dw(in_channels= 512, out_channels= 512, stride= 1),
            conv_dw(in_channels= 512, out_channels= 512, stride= 1),
            conv_dw(in_channels= 512, out_channels= 512, stride= 1),
            # (512, 16, 16) -> (1024, 8, 8)        
            conv_dw(in_channels= 512, out_channels= 1024, stride= 2),
            # (1024, 8, 8) -> (1024, 8, 8)
            conv_dw(in_channels= 1024, out_channels= 1024, stride= 1),
            nn.AvgPool2d(kernel_size=8),
        )
        self.fc = nn.Linear(1024, num_classes)

        
    def forward(self, x):        
        out = self.model(x)        
        out = out.view(-1, 1024)
        out = self.fc(out)
        return out


def test():
    net = MobileNetV1(in_channels=3, num_classes=10)
    x = torch.randn(1, 3, 32, 32)
    y = net(x)
    print(y)

# test()