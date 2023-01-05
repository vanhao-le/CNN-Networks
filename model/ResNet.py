import torch.nn as nn
import math

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride = 1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        return out

# ResNet
class ResNet(nn.Module):
    def __init__(self, block, num_layers, num_classes=10 ):
        super(ResNet, self).__init__()
        self.num_layers = num_layers        
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        # feature map size = 32x32x16
        self.layer1 = self.make_layer(block, 16, 16,  stride=1)
        # feature map size = 16x16x32
        self.layer2 = self.make_layer(block, 16, 32, stride=2)
        # feature map size = 8x8x64
        self.layer3 = self.make_layer(block, 32, 64, stride=2)

        self.avg_pool = nn.AvgPool2d(kernel_size= 8)
        self.fc = nn.Linear(64, num_classes)
    
    # block: Type[Union[BasicBlock, Bottleneck]]
    def make_layer(self, block, in_channels, out_channels, stride = 1):
        downsample = None
        if (stride != 1) or (in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))
        # total number of layers if 6n + 2. if n is 5 then the depth of network is 32.
        for _ in range(1, self.num_layers):
            layers.append(block(out_channels, out_channels))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
def resnet():
    block = ResidualBlock
	# total number of layers if 6n + 2. if n is 5 then the depth of network is 32.
    model = ResNet(block=block, num_layers=3) 
    return model