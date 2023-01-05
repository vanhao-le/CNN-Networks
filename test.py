import torch
import torch.nn as nn


img = torch.ones(1, 3, 7, 7)

m = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1)

out = m(img)

print(out, out.shape)




