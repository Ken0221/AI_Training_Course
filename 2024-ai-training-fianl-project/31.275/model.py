import torch
import torch.nn as nn
from thop import profile


class SuperResolution(nn.Module):
    '''
        This is just an exmaple of a SR model, 
        you can change everything except the model name "SuperResolution"
    '''
    def __init__(self):
        super(SuperResolution, self).__init__()
        self.conv1 = nn.Conv2d(3, 54, 3, 1, 1)
        self.conv1_1 = nn.Conv2d(54, 54, 3, 1, 1)
        self.conv2 = nn.Conv2d(54, 36, 3, 1, 1)
        self.conv3 = nn.Conv2d(36, 36, 1, 1, 0)
        self.conv4 = nn.Conv2d(36, 36, 1, 1, 0)
        self.conv5 = nn.Conv2d(36, 36, 1, 1, 0)
        self.conv6 = nn.Conv2d(36, 36, 1, 1, 0)
        self.conv7 = nn.Conv2d(36, 36, 1, 1, 0)
        self.conv8 = nn.Conv2d(36, 54, 3, 1, 1)
        self.conv9 = nn.Conv2d(54, 54, 1, 1, 0)
        self.PS = nn.PixelShuffle(3)
        self.PRelu = nn.PReLU()
        self.convf = nn.Conv2d(6, 3, 3, 1, 1)

    def forward(self, x):
        x_1 = self.PRelu(self.conv1(x))
        x_2 = self.PRelu(self.conv2(x_1))
        x = self.PRelu(self.conv3(x_2))
        x_4 = self.PRelu(self.conv4(x))
        x = self.PRelu(self.conv5(x_4) + x_2)
        x = self.PRelu(self.conv6(x))
        x = self.PRelu(self.conv7(x) + x_4)
        x = self.PRelu(self.conv8(x) + self.conv1_1(x_1))
        x = self.PRelu(self.conv9(x))
        x = self.PS(x)
        x = self.convf(x)
        return x

# class ChannelAttention(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)

# class RCAB(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(RCAB, self).__init__()
#         self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
#         self.ca = ChannelAttention(channel, reduction)

#     def forward(self, x):
#         res = self.conv1(x)
#         res = self.relu(res)
#         res = self.conv2(res)
#         res = self.ca(res)
#         return x + res

# class RIR(nn.Module):
#     def __init__(self, channel, num_rcab=6):
#         super(RIR, self).__init__()
#         self.rcabs = nn.Sequential(*[RCAB(channel) for _ in range(num_rcab)])

#     def forward(self, x):
#         return x + self.rcabs(x)

# class SuperResolution(nn.Module):
#     def __init__(self):
#         super(SuperResolution, self).__init__()
#         self.conv1 = nn.Conv2d(3, 18, kernel_size=5, stride=1, padding=2)
        
#         self.conv2 = nn.Conv2d(18, 27, kernel_size=1, stride=1, padding=0)
        
#         self.rir = RIR(27)
        
#         self.conv3 = nn.Conv2d(27, 18, kernel_size=1, stride=1, padding=0)
        
#         self.PS = nn.PixelShuffle(3)
#         self.prelu = nn.PReLU()
        
#         self.final_conv = nn.Conv2d(2, 3, kernel_size=3, stride=1, padding=1)

#     def forward(self, x):
#         # Initial feature extraction
#         x1 = self.prelu(self.conv1(x))
#         x = self.prelu(self.conv2(x1))
#         # First RIR block
#         x = self.rir(x)
#         x = self.prelu(self.conv3(x))

#         x = x + x1

#         x = self.PS(x)
        
#         # Final convolution for channel adjustment
#         x = self.final_conv(x)
        
#         return x
    