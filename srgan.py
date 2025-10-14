import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x + residual


class SRGANGenerator(nn.Module):
    def __init__(self, num_residual_blocks=16):
        super(SRGANGenerator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 9, padding=4)
        self.residual_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_residual_blocks)])
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        # No upsampling since input is already upsampled
        self.conv3 = nn.Conv2d(64, 3, 9, padding=4)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        residual = x
        x = self.residual_blocks(x)
        x = self.conv2(x)
        x = x + residual
        x = self.conv3(x)
        return x
