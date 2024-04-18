import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Identity mapping, skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.skip(x)
        out = torch.relu(out)
        return out


class MyResNet(nn.Module):
    def __init__(self):
        super(MyResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # Followed by a number of ResBlocks
        # Assume we are not reducing size in these blocks
        self.res1 = ResBlock(64, 64)
        self.res2 = ResBlock(64, 64)
        self.res3 = ResBlock(64, 64)
        self.res4 = ResBlock(64, 64)
        # Final output should have 1 channel
        self.conv2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.conv2(out)
        return out

'''
# To create the model
model = MyResNet()
# To test the model:
x = torch.randn(1, 1, 512,512)  # random input
y = model(x)
print(y.shape)  # Should print: torch.Size([1, 1, 128, 128])'''