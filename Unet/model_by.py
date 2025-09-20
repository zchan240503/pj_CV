import torch
import torch.nn as nn
import torch.nn.functional as F
class DownsampleBlock(nn.Module):
  def __init__(self, in_channels, out_channels, is_first = False):
    super().__init__()
    layers = []
    if not is_first:
      layers.append(nn.MaxPool2d(2))
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
    layers.append(nn.ReLU(inplace = True))
    layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
    layers.append(nn.ReLU(inplace = True))
    self.block = nn.Sequential(*layers)
  def forward(self, x):
    return self.block(x)
class UpsampleBlock(nn.Module):
  def __init__(self, in_channels, out_channels, is_last = False):
    super().__init__()
    self.is_last = is_last
    self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 2, stride = 2)
    self.conv1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    if self.is_last:
      self.conv3 = nn.Conv2d(out_channels,1, kernel_size=3)
  def forward(self, x, skip):
    x = self.up(x)
    diffY = skip.size()[2] - x.size()[2]
    diffX = skip.size()[3] - x.size()[3]
    skip = skip[:, :,
                diffY // 2 : skip.size()[2] - (diffY - diffY//2),
                diffX // 2 : skip.size()[3] - (diffX - diffX //2)]

    x = torch.cat([x, skip], dim = 1)
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    return x
class UNet(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        self.block1 = DownsampleBlock(3, 64, is_first=True)
        self.block2 = DownsampleBlock(64, 128)
        self.block3 = DownsampleBlock(128, 256)
        self.block4 = DownsampleBlock(256, 512)
        self.block5 = DownsampleBlock(512, 1024)

        self.block6 = UpsampleBlock(1024, 512)
        self.block7 = UpsampleBlock(512, 256)
        self.block8 = UpsampleBlock(256, 128)
        self.block9 = UpsampleBlock(128, 64)
        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        s1 = self.block1(x)
        s2 = self.block2(s1)
        s3 = self.block3(s2)
        s4 = self.block4(s3)
        b  = self.block5(s4)

        u1 = self.block6(b, s4)
        u2 = self.block7(u1, s3)
        u3 = self.block8(u2, s2)
        u4 = self.block9(u3, s1)
        out = self.out_conv(u4)
        return out
if __name__ == "__main__":
    model = UNet()
    X = torch.randn(1, 3, 572, 572)
    print(model(X).shape)