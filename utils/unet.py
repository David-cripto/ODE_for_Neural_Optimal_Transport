import torch
import torch.nn as nn
import torch.nn.functional as F
import functools


class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)

    
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = ConcatConv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = ConcatConv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels) 

    def forward(self, t, x):
        out = self.conv1(t, x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.bn2(out)
        out = self.relu(out)
        return out


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
        self.maxpool = nn.MaxPool2d(2)
        self.doubleconv = DoubleConv(in_channels, out_channels)

    def forward(self, t, x):
        out = self.maxpool(x)
        out = self.doubleconv(t, out)
        return out


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, t, x1, x2):
        # x2 from skip
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(t, x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = ConcatConv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, t, x):
        return self.conv(t, x)

class UNet_func(nn.Module):
    def __init__(self, n_channels, n_classes, base_factor=32, bilinear=True):
        super(UNet_func, self).__init__()
        
        self.nfe = 0
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.base_factor = base_factor

        self.inc = DoubleConv(n_channels, base_factor)
        self.down1 = Down(base_factor, 2 * base_factor)
        self.down2 = Down(2 * base_factor, 4 * base_factor)
        self.down3 = Down(4 * base_factor, 8 * base_factor)
        factor = 2 if bilinear else 1
        self.down4 = Down(8 * base_factor, 16 * base_factor // factor)
        self.up1 = Up(16 * base_factor, 8 * base_factor // factor, bilinear)
        self.up2 = Up(8 * base_factor, 4 * base_factor // factor, bilinear)
        self.up3 = Up(4 * base_factor, 2 * base_factor // factor, bilinear)
        self.up4 = Up(2 * base_factor, base_factor, bilinear)
        self.outc = OutConv(base_factor, n_classes)

    def forward(self, t, x):
        self.nfe += 1
        x1 = self.inc(t, x)
        x2 = self.down1(t, x1)
        x3 = self.down2(t, x2)
        x4 = self.down3(t, x3)
        x5 = self.down4(t, x4)
        x = self.up1(t, x5, x4)
        x = self.up2(t, x, x3)
        x = self.up3(t, x, x2)
        x = self.up4(t, x, x1)
        logits = self.outc(t, x)
        return logits
    
    