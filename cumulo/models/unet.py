"""
Adapted from https://github.com/LobellLab/weakly_supervised
"""

import torch
import torch.nn as nn


class ConvConv(nn.Module):

    def __init__(self, in_channels, out_channels, norm='bn', padding=0):
        super(ConvConv, self).__init__()

        if norm == 'bn':
            norms = [nn.BatchNorm2d(out_channels, momentum=0.01), nn.BatchNorm2d(out_channels, momentum=0.01)]
        elif norm == 'gn':
            norms = [nn.GroupNorm(int(0.5 * out_channels), out_channels), nn.GroupNorm(int(0.5 * out_channels), out_channels)]
        else:
            norms = [nn.Identity(), nn.Identity()]

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding),
            norms[0],
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding),
            norms[1],
            nn.ReLU(inplace=True)
        )

    def forward(self, X):
        X = self.conv(X)
        return X


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, norm='bn', padding=0):
        super(DownConv, self).__init__()
        self.conv = ConvConv(in_channels, out_channels, norm, padding=padding)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, X):
        X = self.conv(X)
        pool_X = self.pool(X)
        return pool_X, X


class UpconvConcat(nn.Module):

    def __init__(self, in_channels, out_channels, norm='bn', padding=0):
        super(UpconvConcat, self).__init__()
        self.upconv = nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = ConvConv(in_channels, out_channels, norm, padding=padding)

    def forward(self, X1, X2):
        X1 = self.upconv(X1)
        X1 = self.conv1(X1)
        X1_dim = X1.size()[2]
        X2 = extract_img(X1_dim, X2)
        X1 = torch.cat((X1, X2), dim=1)
        return self.conv2(X1)


def extract_img(size, in_tensor):
    dim1, dim2 = in_tensor.size()[2:]
    in_tensor = in_tensor[:, :, int((dim1-size)/2):int((dim1+size)/2), int((dim2-size)/2):int((dim2+size)/2)]
    return in_tensor


class UNet(nn.Module):

    def __init__(self, in_channels, out_channels, starting_filters=32, norm='bn', padding=0):
        super(UNet, self).__init__()
        self.conv1 = DownConv(in_channels, starting_filters, norm, padding=padding)
        self.conv2 = DownConv(starting_filters, starting_filters * 2, norm, padding=padding)
        self.conv3 = DownConv(starting_filters * 2, starting_filters * 4, norm, padding=padding)

        self.conv4 = ConvConv(starting_filters * 4, starting_filters * 8, norm, padding=padding)

        self.upconv1 = UpconvConcat(starting_filters * 8, starting_filters * 4, norm, padding=padding)
        self.upconv2 = UpconvConcat(starting_filters * 4, starting_filters * 2, norm, padding=padding)
        self.upconv3 = UpconvConcat(starting_filters * 2, starting_filters, norm, padding=padding)

        self.conv_out = nn.Conv2d(starting_filters, out_channels, kernel_size=1)

    def forward(self, X):
        X, conv1 = self.conv1(X)
        X, conv2 = self.conv2(X)
        X, conv3 = self.conv3(X)
        X = self.conv4(X)
        X = self.upconv1(X, conv3)
        X = self.upconv2(X, conv2)
        X = self.upconv3(X, conv1)
        X = self.conv_out(X)

        # 256 => 164 (offset: 46) with 3 layers
        # 256 => 216 (offset: 20) with 2 layers
        # 256 => 240 (offset: 8) with 1 layer

        return X
