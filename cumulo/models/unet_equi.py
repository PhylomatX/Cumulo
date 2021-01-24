import torch
import torch.nn as nn
import e2cnn.nn as enn
from e2cnn import gspaces


def ConvConv(in_type, out_type, bn_momentum=0.1):
    conv_conv = enn.SequentialModule(
        enn.R2Conv(in_type, out_type, kernel_size=3, padding=1, stride=1),
        enn.InnerBatchNorm(out_type, momentum=bn_momentum),
        enn.ReLU(out_type, inplace=True),
        enn.R2Conv(out_type, out_type, kernel_size=3, padding=1, stride=1),
        enn.InnerBatchNorm(out_type, momentum=bn_momentum),
        enn.ReLU(out_type, inplace=True)
    )
    return conv_conv


def DownConv(in_type, out_type, bn_momentum=0.1):
    down_conv = enn.SequentialModule(
        ConvConv(in_type, out_type, bn_momentum),
        enn.PointwiseMaxPoolAntialiased(out_type, kernel_size=2, stride=2)
    )
    return down_conv


def UpConv(in_type, out_type, bn_momentum=0.1):
    up_conv = enn.SequentialModule(
        enn.R2Upsampling(in_type, scale_factor=2),
        ConvConv(in_type, out_type, bn_momentum)
    )
    return up_conv


class UpConvConcat(nn.Module):
    """ (conv => ReLU) * 2 => maxpool """

    def __init__(self, in_type, out_type, bn_momentum=0.1):
        super(UpConvConcat, self).__init__()
        self.upconv = enn.R2ConvTransposed(in_type, out_type, kernel_size=2, stride=2)
        self.conv = ConvConv(in_type, out_type, bn_momentum)

    def forward(self, x1, x2):
        x1 = self.upconv(x1).tensor
        x1_dim = x1.size()[2]
        x2 = extract_img(x1_dim, x2)
        concat = torch.cat((x1, x2), dim=1)
        concat = enn.GeometricTensor(concat, self.in_type)
        return self.conv(concat)


def extract_img(size, in_tensor):
    """
    Args:
        size (int): size of crop
        in_tensor (tensor): tensor to be cropped
    """
    dim1, dim2 = in_tensor.size()[2:]
    in_tensor = in_tensor[:, :, int((dim1-size)/2):int((dim1+size)/2), int((dim2-size)/2):int((dim2+size)/2)]
    return in_tensor


class UNet_equi(nn.Module):

    def __init__(self, in_channels, out_channels, starting_filters=32, bn_momentum=0.1):
        super(UNet_equi, self).__init__()

        self.r2_act = gspaces.Rot2dOnR2(N=8)

        self.in_channels = enn.FieldType(self.r2_act, in_channels * [self.r2_act.trivial_repr])
        self.filters = enn.FieldType(self.r2_act, starting_filters * [self.r2_act.regular_repr])
        self.filters2 = enn.FieldType(self.r2_act, 2 * starting_filters * [self.r2_act.regular_repr])
        self.filters4 = enn.FieldType(self.r2_act, 4 * starting_filters * [self.r2_act.regular_repr])
        self.filters8 = enn.FieldType(self.r2_act, 8 * starting_filters * [self.r2_act.regular_repr])
        self.out_channels = enn.FieldType(self.r2_act, out_channels * [self.r2_act.regular_repr])

        self.conv1 = DownConv(self.in_channels, self.filters, bn_momentum)
        self.conv2 = DownConv(self.filters, self.filters2, bn_momentum)
        self.conv3 = DownConv(self.filters2, self.filters4, bn_momentum)
        self.conv4 = ConvConv(self.filters4, self.filters8, bn_momentum)

        self.upconv1 = UpConv(self.filters8, self.filters4, bn_momentum)
        self.upconv2 = UpConv(self.filters4, self.filters2, bn_momentum)
        self.upconv3 = UpConv(self.filters2, self.filters, bn_momentum)

        # self.upconv1 = UpconvConcat(self.filters8, self.filters4, bn_momentum)
        # self.upconv2 = UpconvConcat(self.filters4, self.filters2, bn_momentum)
        # self.upconv3 = UpconvConcat(self.filters2, self.filters, bn_momentum)
        self.conv_out = enn.R2Conv(self.filters, self.out_channels, kernel_size=1, padding=0, stride=1)

        self.gpool = enn.GroupPooling(self.out_channels)

    def forward(self, x):
        x = enn.GeometricTensor(x, self.in_channels)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.upconv1(x4)
        x6 = self.upconv2(x5)
        x7 = self.upconv3(x6)
        x8 = self.conv_out(x7)
        out = self.gpool(x8)
        return out.tensor
