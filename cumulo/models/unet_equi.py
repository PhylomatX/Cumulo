import torch
import torch.nn as nn
import e2cnn.nn as enn
from e2cnn import gspaces


class ConvConv(enn.EquivariantModule):
    """ (conv => ReLU) * 2 => maxpool """

    def __init__(self, in_type, out_type, bn_momentum=0.1):
        super(ConvConv, self).__init__()
        self.conv = enn.SequentialModule(
            enn.R2Conv(in_type, out_type, kernel_size=3, padding=1, stride=1),
            enn.InnerBatchNorm(out_type, momentum=bn_momentum),
            enn.ReLU(out_type, inplace=True),
            enn.R2Conv(out_type, out_type, kernel_size=3, padding=1, stride=1),
            enn.InnerBatchNorm(out_type, momentum=bn_momentum),
            enn.ReLU(out_type, inplace=True)
        )

    def forward(self, X):
        X = self.conv(X)
        return X

    def evaluate_output_shape(self, input_shape):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape


class DownConv(enn.EquivariantModule):
    def __init__(self, in_type, out_type, bn_momentum=0.1):
        super(DownConv, self).__init__()
        self.conv = ConvConv(in_type, out_type, bn_momentum)
        self.pool = enn.PointwiseMaxPoolAntialiased(out_type, kernel_size=2, stride=2)

    def forward(self, X):
        X = self.conv(X)
        pool_X = self.pool(X)
        return pool_X, X

    def evaluate_output_shape(self, input_shape):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape


class UpconvConcat(nn.Module):
    """ (conv => ReLU) * 2 => maxpool """

    def __init__(self, in_type, out_type, bn_momentum=0.1):
        super(UpconvConcat, self).__init__()
        self.upconv = enn.R2ConvTransposed(in_type, out_type, kernel_size=2, stride=2)
        self.conv = ConvConv(in_type, out_type, bn_momentum)

    def forward(self, X1, X2):
        X1 = self.upconv(X1)
        X1_dim = X1.size()[2]
        X2 = extract_img(X1_dim, X2)
        X1 = torch.cat((X1, X2), dim=1)
        X1 = self.conv(X1)
        return X1


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
        self.upconv1 = UpconvConcat(self.filters8, self.filters4, bn_momentum)
        self.upconv2 = UpconvConcat(self.filters4, self.filters2, bn_momentum)
        self.upconv3 = UpconvConcat(self.filters2, self.filters, bn_momentum)
        self.conv_out = enn.R2Conv(self.filters, self.out_channels, kernel_size=1, padding=0, stride=1)

    def forward(self, X):
        X, conv1 = self.conv1(X)
        X, conv2 = self.conv2(X)
        X, conv3 = self.conv3(X)
        X = self.conv4(X)
        X = self.upconv1(X, conv3)
        X = self.upconv2(X, conv2)
        X = self.upconv3(X, conv1)
        X = self.conv_out(X)
        return X
