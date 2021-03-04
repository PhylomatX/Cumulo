import torch.nn as nn
import e2cnn.nn as enn
from e2cnn import gspaces


def ConvConv(in_type, out_type, bn_momentum=0.1, padding=0):
    conv_conv = enn.SequentialModule(
        enn.R2Conv(in_type, out_type, kernel_size=3, padding=padding),
        enn.InnerBatchNorm(out_type, momentum=bn_momentum, track_running_stats=False),
        enn.ReLU(out_type, inplace=True),
        enn.R2Conv(out_type, out_type, kernel_size=3, padding=padding),
        enn.InnerBatchNorm(out_type, momentum=bn_momentum, track_running_stats=False),
        enn.ReLU(out_type, inplace=True)
    )
    return conv_conv


class DownConv(nn.Module):

    def __init__(self, in_type, out_type, bn_momentum=0.1, padding=0):
        super(DownConv, self).__init__()
        self.conv = ConvConv(in_type, out_type, bn_momentum, padding=padding)
        self.pool = enn.PointwiseMaxPoolAntialiased(out_type, kernel_size=2, stride=2)

    def forward(self, X):
        X = self.conv(X)
        pool_X = self.pool(X)
        return pool_X, X


class UpConvConcat(nn.Module):

    def __init__(self, in_type, out_type, bn_momentum=0.1, padding=0):
        super(UpConvConcat, self).__init__()
        self.in_type = in_type
        self.upconv = enn.R2Upsampling(in_type, scale_factor=2)
        self.conv1 = enn.R2Conv(in_type, out_type, kernel_size=1, padding=padding)
        self.conv2 = ConvConv(in_type, out_type, bn_momentum, padding=padding)

    def forward(self, x1, x2):
        x1 = self.upconv(x1)
        x1 = self.conv1(x1)
        x1_dim = x1.size()[2]
        x2 = extract_img(x1_dim, x2)
        concat = enn.tensor_directsum([x1, x2])
        return self.conv2(concat)


def extract_img(size, in_tensor):
    dim1, dim2 = in_tensor.size()[2:]
    in_tensor = in_tensor[:, :, int((dim1-size)/2):int((dim1+size)/2), int((dim2-size)/2):int((dim2+size)/2)]
    return in_tensor


class UNet_equi(nn.Module):

    def __init__(self, in_channels, out_channels, starting_filters=32, bn_momentum=0.1, rot=2, padding=0):
        super(UNet_equi, self).__init__()

        self.r2_act = gspaces.Rot2dOnR2(N=rot)
        print(f"Using {rot} rotations.")

        self.in_channels = enn.FieldType(self.r2_act, in_channels * [self.r2_act.trivial_repr])
        self.filters = enn.FieldType(self.r2_act, starting_filters * [self.r2_act.regular_repr])
        self.filters2 = enn.FieldType(self.r2_act, 2 * starting_filters * [self.r2_act.regular_repr])
        self.filters4 = enn.FieldType(self.r2_act, 4 * starting_filters * [self.r2_act.regular_repr])
        self.filters8 = enn.FieldType(self.r2_act, 8 * starting_filters * [self.r2_act.regular_repr])
        self.out_channels = enn.FieldType(self.r2_act, out_channels * [self.r2_act.regular_repr])

        self.conv1 = DownConv(self.in_channels, self.filters, bn_momentum, padding=padding)
        self.conv2 = DownConv(self.filters, self.filters2, bn_momentum, padding=padding)
        self.conv3 = DownConv(self.filters2, self.filters4, bn_momentum, padding=padding)

        self.conv4 = ConvConv(self.filters4, self.filters8, bn_momentum, padding=padding)

        self.upconv1 = UpConvConcat(self.filters8, self.filters4, bn_momentum, padding=padding)
        self.upconv2 = UpConvConcat(self.filters4, self.filters2, bn_momentum, padding=padding)
        self.upconv3 = UpConvConcat(self.filters2, self.filters, bn_momentum, padding=padding)

        self.conv_out = enn.R2Conv(self.filters, self.out_channels, kernel_size=1)
        self.gpool = enn.GroupPooling(self.out_channels)

    def forward(self, X):
        X = enn.GeometricTensor(X, self.in_channels)
        X, conv1 = self.conv1(X)
        X, conv2 = self.conv2(X)
        X, conv3 = self.conv3(X)
        X = self.conv4(X)
        X = self.upconv1(X, conv3)
        X = self.upconv2(X, conv2)
        X = self.upconv3(X, conv1)
        X = self.conv_out(X)
        out = self.gpool(X)

        # 256 => 164 (offset: 46)

        return out.tensor
