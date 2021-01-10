import torch
import e2cnn.nn as enn
from e2cnn import gspaces
from typing import Tuple


class Conv(enn.EquivariantModule):
    """(Convolution2d + BatchNormalization + ReLU) * 2"""

    def __init__(self, in_type, out_type):
        super(Conv, self).__init__()

        self.conv = enn.SequentialModule(
            enn.R2Conv(in_type, out_type, kernel_size=3, stride=1, padding=0),
            enn.InnerBatchNorm(out_type),
            enn.ReLU(out_type)
        )

    def forward(self, input):
        return self.conv(input)

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape


class DownSample(enn.EquivariantModule):
    """Downscale with maxpool, convolution afterwards"""

    def __init__(self, in_type, out_type):
        super(DownSample, self).__init__()

        self.pool = enn.PointwiseMaxPoolAntialiased(in_type, kernel_size=2, stride=2)
        self.conv = Conv(in_type, out_type)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape


class UpSample(torch.nn.Module):
    """cropping -> concatenation -> convolution -> upscaling -> convolution"""

    def __init__(self, in_type, out_type, mid_type, padding=0):
        super(UpSample, self).__init__()
        self.mid_type = mid_type
        self.upsample = enn.R2Upsampling(mid_type, 2)
        self.conv1 = Conv(in_type, mid_type)
        self.conv2 = enn.R2Conv(mid_type, out_type, kernel_size=1)
        self.conv3 = Conv(out_type, out_type)

    def forward(self, x1, x2):
        x1 = x1.tensor
        crop1 = self.center_crop(x1, x2.shape[2:])
        crop1 = enn.GeometricTensor(crop1, self.mid_type)
        concat = enn.tensor_directsum([crop1, x2])
        out = self.conv1(concat)
        out = self.upsample(out)
        out = self.conv2(out)
        out = self.conv3(out)
        return out

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape

    def center_crop(self, layer, target_size):
        """cropping function to get the same size for concatenation"""
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
               :, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])
               ]


class LastConcat(torch.nn.Module):
    """cropping -> concatenation -> double convolution
    since we dont upsample anymore, an extra class must be created"""

    def __init__(self, in_type, out_type, mid_type, padding=0):
        super(LastConcat, self).__init__()
        self.mid_type = mid_type
        self.conv1 = Conv(in_type, mid_type)
        self.conv2 = enn.R2Conv(mid_type, out_type, kernel_size=1, stride=1, padding=0)

    def forward(self, x1, x2):
        x1 = x1.tensor
        crop1 = self.center_crop(x1, x2.shape[2:])
        crop1 = enn.GeometricTensor(crop1, self.mid_type)
        concat = enn.tensor_directsum([crop1, x2])
        out = self.conv1(concat)
        out = self.conv2(out)

        return out

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
               :, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])
               ]


class UNet(torch.nn.Module):

    def __init__(self):
        super(UNet, self).__init__()

        self.r2_act = gspaces.Rot2dOnR2(N=8)

        self.field_type_1 = enn.FieldType(self.r2_act, 1 * [self.r2_act.regular_repr])
        self.field_type_3 = enn.FieldType(self.r2_act, 3 * [self.r2_act.trivial_repr])
        self.field_type_8 = enn.FieldType(self.r2_act, 8 * [self.r2_act.regular_repr])
        self.field_type_16 = enn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr])
        self.field_type_32 = enn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr])
        self.field_type_64 = enn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr])
        self.field_type_128 = enn.FieldType(self.r2_act, 128 * [self.r2_act.regular_repr])

        self.conv1 = Conv(in_type=self.field_type_3, out_type=self.field_type_8)
        self.conv2 = Conv(in_type=self.field_type_8, out_type=self.field_type_8)

        self.down1 = DownSample(in_type=self.field_type_8, out_type=self.field_type_16)
        self.conv12 = Conv(in_type=self.field_type_16, out_type=self.field_type_16)
        self.down2 = DownSample(in_type=self.field_type_16, out_type=self.field_type_32)
        self.conv22 = Conv(in_type=self.field_type_32, out_type=self.field_type_32)
        self.down3 = DownSample(in_type=self.field_type_32, out_type=self.field_type_32)
        self.conv32 = Conv(in_type=self.field_type_32, out_type=self.field_type_32)

        self.up41 = UpSample(in_type=self.field_type_64, out_type=self.field_type_32, mid_type=self.field_type_32)
        self.up31 = UpSample(in_type=self.field_type_64, out_type=self.field_type_16, mid_type=self.field_type_32)
        self.up21 = UpSample(in_type=self.field_type_32, out_type=self.field_type_8, mid_type=self.field_type_16)
        self.up11 = LastConcat(in_type=self.field_type_16, out_type=self.field_type_1, mid_type=self.field_type_8)

        self.gpool1 = enn.GroupPooling(self.field_type_1)

    def forward(self, x):
        x = x.reshape((-1, x.shape[1], x.shape[2], x.shape[3]))
        x = enn.GeometricTensor(x, self.field_type_3)

        x1 = self.conv1(x)
        x2 = self.conv2(x1)

        x2 = self.down1(x2)
        x3 = self.conv12(x2)

        x4 = self.down2(x3)
        x5 = self.conv22(x4)

        x6 = self.down3(x5)
        x7 = self.conv32(x6)

        x8 = self.up41(x6, x7)
        x9 = self.up31(x4, x8)
        x10 = self.up21(x2, x9)
        out = self.up11(x1, x10)
        out = self.gpool1(out)
        out = out.tensor

        return out
