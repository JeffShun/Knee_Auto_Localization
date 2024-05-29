import torch.nn as nn
import torch.nn.functional as F

class Model_Head(nn.Module):

    def __init__(
        self,
        in_channels: int,
        num_class: int
    ):
        super(Model_Head, self).__init__()
        # TODO: 定制Head模型
        self.conv1 = nn.Conv3d(in_channels, num_class, 1)
        self.conv2 = nn.Conv3d(in_channels, num_class, 1)

    def forward(self, inputs):
        # TODO: 定制forward网络
        input1, input2 = inputs
        out1 = self.conv1(F.interpolate(input1, scale_factor=2, mode="trilinear"))
        out2 = self.conv2(F.interpolate(input2, scale_factor=2, mode="trilinear"))
        return out1, out2