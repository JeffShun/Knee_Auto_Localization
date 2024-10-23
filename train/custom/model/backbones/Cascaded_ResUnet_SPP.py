import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



def make_res_layer(inplanes, planes, blocks, stride=1):
    downsample = nn.Sequential(
        conv1x1(inplanes, planes, stride),
        nn.BatchNorm3d(planes),
    )

    layers = []
    layers.append(BasicBlock(inplanes, planes, stride, downsample))
    for _ in range(1, blocks):
        layers.append(BasicBlock(planes, planes))

    return nn.Sequential(*layers)


class DoubleConv(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=int(kernel_size / 2)),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True), 
            nn.Conv3d(out_ch, out_ch, 3, padding=1, dilation=1),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)



class ResUnet_SPP(nn.Module):

    def __init__(self, channels=16, blocks=3):
        super(ResUnet_SPP, self).__init__()

        self.layer1 = make_res_layer(channels * 1, channels * 2, blocks, stride=2)
        self.layer2 = make_res_layer(channels * 2, channels * 4, blocks, stride=2)
        self.layer3 = make_res_layer(channels * 4, channels * 8, blocks, stride=2)
        self.layer4 = make_res_layer(channels * 8, channels * 16, blocks, stride=2)

        self.up5 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv5 = DoubleConv(channels * 24, channels * 8)
        self.up6 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv6 = DoubleConv(channels * 12, channels * 4)
        self.up7 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv7 = DoubleConv(channels * 6, channels * 2)
        self.up8 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv8 = DoubleConv(channels * 3, channels)


        self.spp_up_conv1 = nn.Sequential(
                    conv1x1(channels*2, channels),
                    nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
                    )
        self.spp_up_conv2 = nn.Sequential(
                    conv1x1(channels*4, channels),
                    nn.Upsample(scale_factor=4, mode='trilinear', align_corners=False)
                    )
        self.spp_up_conv3 = nn.Sequential(
                    conv1x1(channels*8, channels),
                    nn.Upsample(scale_factor=8, mode='trilinear', align_corners=False)
                    )

    def forward(self, inpt):
        c1 = inpt
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        up_5 = self.up5(c5)
        merge5 = torch.cat([up_5, c4], dim=1)
        c6 = self.conv5(merge5)
        up_6 = self.up6(c6)
        merge6 = torch.cat([up_6, c3], dim=1)
        c7 = self.conv6(merge6)
        up_7 = self.up7(c7)
        merge7 = torch.cat([up_7, c2], dim=1)
        c8 = self.conv7(merge7)
        up_8 = self.up8(c8)
        merge8 = torch.cat([up_8, c1], dim=1)
        c9 = self.conv8(merge8)

        out1 = c9
        out2 = self.spp_up_conv1(c8)
        out3 = self.spp_up_conv2(c7)
        out4 = self.spp_up_conv3(c6)
        
        out = out1+out2+out3+out4

        return out



class Cascaded_ResUnet_SPP(nn.Module):
    def __init__(self, in_ch, out_ch, channels=12, blocks=2):
        super(Cascaded_ResUnet_SPP, self).__init__()
        self.conv_in = DoubleConv(in_ch, channels, stride=2, kernel_size=3)
        self.Unet1 = ResUnet_SPP(channels, blocks)
        self.Unet2 = ResUnet_SPP(channels, blocks)
        self.conv_out1 = nn.Sequential(
            DoubleConv(channels,channels),
            DoubleConv(channels,channels),
            nn.Conv3d(channels, out_ch, kernel_size=1)
            )
        self.conv_out2 = nn.Sequential(
            DoubleConv(channels,channels),
            DoubleConv(channels,channels),
            nn.Conv3d(channels, out_ch, kernel_size=1)
            )
    def forward(self, input):
        fea = self.conv_in(input)
        u1 = self.Unet1(fea)
        u2 = self.Unet2(u1)
        out1 = self.conv_out1(F.interpolate(u1, scale_factor=2, mode="trilinear"))
        out2 = self.conv_out2(F.interpolate(u2, scale_factor=2, mode="trilinear"))
        return out1, out2 



if __name__ == '__main__':
    model = Cascaded_ResUnet_SPP(1, 1)
    print(model)

