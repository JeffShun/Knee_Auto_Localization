import torch
import torch.nn as nn


def conv3x3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3x3 convolution with padding."""
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


def conv1x1x1(in_planes, out_planes, stride=1):
    """1x1x1 convolution."""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class DownConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownConv, self).__init__()
        self.conv = nn.Sequential(
            conv3x3x3(in_ch, out_ch, stride=2),
            nn.BatchNorm3d(out_ch), 
            nn.ReLU(inplace=True), 
            )
    def forward(self, input):
        return self.conv(input)


class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpConv, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(out_ch), 
            nn.ReLU(inplace=True), 
            )
    def forward(self, input):
        return self.conv(input)


class DoubleConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            conv3x3x3(in_ch, out_ch),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True), 
            conv3x3x3(out_ch, out_ch),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Cascaded_Unet_BiFPN(nn.Module):
    def __init__(self, in_ch, channels=12):
        super(Cascaded_Unet_BiFPN, self).__init__()
        self.conv_in = nn.Sequential(
            conv3x3x3(in_ch, channels, stride=2),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True)
            )

        # Unet1
        self.dl_u1_1 = DoubleConv(channels * 1, channels * 1)
        self.ds_u1_1 = DownConv(channels * 1, channels * 2)

        self.dl_u1_2 = DoubleConv(channels * 2, channels * 2)
        self.ds_u1_2 = DownConv(channels * 2, channels * 4)

        self.dl_u1_3 = DoubleConv(channels * 4, channels * 4)
        self.ds_u1_3 = DownConv(channels * 4, channels * 8)
        
        self.dl_u1_4 = DoubleConv(channels * 8, channels * 8)
        self.ds_u1_4 = DownConv(channels * 8, channels * 16)

        self.dl_u1_5 = DoubleConv(channels * 16, channels * 16)

        self.up_u1_4 = UpConv(channels * 16, channels * 8)
        self.s1_w4 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.s1_w4_relu = nn.ReLU()
        self.ul_u1_4 = DoubleConv(channels * 8, channels * 8)

        self.up_u1_3 = UpConv(channels * 8, channels * 4)
        self.s1_w3 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.s1_w3_relu = nn.ReLU()
        self.ul_u1_3 = DoubleConv(channels * 4, channels * 4)

        self.up_u1_2 = UpConv(channels * 4, channels * 2)
        self.s1_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.s1_w2_relu = nn.ReLU()
        self.ul_u1_2 = DoubleConv(channels * 2, channels * 2)

        self.up_u1_1 = UpConv(channels * 2, channels * 1)
        self.s1_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.s1_w1_relu = nn.ReLU()
        self.ul_u1_1 = DoubleConv(channels * 1, channels * 1)


        # Unet2
        self.dl_u2_1 = DoubleConv(channels * 1, channels * 1)
        self.ds_u2_1 = DownConv(channels * 1, channels * 2)

        self.s2_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.s2_w2_relu = nn.ReLU()
        self.dl_u2_2 = DoubleConv(channels * 2, channels * 2)
        self.ds_u2_2 = DownConv(channels * 2, channels * 4)

        self.s2_w3 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.s2_w3_relu = nn.ReLU()
        self.dl_u2_3 = DoubleConv(channels * 4, channels * 4)
        self.ds_u2_3 = DownConv(channels * 4, channels * 8)

        self.s2_w4 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.s2_w4_relu = nn.ReLU()        
        self.dl_u2_4 = DoubleConv(channels * 8, channels * 8)
        self.ds_u2_4 = DownConv(channels * 8, channels * 16)

        self.s2_w5 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.s2_w5_relu = nn.ReLU()   
        self.dl_u2_5 = DoubleConv(channels * 16, channels * 16)

        self.up_u2_4 = UpConv(channels * 16, channels * 8)
        self.s3_w4 = nn.Parameter(torch.ones(4, dtype=torch.float32), requires_grad=True)
        self.s3_w4_relu = nn.ReLU()
        self.ul_u2_4 = DoubleConv(channels * 8, channels * 8)

        self.up_u2_3 = UpConv(channels * 8, channels * 4)
        self.s3_w3 = nn.Parameter(torch.ones(4, dtype=torch.float32), requires_grad=True)
        self.s3_w3_relu = nn.ReLU()
        self.ul_u2_3 = DoubleConv(channels * 4, channels * 4)

        self.up_u2_2 = UpConv(channels * 4, channels * 2)
        self.s3_w2 = nn.Parameter(torch.ones(4, dtype=torch.float32), requires_grad=True)
        self.s3_w2_relu = nn.ReLU()
        self.ul_u2_2 = DoubleConv(channels * 2, channels * 2)

        self.up_u2_1 = UpConv(channels * 2, channels * 1)
        self.s3_w1 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.s3_w1_relu = nn.ReLU()
        self.ul_u2_1 = DoubleConv(channels * 1, channels * 1)
        

    def forward(self, input):
        epsilon = 1e-4
        fea_in = self.conv_in(input)

        # Unet1 encoder
        u1_dc1 = self.dl_u1_1(fea_in)
        u1_dc2 = self.dl_u1_2(self.ds_u1_1(u1_dc1))
        u1_dc3 = self.dl_u1_3(self.ds_u1_2(u1_dc2))
        u1_dc4 = self.dl_u1_4(self.ds_u1_3(u1_dc3))
        u1_dc5 = self.dl_u1_5(self.ds_u1_4(u1_dc4))

        # Unet2 decoder
        s1_w4 = self.s1_w4_relu(self.s1_w4)
        weight = s1_w4 / (torch.sum(s1_w4, dim=0) + epsilon)     
        u1_uc4 = self.ul_u1_4(weight[0] * u1_dc4 + weight[1] * self.up_u1_4(u1_dc5))

        s1_w3 = self.s1_w3_relu(self.s1_w3)
        weight = s1_w3 / (torch.sum(s1_w3, dim=0) + epsilon)        
        u1_uc3 = self.ul_u1_3(weight[0] * u1_dc3 + weight[1] * self.up_u1_3(u1_uc4))

        s1_w2 = self.s1_w2_relu(self.s1_w2)
        weight = s1_w2 / (torch.sum(s1_w2, dim=0) + epsilon)        
        u1_uc2 = self.ul_u1_2(weight[0] * u1_dc2 + weight[1] * self.up_u1_2(u1_uc3))

        s1_w1 = self.s1_w1_relu(self.s1_w1)
        weight = s1_w1 / (torch.sum(s1_w1, dim=0) + epsilon)        
        u1_uc1 = self.ul_u1_1(weight[0] * u1_dc1 + weight[1] * self.up_u1_1(u1_uc2))

        # Unet2 encoder
        u2_dc1 = self.dl_u2_1(u1_uc1)

        s2_w2 = self.s2_w2_relu(self.s2_w2)
        weight = s2_w2 / (torch.sum(s2_w2, dim=0) + epsilon)  
        u2_dc2 = self.dl_u2_2(weight[0] * u1_dc2 + weight[1] * u1_uc2 + weight[2] * self.ds_u2_1(u2_dc1))

        s2_w3 = self.s2_w3_relu(self.s2_w3)
        weight = s2_w3 / (torch.sum(s2_w3, dim=0) + epsilon)  
        u2_dc3 = self.dl_u2_3(weight[0] * u1_dc3 + weight[1] * u1_uc3 + weight[2] * self.ds_u2_2(u2_dc2))

        s2_w4 = self.s2_w4_relu(self.s2_w4)
        weight = s2_w4 / (torch.sum(s2_w4, dim=0) + epsilon)  
        u2_dc4 = self.dl_u2_4(weight[0] * u1_dc4 + weight[1] * u1_uc4 + weight[2] * self.ds_u2_3(u2_dc3))

        s2_w5 = self.s2_w5_relu(self.s2_w5)
        weight = s2_w5 / (torch.sum(s2_w5, dim=0) + epsilon)  
        u2_dc5 = self.dl_u2_5(weight[0] * u1_dc5 + weight[1] * self.ds_u2_4(u2_dc4))

        # Unet2 decoder
        s3_w4 = self.s3_w4_relu(self.s3_w4)
        weight = s3_w4 / (torch.sum(s3_w4, dim=0) + epsilon) 
        u2_uc4 = self.ul_u2_4(weight[0] * u1_dc4 + weight[1] * u1_uc4 + weight[2] * u2_dc4 + weight[3] * self.up_u2_4(u2_dc5))

        s3_w3 = self.s3_w3_relu(self.s3_w3)
        weight = s3_w3 / (torch.sum(s3_w3, dim=0) + epsilon) 
        u2_uc3 = self.ul_u2_3(weight[0] * u1_dc3 + weight[1] * u1_uc3 + weight[2] * u2_dc3 + weight[3] * self.up_u2_3(u2_uc4))

        s3_w2 = self.s3_w2_relu(self.s3_w2)
        weight = s3_w2 / (torch.sum(s3_w2, dim=0) + epsilon) 
        u2_uc2 = self.ul_u2_2(weight[0] * u1_dc2 + weight[1] * u1_uc2 + weight[2] * u2_dc2 + weight[3] * self.up_u2_2(u2_uc3))

        s3_w1 = self.s3_w1_relu(self.s3_w1)
        weight = s3_w1 / (torch.sum(s3_w1, dim=0) + epsilon) 
        u2_uc1 = self.ul_u2_1(weight[0] * u1_dc1 + weight[1] * u2_dc1 + weight[2] * self.up_u2_1(u2_uc2))


        return u1_uc1, u2_uc1 



if __name__ == "__main__":
    data = torch.rand(1, 1, 96, 192, 192)
    net = Cascaded_Unet_BiFPN(1, 16)
    c1, c2 = net(data)
    print(c1.shape, c2.shape)

