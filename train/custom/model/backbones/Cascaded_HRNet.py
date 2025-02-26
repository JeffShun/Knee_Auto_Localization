import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, bn_momentum=0.1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes, momentum=bn_momentum)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class StageModule(nn.Module):
    def __init__(self, stage, output_branches, channels, bn_momentum):
        super(StageModule, self).__init__()
        self.stage = stage
        self.output_branches = output_branches

        self.branches = nn.ModuleList()
        for i in range(self.stage):
            w = channels * (2 ** i)
            branch = nn.Sequential(
                BasicBlock(w, w, bn_momentum=bn_momentum),
                BasicBlock(w, w, bn_momentum=bn_momentum),
                BasicBlock(w, w, bn_momentum=bn_momentum)
            )
            self.branches.append(branch)

        self.fuse_layers = nn.ModuleList()
        # for each output_branches (i.e. each branch in all cases but the very last one)
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.stage):  # for each branch
                if i == j:
                    self.fuse_layers[-1].append(nn.Sequential())  # Used in place of "None" because it is callable
                elif i < j:
                    self.fuse_layers[-1].append(nn.Sequential(
                        nn.Conv3d(channels * (2 ** j), channels * (2 ** i), kernel_size=1, stride=1, bias=False),
                        nn.BatchNorm3d(channels * (2 ** i), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Upsample(scale_factor=(2.0 ** (j - i)), mode='nearest'),
                    ))
                elif i > j:
                    ops = []
                    for k in range(i - j - 1):
                        ops.append(nn.Sequential(
                            nn.Conv3d(channels * (2 ** j), channels * (2 ** j), kernel_size=3, stride=2, padding=1,
                                      bias=False),
                            nn.BatchNorm3d(channels * (2 ** j), eps=1e-05, momentum=0.1, affine=True,
                                           track_running_stats=True),
                            nn.ReLU(inplace=True),
                        ))
                    ops.append(nn.Sequential(
                        nn.Conv3d(channels * (2 ** j), channels * (2 ** i), kernel_size=3, stride=2, padding=1,
                                  bias=False),
                        nn.BatchNorm3d(channels * (2 ** i), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    ))
                    self.fuse_layers[-1].append(nn.Sequential(*ops))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        assert len(self.branches) == len(x)

        x = [branch(b) for branch, b in zip(self.branches, x)]

        x_fused = []
        for i in range(len(self.fuse_layers)):
            for j in range(0, len(self.branches)):
                if j == 0:
                    x_fused.append(self.fuse_layers[i][0](x[0]))
                else:
                    x_fused[i] = x_fused[i] + self.fuse_layers[i][j](x[j])

        for i in range(len(x_fused)):
            x_fused[i] = self.relu(x_fused[i])

        return x_fused


class HRNet(nn.Module):
    def __init__(self, channels=32, bn_momentum=0.1):
        super(HRNet, self).__init__()

        # Stage 1 (layer1)      - First group of bottleneck (resnet) modules
        self.layer1 = nn.Sequential(
            BasicBlock(channels, channels),
            BasicBlock(channels, channels),
            BasicBlock(channels, channels)
            )

        # Fusion layer 1 (transition1)      - Creation of the first two branches (one full and one half resolution)
        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm3d(channels, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
                nn.Conv3d(channels, channels * (2 ** 1), kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm3d(channels * (2 ** 1), eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            )),
        ])


        # Stage 2 (stage2)      - Second module with 1 group of bottleneck (resnet) modules. This has 2 branches
        self.stage2 = nn.Sequential(
            StageModule(stage=2, output_branches=2, channels=channels, bn_momentum=bn_momentum),
            )

        # Fusion layer 2 (transition2)      - Creation of the third branch (1/4 resolution)
        self.transition2 = nn.ModuleList([
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
                nn.Conv3d(channels * (2 ** 1), channels * (2 ** 2), kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm3d(channels * (2 ** 2), eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            )),  # ToDo Why the new branch derives from the "upper" branch only?
        ])

        # Stage 3 (stage3)      - Third module with 4 groups of bottleneck (resnet) modules. This has 3 branches
        self.stage3 = nn.Sequential(
            StageModule(stage=3, output_branches=3, channels=channels, bn_momentum=bn_momentum),
            StageModule(stage=3, output_branches=3, channels=channels, bn_momentum=bn_momentum),
            StageModule(stage=3, output_branches=3, channels=channels, bn_momentum=bn_momentum),
            StageModule(stage=3, output_branches=3, channels=channels, bn_momentum=bn_momentum),
            )

        # Fusion layer 3 (transition3)      - Creation of the fourth branch (1/8 resolution)
        self.transition3 = nn.ModuleList([
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
                nn.Conv3d(channels * (2 ** 2), channels * (2 ** 3), kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm3d(channels * (2 ** 3), eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            )),  # ToDo Why the new branch derives from the "upper" branch only?
        ])


        # Stage 4 (stage4)      - Fourth module with 3 groups of bottleneck (resnet) modules. This has 4 branches
        self.stage4 = nn.Sequential(
            StageModule(stage=4, output_branches=4, channels=channels, bn_momentum=bn_momentum),
            StageModule(stage=4, output_branches=4, channels=channels, bn_momentum=bn_momentum),
            StageModule(stage=4, output_branches=4, channels=channels, bn_momentum=bn_momentum),
            )

        # Final layer (final_layer)
        self.final_layer = nn.Conv3d(channels * 15, channels, kernel_size=1, stride=1)

    def forward(self, x):

        x = self.layer1(x)
        x = [trans(x) for trans in self.transition1]  # Since now, x is a list (# == nof branches)

        x = self.stage2(x)
        # x = [trans(x[-1]) for trans in self.transition2]    # New branch derives from the "upper" branch only
        x = [
            self.transition2[0](x[0]),
            self.transition2[1](x[1]),
            self.transition2[2](x[-1])
            ]  # New branch derives from the "upper" branch only

        x = self.stage3(x)
        # x = [trans(x) for trans in self.transition3]    # New branch derives from the "upper" branch only
        x = [
            self.transition3[0](x[0]),
            self.transition3[1](x[1]),
            self.transition3[2](x[2]),
            self.transition3[3](x[-1])
            ]  # New branch derives from the "upper" branch only

        x = self.stage4(x)
        x1 = F.interpolate(x[1], scale_factor=2, mode='trilinear', align_corners=False)
        x2 = F.interpolate(x[2], scale_factor=4, mode='trilinear', align_corners=False)
        x3 = F.interpolate(x[3], scale_factor=8, mode='trilinear', align_corners=False)
        x = torch.cat([x[0], x1, x2, x3], 1)  
         
        x = self.final_layer(x)

        return x


class Cascaded_HRNet(nn.Module):
    def __init__(self, in_ch=1, channels=32, bn_momentum=0.1):
        super(Cascaded_HRNet, self).__init__()
        self.conv_in = nn.Sequential(
            nn.Conv3d(in_ch, channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(channels, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
            )
        self.Net1 = HRNet(channels, channels)
        self.Net2 = HRNet(channels, channels)

    def forward(self, input):
        fea = self.conv_in(input)
        out1 = self.Net1(fea)
        out2 = self.Net2(out1)
        return out1, out2



if __name__ == '__main__':
    model = Cascaded_HRNet(in_ch=1, channels=32, bn_momentum=0.1).cuda()
    x = torch.randn((1, 1, 96, 192, 192)).cuda()
    out1, out2 = model(x)
    print(out1.shape, out2.shape)

