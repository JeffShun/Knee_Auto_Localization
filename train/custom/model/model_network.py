import torch
import torch.nn as nn

class Model_Network(nn.Module):

    def __init__(
        self,
        backbone
        ):
        super(Model_Network, self).__init__()

        self.backbone = backbone
        self.initialize_weights()

    def forward(self, img):
        outs = self.backbone(img)
        return outs

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()



