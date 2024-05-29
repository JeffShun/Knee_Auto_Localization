import torch
import torch.nn as nn
from torch.nn import functional as F

# 损失函数，必须以字典的形式返回
class MixLoss(nn.Module):
    def __init__(self, point_radius):
        super(MixLoss, self).__init__()
        self.point_radius = point_radius

    def forward(self, inputs, targets):
        # 2 scale 监督
        kp_area_loss = []
        direction_loss = []
        for i, point_radiu in enumerate(self.point_radius):
            n_scale = len(self.point_radius)
            targets_dilate = F.max_pool3d(targets, kernel_size=point_radiu*2+1, stride=1, padding=point_radiu)
            w = (i+1) / ((1 + n_scale) * n_scale / 2)
            kp_area_loss.append(w * self.kp_area_loss(inputs[i], targets_dilate))
            direction_loss.append(w * self.direction_loss(inputs[i],targets))
        return {"kp_area_loss": sum(kp_area_loss) ,
                "direction_loss":0.1*sum(direction_loss)}      

    def kp_area_loss(self,inputs, targets):
        input_flatten = torch.flatten(inputs, start_dim=2, end_dim=-1)
        target_flatten = torch.flatten(targets, start_dim=2, end_dim=-1)
        input_flatten_softmax = F.softmax(input_flatten, -1)
        p = (input_flatten_softmax * target_flatten).sum(dim=2)
        loss = -((1-p)**2)*torch.log(p+1e-24)
        return loss.mean()        
    

    def direction_loss(self, inputs, targets):
        loss = []
        points_coord_inputs = self._argmax(inputs, soft=True)
        points_coord_targets = self._argmax(targets, soft=False)

        # 计算冠状面法向量方向损失
        phy_axial_normal = torch.tensor([1,0,0], dtype=torch.float32, device=inputs.device)[None]
        vector12_inputs = F.normalize(points_coord_inputs[:,1,:]-points_coord_inputs[:,0,:], dim=-1)
        cor_norm_inputs = F.normalize(torch.cross(vector12_inputs, phy_axial_normal, dim=-1), dim=-1)
        vector12_targets = F.normalize(points_coord_targets[:,1,:]-points_coord_targets[:,0,:], dim=-1)
        cor_norm_targets = F.normalize(torch.cross(vector12_targets, phy_axial_normal, dim=-1), dim=-1)
        cor_norm_loss = 1-F.cosine_similarity(cor_norm_inputs, cor_norm_targets)
        loss.append(cor_norm_loss)

        # 计算横断面法向量方向损失
        vector45_inputs = F.normalize(points_coord_inputs[:,4,:]-points_coord_inputs[:,3,:], dim=-1)
        axial_norm_inputs = F.normalize(torch.cross(vector45_inputs, cor_norm_inputs, dim=-1), dim=-1)
        vector45_targets = F.normalize(points_coord_targets[:,4,:]-points_coord_targets[:,3,:], dim=-1)
        axial_norm_targets = F.normalize(torch.cross(vector45_targets, cor_norm_targets, dim=-1), dim=-1)
        axial_norm_loss = 1 - F.cosine_similarity(axial_norm_inputs, axial_norm_targets)
        loss.append(axial_norm_loss)

        # 计算矢状面法向量方向损失
        sag_norm_inputs = vector12_inputs
        sag_norm_targets = vector12_targets
        sag_norm_loss = 1-F.cosine_similarity(sag_norm_inputs, sag_norm_targets)
        loss.append(sag_norm_loss)

        # point1,2,3共同一物理横断面，即z坐标相等
        diff123_loss = (points_coord_inputs[:,1,0] - points_coord_inputs[:,0,0])**2 + (points_coord_inputs[:,2,0] - points_coord_inputs[:,0,0])**2
        loss.append(diff123_loss)

        # vector45与正冠状面法向量垂直
        vector_constraint_45 = (torch.sum(vector45_inputs * cor_norm_inputs, dim=-1))**2
        loss.append(vector_constraint_45)

        return sum(loss).mean()
    

    def _argmax(self, inputs, soft=True):
        # 对输入进行softmax操作
        input_flatten = torch.flatten(inputs, start_dim=2, end_dim=-1)
        if soft:
            input_flatten = F.softmax(input_flatten, -1)

        grid_x = torch.linspace(-1, 1, inputs.size(2), device=inputs.device)
        grid_y = torch.linspace(-1, 1, inputs.size(3), device=inputs.device)
        grid_z = torch.linspace(-1, 1, inputs.size(4), device=inputs.device)

        # 生成位置索引的网格
        grid_x, grid_y, grid_z = torch.meshgrid(grid_x, grid_y, grid_z)

        grid_x = grid_x.reshape(1, 1, -1).float()
        grid_y = grid_y.reshape(1, 1, -1).float()
        grid_z = grid_z.reshape(1, 1, -1).float()

        position_x = torch.sum(grid_x * input_flatten, dim=-1, keepdim=True)
        position_y = torch.sum(grid_y * input_flatten, dim=-1, keepdim=True)
        position_z = torch.sum(grid_z * input_flatten, dim=-1, keepdim=True)

        return torch.cat((position_x, position_y, position_z), dim=-1)    