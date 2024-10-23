import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

# 损失函数，必须以字典的形式返回
class MixLoss(nn.Module):
    def __init__(self, point_radiu):
        super(MixLoss, self).__init__()
        self.point_radiu = point_radiu

    # def forward(self, inputs, targets):
    #     # 2 scale 监督
    #     kp_area_loss = []
    #     direction_loss = []
    #     # dense_direction_loss = []
    #     for i, point_radiu in enumerate(self.point_radiu):
    #         n_scale = len(self.point_radiu)
    #         targets_dilate = F.max_pool3d(targets, kernel_size=point_radiu*2+1, stride=1, padding=point_radiu)
    #         w = (i+1) / ((1 + n_scale) * n_scale / 2)
    #         kp_area_loss.append(w * self.kp_area_loss(inputs[i], targets_dilate))
    #         direction_loss.append(w * self.direction_loss(inputs[i],targets))
    #         # dense_direction_loss.append(w * self.dense_direction_loss(inputs[i],targets)) 
    #     return {"kp_area_loss": sum(kp_area_loss) ,
    #             "direction_loss": 0.1*sum(direction_loss)}      
        
    def forward(self, inputs, targets):
        out_heatmap, out_regression = inputs
        targets_dilate = F.max_pool3d(targets, kernel_size=self.point_radiu*2+1, stride=1, padding=self.point_radiu)
        kp_area_loss = self.kp_area_loss(out_heatmap, targets_dilate)
        direction_loss = self.direction_loss(out_heatmap, out_regression, targets)
        regression_loss = self.regression_loss(out_heatmap, out_regression, targets)
        # kp_gaussian_loss = self.kp_gaussian_loss(inputs, targets)
        return {"kp_area_loss": kp_area_loss,
                "direction_loss": 0.1*direction_loss,
                "regression_loss": regression_loss}
                # "kp_gaussian_loss":kp_gaussian_loss}   

    def kp_area_loss(self,inputs, targets):
        input_flatten = torch.flatten(inputs, start_dim=2, end_dim=-1)
        target_flatten = torch.flatten(targets, start_dim=2, end_dim=-1)
        input_flatten_softmax = F.softmax(input_flatten, -1)
        p = (input_flatten_softmax * target_flatten).sum(dim=2)
        loss = -((1-p)**2)*torch.log(p+1e-24)
        return loss.mean()         
    
    def regression_loss(self, out_heatmap, out_regression, targets):
        # 计算坐标回归损失
        points_coord_inputs = self._argmax(out_heatmap, out_regression, soft=True)
        points_coord_targets = self._argmax(targets, soft=False)
        loss = torch.sqrt(((points_coord_inputs - points_coord_targets)**2).sum(-1))
        return loss.mean()
    
    def direction_loss(self, out_heatmap, out_regression, targets):
        loss = []
        points_coord_inputs = self._argmax(out_heatmap, out_regression, soft=True)
        points_coord_targets = self._argmax(targets, soft=False)

        # 计算冠状面法向量方向损失
        phy_axial_normal = torch.tensor([1,0,0], dtype=torch.float32, device=out_heatmap.device)[None]
        vector12_inputs = points_coord_inputs[:,1,:]-points_coord_inputs[:,0,:]
        cor_norm_inputs = torch.cross(vector12_inputs, phy_axial_normal, dim=-1)
        vector12_targets = points_coord_targets[:,1,:]-points_coord_targets[:,0,:]
        cor_norm_targets = torch.cross(vector12_targets, phy_axial_normal, dim=-1)
        cor_norm_loss = 1-F.cosine_similarity(cor_norm_inputs, cor_norm_targets)
        loss.append(cor_norm_loss)

        # 计算横断面法向量方向损失
        vector45_inputs = points_coord_inputs[:,4,:]-points_coord_inputs[:,3,:]
        axial_norm_inputs = torch.cross(vector45_inputs, cor_norm_inputs, dim=-1)
        vector45_targets = points_coord_targets[:,4,:]-points_coord_targets[:,3,:]
        axial_norm_targets = torch.cross(vector45_targets, cor_norm_targets, dim=-1)
        axial_norm_loss = 1 - F.cosine_similarity(axial_norm_inputs, axial_norm_targets)
        loss.append(axial_norm_loss)

        # 计算矢状面法向量方向损失
        sag_norm_inputs = vector12_inputs
        sag_norm_targets = vector12_targets
        sag_norm_loss = 1-F.cosine_similarity(sag_norm_inputs, sag_norm_targets)
        loss.append(sag_norm_loss)

        # # point1,2,3共同一物理横断面，即z坐标相等
        # diff123_loss = (points_coord_inputs[:,1,0] - points_coord_inputs[:,0,0])**2 + (points_coord_inputs[:,2,0] - points_coord_inputs[:,0,0])**2
        # loss.append(diff123_loss)

        # # vector45与正冠状面法向量垂直
        # vector_constraint_45 = (torch.sum(vector45_inputs * cor_norm_inputs, dim=-1))**2
        # loss.append(vector_constraint_45)

        return sum(loss).mean()
    
    def dense_direction_loss(self, inputs, targets):
        loss = []
        max_points_inputs = self._argmax(inputs, soft=True)
        max_points_targets = self._argmax(targets, soft=False)
        for i in range(max_points_inputs.shape[1]-1):
            for j in range(i, max_points_targets.shape[1]):
                norm_vector_inputs = F.normalize(max_points_inputs[:,i,:]-max_points_inputs[:,j,:], dim=-1)
                norm_vector_targets = F.normalize(max_points_targets[:,i,:]-max_points_targets[:,j,:],dim=-1)
                loss.append(1-F.cosine_similarity(norm_vector_inputs, norm_vector_targets))
        return sum(loss).mean()

    def _argmax(self, inputs, shift=None, soft=True):
        X, Y, Z = inputs.shape[2:]
        # 对输入进行softmax操作
        input_flatten = torch.flatten(inputs, start_dim=2, end_dim=-1)
        if soft:
            input_flatten = F.softmax(input_flatten, -1)

        grid_x = torch.arange(0, X, device=inputs.device)
        grid_y = torch.arange(0, Y, device=inputs.device)
        grid_z = torch.arange(0, Z, device=inputs.device)
    
        grid_x, grid_y, grid_z = torch.meshgrid(grid_x, grid_y, grid_z)
        grid_x = grid_x.unsqueeze(0).unsqueeze(1).unsqueeze(-1)
        grid_y = grid_y.unsqueeze(0).unsqueeze(1).unsqueeze(-1)
        grid_z = grid_z.unsqueeze(0).unsqueeze(1).unsqueeze(-1)
        grids = torch.cat((grid_x, grid_y, grid_z), -1)  # [1, 1, D, H, W, 3] 
    
        if shift is not None:
            grids = grids + shift
        grids = torch.flatten(grids, start_dim=2, end_dim=-2)

        position_x = torch.sum(grids[...,0] * input_flatten, dim=-1, keepdim=True)
        position_y = torch.sum(grids[...,1] * input_flatten, dim=-1, keepdim=True)
        position_z = torch.sum(grids[...,2] * input_flatten, dim=-1, keepdim=True)

        return torch.cat((position_x, position_y, position_z), dim=-1) 
        
    def _argmax_(self, inputs, soft=True):
        X, Y, Z = inputs.shape[2:]
        # 对输入进行softmax操作
        input_flatten = torch.flatten(inputs, start_dim=2, end_dim=-1)
        if soft:
            input_flatten = F.softmax(input_flatten, -1)

        grid_x = torch.arange(0, X, device=inputs.device)
        grid_y = torch.arange(0, Y, device=inputs.device)
        grid_z = torch.arange(0, Z, device=inputs.device) 

        # 生成位置索引的网格
        grid_x, grid_y, grid_z = torch.meshgrid(grid_x, grid_y, grid_z)

        grid_x = grid_x.reshape(1, 1, -1).float()
        grid_y = grid_y.reshape(1, 1, -1).float()
        grid_z = grid_z.reshape(1, 1, -1).float()

        position_x = torch.sum(grid_x * input_flatten, dim=-1, keepdim=True)
        position_y = torch.sum(grid_y * input_flatten, dim=-1, keepdim=True)
        position_z = torch.sum(grid_z * input_flatten, dim=-1, keepdim=True)

        return torch.cat((position_x, position_y, position_z), dim=-1)    

    # def kp_gaussian_loss(self,inputs, targets, sigma=2): 
    #     def get_gaussian_kernel(sigma, size):
    #         coords = torch.arange(size, dtype=torch.float32) - size // 2
    #         g = torch.exp(-coords**2 / (2 * sigma**2))
    #         g /= g.sum()
    #         return g

    #     kernel_size = int(2 * (sigma * 3) + 1)  # 根据 sigma 动态计算核大小
    #     gaussian_kernel = get_gaussian_kernel(sigma, kernel_size)
    #     gaussian_kernel_3d = gaussian_kernel[:, None, None] * gaussian_kernel[None, :, None] * gaussian_kernel[None, None, :]
    #     gaussian_kernel_3d = gaussian_kernel_3d.expand(targets.shape[1], 1, -1, -1, -1).to(targets.device)
    #     with torch.no_grad():
    #         smoothed_targets = F.conv3d(targets, gaussian_kernel_3d, padding=kernel_size//2, groups=targets.shape[1])
    #         target_flatten = torch.flatten(smoothed_targets, start_dim=2, end_dim=-1)
    #         target_flatten = target_flatten / torch.max(target_flatten, dim=-1, keepdim=True)[0]
    #     input_flatten = torch.flatten(inputs, start_dim=2, end_dim=-1)
    #     loss = (input_flatten - target_flatten)**2
    #     return loss.mean()    
    
    # def direction_loss(self, inputs, targets):
    #     self.mse = nn.MSELoss(reduce=True)
    #     loss = []
    #     points_coord_inputs = self._argmax(inputs, soft=True)
    #     points_coord_targets = self._argmax(targets, soft=False)

    #     # 计算冠状面法向量方向损失
    #     phy_axial_normal = torch.tensor([1,0,0], dtype=torch.float32, device=inputs.device)[None]
    #     vector12_inputs = points_coord_inputs[:,1,:]-points_coord_inputs[:,0,:]
    #     cor_norm_inputs = torch.cross(vector12_inputs, phy_axial_normal, dim=-1)
    #     vector12_targets = points_coord_targets[:,1,:]-points_coord_targets[:,0,:]
    #     cor_norm_targets = torch.cross(vector12_targets, phy_axial_normal, dim=-1)
    #     cor_norm_loss = self.mse(cor_norm_inputs, cor_norm_targets)
    #     loss.append(cor_norm_loss)

    #     # 计算横断面法向量方向损失
    #     vector45_inputs = points_coord_inputs[:,4,:]-points_coord_inputs[:,3,:]
    #     axial_norm_inputs = torch.cross(vector45_inputs, cor_norm_inputs, dim=-1)
    #     vector45_targets = points_coord_targets[:,4,:]-points_coord_targets[:,3,:]
    #     axial_norm_targets = torch.cross(vector45_targets, cor_norm_targets, dim=-1)
    #     axial_norm_loss = self.mse(axial_norm_inputs, axial_norm_targets)
    #     loss.append(axial_norm_loss)

    #     # 计算矢状面法向量方向损失
    #     sag_norm_inputs = vector12_inputs
    #     sag_norm_targets = vector12_targets
    #     sag_norm_loss = self.mse(sag_norm_inputs, sag_norm_targets)
    #     loss.append(sag_norm_loss)

    #     # # point1,2,3共同一物理横断面，即z坐标相等
    #     # diff123_loss = self.mse(points_coord_inputs[:,1,0], points_coord_inputs[:,0,0]) + self.mse(points_coord_inputs[:,2,0],  points_coord_inputs[:,0,0])
    #     # loss.append(diff123_loss)

    #     # # vector45与正冠状面法向量垂直
    #     # vector_constraint_45 = (torch.sum(vector45_inputs * cor_norm_inputs, dim=-1))**2
    #     # loss.append(vector_constraint_45)

    #     return sum(loss).mean()

    # def dense_direction_loss(self, inputs, targets):
    #     loss = []
    #     max_points_inputs = self._argmax(inputs, soft=True)
    #     max_points_targets = self._argmax(targets, soft=False)
    #     for i in range(max_points_inputs.shape[1]-1):
    #         for j in range(i, max_points_targets.shape[1]):
    #             norm_vector_inputs = max_points_inputs[:,i,:]-max_points_inputs[:,j,:]
    #             norm_vector_targets = max_points_targets[:,i,:]-max_points_targets[:,j,:]
    #             loss.append(self.mse(norm_vector_inputs, norm_vector_targets))
    #     return sum(loss).mean()

class ValidLoss(nn.Module):
    def __init__(self):
        super(ValidLoss, self).__init__()

    def forward(self, inputs, targets):
        errors = dict()
        pred_img = inputs[-1]
        tgt_img = targets
        origin = torch.tensor([-141.9, -154.4, -140], dtype=torch.float32, device=targets.device)[None,None]
        spacing = torch.tensor([3, 1.46, 1.46], dtype=torch.float32, device=targets.device)[None,None]
        pred_index = self.max_index(pred_img) * spacing + origin
        tgt_index = self.max_index(tgt_img) * spacing + origin

        kp_dist_error = torch.mean(torch.sqrt(torch.sum((pred_index - tgt_index)**2, -1)), 0)
        for i in range(kp_dist_error.shape[0]):
            errors[f"kp_dist_error_{i+1}"] = kp_dist_error[i]

        # 计算冠状面的法线向量error
        phy_axial_normal = torch.tensor([1,0,0], dtype=torch.float32, device=targets.device)[None]
        pred_v12 = F.normalize(pred_index[:,0] - pred_index[:,1], dim=-1)  
        pred_normal_coronal = F.normalize(torch.cross(phy_axial_normal, pred_v12), dim=-1)
        tgt_v12 = F.normalize(tgt_index[:,0] - tgt_index[:,1], dim=-1)  
        tgt_normal_coronal = F.normalize(torch.cross(phy_axial_normal, tgt_v12), dim=-1) 
        cor_error = self.angle_error(pred_normal_coronal, tgt_normal_coronal)
        errors[f"cor_error"] = torch.mean(cor_error, 0)

        # 计算横断面的法线向量error
        pred_v45 = F.normalize(pred_index[:,3] - pred_index[:,4], dim=-1)  
        pred_normal_axis = F.normalize(torch.cross(pred_v45, pred_normal_coronal), dim=-1)
        tgt_v45 = F.normalize(tgt_index[:,3] - tgt_index[:,4], dim=-1)  
        tgt_normal_axis = F.normalize(torch.cross(tgt_v45, tgt_normal_coronal), dim=-1)        
        axis_error = self.angle_error(pred_normal_axis, tgt_normal_axis)
        errors[f"axis_error"] = torch.mean(axis_error, 0)

        # 计算矢状面的法线向量error
        pred_normal_sag = pred_v12
        tgt_normal_sag = tgt_v12       
        sag_error = self.angle_error(pred_normal_sag, tgt_normal_sag)
        errors[f"sag_error"] = torch.mean(sag_error, 0)

        return errors 
    

    def max_index(self, img):
        b, c, d, h, w = img.shape
        _, max_indices = torch.max(img.view(b, c, -1), dim=-1)
        max_indices_3d = np.array([np.unravel_index(idx, (d, h, w)) for idx in max_indices.view(-1).tolist()])
        max_indices_3d = torch.from_numpy(max_indices_3d.reshape(b, c, 3)).to(img.device)
        return max_indices_3d
    
    def angle_error(self, vec1 , vec2 ):
        # 将两个向量调整为 (b, c, 1) 和 (b, 1, c)
        vec1_expanded = vec1.unsqueeze(1)  # (b, 1, c)
        vec2_expanded = vec2.unsqueeze(2)  # (b, c, 1)
        product = torch.bmm(vec1_expanded, vec2_expanded)
        error = 180*torch.arccos(product)/torch.pi
        return error
