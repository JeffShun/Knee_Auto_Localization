
import random
import torch
import numpy as np
import math
from torch.nn import functional as F

"""
数据预处理工具
1、所有数据预处理函数都包含两个输入: img 、mask
2、img、mask的输入维度为3维[C,H,W]，第一个维度是通道数
"""

class TransformCompose(object):

    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

# 数据预处理工具
""""
img.shape = [C,H,W,D], mask.shape = [C,H,W,D]
"""
class to_tensor(object):
    def __call__(self, img, mask):
        img_o = torch.from_numpy(img.astype("float32"))
        mask_o = torch.from_numpy(mask.astype("float32"))
        return img_o, mask_o

class normlize(object):
    def __init__(self, win_clip=None):
        self.win_clip = win_clip

    def __call__(self, img, mask):
        ori_shape = img.shape
        img_o = img.view(ori_shape[0], -1)
        if self.win_clip is not None:
            img_o = torch.clip(img_o, self.win_clip[0], self.win_clip[1])
        img_min = img_o.min(dim=-1,keepdim=True)[0]
        img_max = img_o.max(dim=-1,keepdim=True)[0]
        img_o = (img_o - img_min)/(img_max - img_min)
        img_o = img_o.view(ori_shape)
        mask_o = mask
        return img_o, mask_o

class random_flip(object):
    def __init__(self, axis=0, prob=0.5):
        assert isinstance(axis, int) and axis in [1,2,3]
        self.axis = axis
        self.prob = prob
    def __call__(self, img, mask):
        img_o, mask_o = img, mask
        if random.random() < self.prob:
            img_o = torch.flip(img, [self.axis])
            mask_o = torch.flip(mask, [self.axis])
        return img_o, mask_o

class random_contrast(object):
    def __init__(self, alpha_range=[0.8, 1.2], prob=0.5):
        self.alpha_range = alpha_range
        self.prob = prob
    def __call__(self, img, mask):
        img_o, mask_o = img, mask
        if random.random() < self.prob:
            alpha = random.uniform(self.alpha_range[0], self.alpha_range[1])
            mean_val = torch.mean(img, (1,2,3), keepdim=True)
            img_o = mean_val + alpha * (img - mean_val)
            img_o = torch.clip(img_o, 0.0, 1.0)
        return img_o, mask_o

class random_gamma_transform(object):
    """
    input must be normlized before gamma transform
    """
    def __init__(self, gamma_range=[0.8, 1.2], prob=0.5):
        self.gamma_range = gamma_range
        self.prob = prob
    def __call__(self, img, mask):
        img_o, mask_o = img, mask
        if random.random() < self.prob:
            gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
            img_o = img**gamma
        return img_o, mask_o

class random_apply_mosaic(object):
    def __init__(self, prob=0.2, mosaic_size=40, mosaic_num=4):
        self.prob = prob
        self.mosaic_size = mosaic_size
        self.mosaic_num = mosaic_num

    def __call__(self, img, mask):
        img_o, mask_o = img, mask
        if random.random() < self.prob:
            n_channel, depth, height, width = img.shape
            for i in range(self.mosaic_num):
                x = torch.randint(0, depth - self.mosaic_size, (1,))
                y = torch.randint(0, height - self.mosaic_size, (1,))
                z = torch.randint(0, width - self.mosaic_size, (1,))
                mosaic_block = np.random.rand()*torch.rand_like(img[:, x:x+self.mosaic_size, y:y+self.mosaic_size, z:z+self.mosaic_size])
                img_o[:, x:x+self.mosaic_size, y:y+self.mosaic_size, z:z+self.mosaic_size] = mosaic_block
        return img_o, mask_o

class random_add_gaussian_noise(object):
    def __init__(self, prob=0.2, mean=0, std=1):
        self.prob = prob
        self.mean = mean
        self.std = std

    def __call__(self, img, mask):
        img_o, mask_o = img, mask
        if random.random() < self.prob:
            noise = torch.randn_like(img) * self.std + self.mean
            noisy_image = img + noise
            img_o = torch.clip(noisy_image, 0 , 1)
        return img_o, mask_o

class random_rotate3d(object):
    def __init__(self,
                x_theta_range=[-180,180], 
                y_theta_range=[-180,180], 
                z_theta_range=[-180,180],
                prob=0.5, 
                ):
        self.prob = prob
        self.x_theta_range = x_theta_range
        self.y_theta_range = y_theta_range
        self.z_theta_range = z_theta_range

    def _rotate3d(self, data, angles=[0,0,0], itp_mode="bilinear"): 
        alpha, beta, gama = [(angle/180)*math.pi for angle in angles]
        transform_matrix = torch.tensor([
            [math.cos(beta)*math.cos(gama), math.sin(alpha)*math.sin(beta)*math.cos(gama)-math.sin(gama)*math.cos(alpha), math.sin(beta)*math.cos(alpha)*math.cos(gama)+math.sin(alpha)*math.sin(gama), 0],
            [math.cos(beta)*math.sin(gama), math.cos(alpha)*math.cos(gama)+math.sin(alpha)*math.sin(beta)*math.sin(gama), -math.sin(alpha)*math.cos(gama)+math.sin(gama)+math.sin(beta)*math.cos(alpha), 0],
            [-math.sin(beta), math.sin(alpha)*math.cos(beta),math.cos(alpha)*math.cos(beta), 0]
            ])
        # 旋转变换矩阵
        transform_matrix = transform_matrix.unsqueeze(0)
        # 为了防止形变，先将原图padding为正方体，变换完成后再切掉
        data = data.unsqueeze(0)
        data_size = data.shape[2:]
        pad_x = (max(data_size)-data_size[0])//2
        pad_y = (max(data_size)-data_size[1])//2
        pad_z = (max(data_size)-data_size[2])//2
        pad = [pad_z,pad_z,pad_y,pad_y,pad_x,pad_x]
        pad_data = F.pad(data, pad=pad, mode="constant",value=0).to(torch.float32)
        grid = F.affine_grid(transform_matrix, pad_data.shape)
        output = F.grid_sample(pad_data, grid, mode=itp_mode)
        output = output.squeeze(0)
        output = output[:,pad_x:output.shape[1]-pad_x, pad_y:output.shape[2]-pad_y, pad_z:output.shape[3]-pad_z]
        return output
    
    def __call__(self, img, mask):
        img_o, mask_o = img, mask
        if random.random() < self.prob:
            random_angle_x = random.uniform(self.x_theta_range[0], self.x_theta_range[1])
            random_angle_y = random.uniform(self.y_theta_range[0], self.y_theta_range[1])
            random_angle_z = random.uniform(self.z_theta_range[0], self.z_theta_range[1]) 
            img_o = self._rotate3d(img,angles=[random_angle_x,random_angle_y,random_angle_z],itp_mode="bilinear")
            mask_o = self._rotate3d(mask,angles=[random_angle_x,random_angle_y,random_angle_z],itp_mode="bilinear")
            # 如果关键点旋转到边界以外，则不做旋转直接返回
            if torch.any(torch.sum(mask_o, (1,2,3))==0):
                return img, mask
            # 插值之后，mask的一个点可能会变成很多点，需要处理一下
            _shape = mask_o.shape
            mask_flatten = mask_o.reshape(_shape[0], -1)
            mask_zero = torch.zeros_like(mask_flatten, dtype=torch.float32)
            mask_zero[(torch.arange(_shape[0]), mask_flatten.max(dim=1, keepdim=False)[1])]=1
            mask_o = mask_zero.reshape(_shape)
        return img_o, mask_o

class resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        img_o = torch.nn.functional.interpolate(img[None], size=self.size, mode="trilinear") 
        mask_o = torch.nn.functional.interpolate(mask[None], size=self.size, mode="trilinear")
        img_o = img_o.squeeze(0)
        mask_o = mask_o.squeeze(0)
        # 插值之后，mask的一个点可能会变成很多点，需要处理一下
        _shape = mask_o.shape
        mask_flatten = mask_o.view(_shape[0], -1)
        mask_zero = torch.zeros_like(mask_flatten, dtype=torch.float32)
        mask_zero[(torch.arange(_shape[0]), mask_flatten.max(dim=1, keepdim=False)[1])]=1
        mask_o = mask_zero.view(_shape)
        return img_o, mask_o