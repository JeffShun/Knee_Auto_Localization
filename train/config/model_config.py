import sys, os
work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(work_dir)
import torch
from custom.dataset.dataset import MyDataset
from custom.utils.data_transforms import *
from custom.model.backbones import Cascaded_ResUnet_SPP, ResUNET_Twohead, HRNet
from custom.model.model_network import Model_Network
from custom.model.model_loss import MixLoss


class network_cfg:
    device = torch.device('cuda')
    dist_backend = 'nccl'
    dist_url = 'env://'

    # img
    img_size = (96, 192, 192)

    # network
    network = Model_Network(
        # backbone = HRNet()
        # backbone = Cascaded_ResUnet_SPP(in_ch=1, out_ch=5, channels=16, blocks=3)
        backbone = ResUNET_Twohead(in_ch=1, out_ch=5, channels=16, blocks=3)
        )

    # loss function
    train_loss_func = MixLoss(point_radiu=1)
    valid_loss_func = train_loss_func

    # dataset
    train_dataset = MyDataset(
        dst_list_file = work_dir + "/train_data/processed_data/train.txt",
        transforms = TransformCompose([
            to_tensor(),
            normlize(win_clip=None),
            random_flip(axis=3, prob=0.5),  # 左右对称
            # random_flip(axis=1, prob=0.5), # 上下对称
            random_rotate3d(x_theta_range=[-0,0],
                            y_theta_range=[-0,0],
                            z_theta_range=[-20,20],
                            prob=0.5),
            random_gamma_transform(gamma_range=[0.8,1.2], prob=0.5),
            random_apply_mosaic(prob=0.2, mosaic_size=20, mosaic_num=3),
            random_add_gaussian_noise(prob=0.2, mean=0, std=0.02),
            resize(img_size)
            ])
        )
    
    valid_dataset = MyDataset(
        dst_list_file = work_dir + "/train_data/processed_data/valid.txt",
        transforms = TransformCompose([
            to_tensor(),           
            normlize(win_clip=None),
            resize(img_size)
            ])
        )
    
    # train dataloader
    batchsize = 2
    shuffle = True
    num_workers = 12
    drop_last = False

    # optimizer
    lr = 1e-4
    weight_decay = 5e-4

    # scheduler
    milestones = [50, 100, 150]
    gamma = 0.5
    warmup_factor = 0.1
    warmup_iters = 1
    warmup_method = "linear"
    last_epoch = -1

    # debug
    total_epochs = 200
    valid_interval = 2
    checkpoint_save_interval = 2
    log_dir = work_dir + "/Logs/ResUNET+Refine+AUG+DirectionLoss"
    checkpoints_dir = work_dir + '/checkpoints/ResUNET+Refine+AUG+DirectionLoss'
    load_from = work_dir + '/checkpoints/ResUNET+Refine+AUG+DirectionLoss/none.pth'

