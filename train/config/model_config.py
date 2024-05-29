import sys, os
work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(work_dir)
import torch
from custom.dataset.dataset import MyDataset
from custom.utils.data_transforms import *
from custom.model.backbones import Cascaded_ResUnet_refine_SPP
from custom.model.model_head import Model_Head
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
        backbone = Cascaded_ResUnet_refine_SPP(in_ch=1,channels=12, blocks=2),
        head = Model_Head(in_channels=12, num_class=5),
        apply_sync_batchnorm=False
        )

    # loss function
    loss_func = MixLoss(point_radius=[3, 1])

    # dataset
    train_dataset = MyDataset(
        dst_list_file = work_dir + "/train_data/processed_data/train.txt",
        transforms = TransformCompose([
            to_tensor(),
            normlize(win_clip=None),
            random_flip(axis=3, prob=0.5),
            random_flip(axis=1, prob=0.5),
            random_rotate3d(x_theta_range=[-0,0],
                            y_theta_range=[-0,0],
                            z_theta_range=[-10,10],
                            prob=0.3),
            random_gamma_transform(gamma_range=[0.7,1.3], prob=0.2),
            random_apply_mosaic(prob=0.2, mosaic_size=40, mosaic_num=4),
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
    num_workers = 4
    drop_last = False

    # optimizer
    lr = 1e-3
    weight_decay = 5e-4

    # scheduler
    milestones = [40,60,80]
    gamma = 0.5
    warmup_factor = 0.1
    warmup_iters = 1
    warmup_method = "linear"
    last_epoch = -1

    # debug
    total_epochs = 100
    valid_interval = 1
    checkpoint_save_interval = 1
    log_dir = work_dir + "/Logs/v1"
    checkpoints_dir = work_dir + '/checkpoints/v1'
    load_from = work_dir + '/checkpoints/v1/none.pth'

