import warnings
warnings.filterwarnings('ignore')
import os
import shutil
from config.model_config import network_cfg
import torch
from torch import optim
from torch.autograd import Variable as V
from torch.utils.data import DataLoader
import time
from custom.utils.logger import Logger
from custom.utils.model_backup import model_backup
from custom.utils.lr_scheduler import WarmupMultiStepLR
from custom.utils.tensorboad_utils import get_writer

def train():
    # 训练准备
    os.makedirs(network_cfg.checkpoints_dir, exist_ok=True)
    logger_dir = network_cfg.log_dir
    os.makedirs(logger_dir, exist_ok=True)
    logger = Logger(logger_dir+"/trainlog.txt", level='debug').logger
    model_backup(logger_dir+"/backup.tar")
    tensorboad_dir = logger_dir + "/tf_logs"
    if os.path.exists(tensorboad_dir): 
        shutil.rmtree(tensorboad_dir)
    writer = get_writer(tensorboad_dir)

    # 网络定义
    net = network_cfg.network.cuda()
    # 定义损失函数
    train_loss_func = network_cfg.train_loss_func
    valid_loss_func = network_cfg.valid_loss_func

    if os.path.exists(network_cfg.load_from):
        print("Load pretrain weight from: " + network_cfg.load_from)
        net.load_state_dict(torch.load(network_cfg.load_from, map_location=network_cfg.device))

    train_dataset = network_cfg.train_dataset
    train_dataloader = DataLoader(train_dataset, 
                                batch_size=network_cfg.batchsize, 
                                shuffle=network_cfg.shuffle,
                                num_workers=network_cfg.num_workers, 
                                drop_last=network_cfg.drop_last)
    valid_dataset = network_cfg.valid_dataset
    valid_dataloader = DataLoader(valid_dataset, 
                                batch_size=network_cfg.batchsize, 
                                shuffle=False,
                                num_workers=network_cfg.num_workers, 
                                drop_last=network_cfg.drop_last)
    
    optimizer = optim.AdamW(params=net.parameters(), lr=network_cfg.lr, weight_decay=network_cfg.weight_decay)
    scheduler = WarmupMultiStepLR(optimizer=optimizer,
                                milestones=network_cfg.milestones,
                                gamma=network_cfg.gamma,
                                warmup_factor=network_cfg.warmup_factor,
                                warmup_iters=network_cfg.warmup_iters,
                                warmup_method=network_cfg.warmup_method,
                                last_epoch=network_cfg.last_epoch)
    
    
    time_start=time.time()
    for epoch in range(network_cfg.total_epochs): 
        #Training Step!
        net.train()
        for ii, (train_data, train_label) in enumerate(train_dataloader):
            train_data = V(train_data).cuda()
            train_label = V(train_label).cuda()
            t_out = net(train_data)
            t_loss = train_loss_func(t_out, train_label)
            loss_all = V(torch.zeros(1)).cuda()
            loss_info = ""
            for loss_item, loss_val in t_loss.items():
                loss_all += loss_val
                loss_info += "{}={:.4f}\t ".format(loss_item,loss_val.item())
                if ii == 0:
                    writer.add_scalar('TrainLoss/{}'.format(loss_item),loss_val.item(), epoch+1)
            time_temp=time.time()
            eta = ((network_cfg.total_epochs-epoch)+(1-(ii+1)/len(train_dataloader)))/(epoch+(ii+1)/len(train_dataloader))*(time_temp-time_start)/60
            if eta < 60:
                eta = "{:.1f}min".format(eta)
            else:
                eta = "{:.1f}h".format(eta/60.0)
            logger.info('Epoch:[{}/{}]\t Iter:[{}/{}]\t Eta:{}\t {}'.format(epoch+1 ,network_cfg.total_epochs, ii+1, len(train_dataloader), eta, loss_info))

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

        writer.add_scalar('LR', optimizer.state_dict()['param_groups'][0]['lr'], epoch)
        scheduler.step()

        # Valid Step!
        if (epoch+1) % network_cfg.valid_interval == 0:
            valid_loss = dict()
            net.eval()
            for ii, (valid_data,valid_label) in enumerate(valid_dataloader):
                valid_data = V(valid_data).cuda()
                valid_label = V(valid_label).cuda()
                with torch.no_grad():
                    v_out = net(valid_data)
                    v_loss = valid_loss_func(v_out, valid_label)
                    
                for loss_item, loss_val in v_loss.items():
                    if loss_item not in valid_loss:
                        valid_loss[loss_item] = loss_val.item()
                    else:
                        valid_loss[loss_item] += loss_val.item() 
            loss_info = ""              
            for loss_item, loss_val in valid_loss.items():
                valid_loss[loss_item] /= (ii+1)
                loss_info += "{}={:.4f}\t ".format(loss_item,valid_loss[loss_item])
                writer.add_scalar('ValidLoss/{}'.format(loss_item),valid_loss[loss_item], (epoch+1))
            
            logger.info('Validating Step:\t {}'.format(loss_info))
            
        if (epoch+1) % network_cfg.checkpoint_save_interval == 0:
            torch.save(net.state_dict(), network_cfg.checkpoints_dir+"/{}.pth".format(epoch+1))
    writer.close()

if __name__ == '__main__':
	train()
