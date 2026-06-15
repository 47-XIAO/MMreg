import glob
# from torch.utils.tensorboard import SummaryWriter
import os, losses, utils
import shutil
import sys
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
# from model import RDP
from model import PRC as network
import random
import torch.nn.functional as F
import math
# os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

same_seeds(24)
class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def sigmoid_ramp(epoch, start, end, target=1, slope=10):
    """
    Smoothly ramps from 0 to `target` between `start` and `end` using a sigmoid curve.
    
    Parameters:
        epoch (int): Current epoch.
        start (int): Epoch when ramp starts.
        end (int): Epoch when ramp ends.
        target (float): Maximum weight value at the end of ramp.
        slope (float): Controls the steepness of the curve. Higher = steeper.
        
    Returns:
        float: Ramp weight at this epoch.
    """
    if epoch <= start:
        return 0.0
    if epoch >= end:
        return target
    t = (epoch - start) / (end - start)  # normalize to [0,1]
    return target * 1 / (1 + math.exp(-slope * (t - 0.5)))

def gaussian_rampup(epoch, rampup_length, target=1.0):
    if epoch < 0:
        return 0.0
    elif epoch < rampup_length:
        phase = 1.0 - epoch / rampup_length
        return target * math.exp(-5.0 * phase * phase)
    else:
        return target

def main():
    #  Distance function
    simfunctions = {
    "euclidean" : lambda x, y: -torch.norm(x - y, p=2, dim=1).mean(),
    "L1"        : lambda x, y: -torch.norm(x - y, p=1, dim=1).mean(),
    "L2"        : lambda x, y: -torch.norm(x - y, p=2, dim=1).mean(),
    "MSE"       : lambda x, y: -(x - y).pow(2).mean(),
    "L3"        : lambda x, y: -torch.norm(x - y, p=3, dim=1).mean(),
    "Linf"      : lambda x, y: -torch.norm(x - y, p=float("inf"), dim=1).mean(),
    "soft_corr" : lambda x, y: F.softplus(x*y).sum(axis=1),
    # "corr"      : lambda x, y: (x*y).sum(axis=1),
    "corr"      : lambda x, y: (x*y).sum().mean(),
    "cosine"    : lambda x, y: F.cosine_similarity(x, y, dim=1, eps=1e-8).mean(),
    "angular"   : lambda x, y: F.cosine_similarity(x, y, dim=1, eps=1e-8).acos().mean() / math.pi,
    }
    sim_func = simfunctions["cosine"]
    batch_size = 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dir = '/hy-tmp/xsq/SR-Reg-select/train/volumes_center_norm_crop'
    val_dir = '/hy-tmp/xsq/SR-Reg-select/val/volumes_center_norm_crop'
    weights = [1, 1]  # loss weights
    lr = 1e-4
    save_dir = 'prwhc1_{}_reg_{}_lr_{}_sr_select_30e/'.format(*weights, lr)
    root = '/hy-tmp/xsq/RDP-main2/'
    if not os.path.exists('/hy-tmp/xsq/RDP-main2/experiments_sr/' + save_dir):
        os.makedirs('/hy-tmp/xsq/RDP-main2/experiments_sr/' + save_dir)
    if not os.path.exists('/hy-tmp/xsq/RDP-main2/logs_sr/' + save_dir):
        os.makedirs('/hy-tmp/xsq/RDP-main2/logs_sr/' + save_dir)
    sys.stdout = Logger('/hy-tmp/xsq/RDP-main2/logs_sr/' + save_dir)
    f = open(os.path.join(root+'/logs_sr/'+save_dir, 'losses and dice' + ".txt"), "a")
    py_path_old = sys.argv[0] #获取当前运行的脚本路径
    py_path_new = os.path.join(root+'/logs_sr/'+save_dir, os.path.basename(py_path_old))
    shutil.copy(py_path_old, py_path_new)

    epoch_start = 1
    max_epoch = 30
    # img_size = (144,176,160)
    img_size = (160, 192, 160)
    # img_size = (128,128,128)
    cont_training = False
    # rampup_start = 10
    # rampup_length = max_epoch - rampup_start
    '''
    Initialize model
    '''
    model = network(img_size, channels=16)
    model.cuda()

    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.register_model(img_size, 'bilinear')
    reg_model.cuda()
    stn_val = utils.SpatialTransformer(img_size, mode='nearest')
    stn_val.cuda()
    '''
    If continue from previous training
    '''
    if cont_training:
        model_dir = 'experiments/'+save_dir
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch,0.9),8)
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-1])['state_dict']
        model.load_state_dict(best_model)
        print(model_dir + natsorted(os.listdir(model_dir))[-1])
    else:
        updated_lr = lr
    '''
    Initialize training
    '''
    # LPBA
    # train_composed = transforms.Compose([trans.NumpyType((np.float32, np.float32))])

    # val_composed = transforms.Compose([trans.Seg_norm(),
    #                                    trans.NumpyType((np.float32, np.int16))])
    # train_set = datasets.LPBABrainDatasetS2S(glob.glob(train_dir + '*.pkl'), transforms=train_composed)
    # val_set = datasets.LPBABrainInferDatasetS2S(glob.glob(val_dir + '*.pkl'), transforms=val_composed)
    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    # Regpros
    # train_dataset = datasets.RegProDataset3(mr_dir=mr_dir, us_dir=us_dir, for_test=False)
    # val_dataset = datasets.RegProDataset3(mr_dir=mr_dir.replace('train', 'val'), us_dir=us_dir.replace('train', 'val'),
    #                              for_test=True)
    
    # train_loader = DataLoader(dataset=train_dataset,
    #                           batch_size=1,
    #                           shuffle=True,
    #                           drop_last=True,
    #                           num_workers=4)

    # val_loader = DataLoader(dataset=val_dataset,
    #                         batch_size=1, num_workers=4)

    # SR-Reg Brain
    train_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16)),
                                         ])

    val_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])
    train_set = datasets.CTMRI_SR_REG_Dataset(train_dir, transforms=train_composed)
    val_set = datasets.CTMRI_SR_REG_Dataset(val_dir, transforms=val_composed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)


    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    # optimizer = optim.SGD(model.parameters(),lr=updated_lr, momentum=0.9,weight_decay=1e-5,nesterov=True)
    # optimizer = optim.AdamW(model.parameters(),lr=updated_lr,weight_decay=1e-5, amsgrad=True)


    # criterion = losses.NCC_vxm()
    # weight = [0,0,2,3,2,2,2]
    # criterion = losses.DiceLoss(
    #   ignore_index=[0]
    # )
    # num_classes = 19
    # criterion = losses.MultiClassDiceLoss(
    #     num_labels=num_classes,
    #     weights=None,
    #     # use_softmax=False
    # )
    # dice
    # criterion = losses.DiceLoss(
    #   ignore_index=[0],
    #   )
    # NCC
    # criterion = losses.NCC_vxm()
    criterion = losses.MIND_loss
    criterions = [criterion]
    criterions += [losses.Grad3d(penalty='l2')]
    # criterions += [losses.compute_pairwise_loss]
    best_dsc = 0
    weights_CL = [1,0.25,0.25,0.25]
    # writer = SummaryWriter(log_dir='logs/'+save_dir)
    for epoch in range(epoch_start, max_epoch + 1):
        print('Training Starts')
        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        idx = 0
        for data in train_loader:
            idx += 1
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            # x_seg_pros = data[2]
            # y_seg_pros = data[3]
            # x_seg = data[2]
            # y_seg =data[3]
            # data_z = data[4]

            output = model(x,y)
            # faet = output[2:] 


            # SR-Reg label连续化
            # seg_normalizer = trans.Seg_norm()
            # x_seg = seg_normalizer(x_seg.cpu().numpy())
            # x_seg = torch.from_numpy(x_seg).long().to(device)
            # y_seg = seg_normalizer(y_seg.cpu().numpy())
            # y_seg = torch.from_numpy(y_seg).long().to(device)

            # x_seg_val_one_hot = F.one_hot(x_seg.long().squeeze(1),num_classes=num_classes).permute(0,4,1,2,3).float()
            # y_seg_val_one_hot = F.one_hot(y_seg.long().squeeze(1),num_classes=num_classes).permute(0,4,1,2,3).float()
            # def_seg_out = reg_model([x_seg_val_one_hot.cuda().float(), output[1].cuda()])
            # def_segpros_out = reg_model([x_seg_pros.cuda().float(), output[1].cuda()])

            loss = 0
            loss_vals = []
            # for n, loss_function in enumerate(criterions):
            #     curr_loss = loss_function(output[n], y) * weights[n]
            #     loss_vals.append(curr_loss)
            #     loss += curr_loss
            # dce_loss = criterions[0](def_seg_out, y_seg_val_one_hot)* weights[0]
            sim_loss = criterions[0](output[0], y)* weights[0]
            loss_vals.append(sim_loss)
            loss += sim_loss
            # softmaxes, _ = criterions[2](Ls=faet[0:2], similarity_fn=sim_func, batch_size=2)* weights[0]
            # Cont_loss1 = softmaxes.mean()
            # softmaxes, _ = criterions[2](Ls=faet[2:4], similarity_fn=sim_func, batch_size=2)* weights[0]
            # Cont_loss2 = softmaxes.mean()
            # softmaxes, _ = criterions[2](Ls=faet[4:6], similarity_fn=sim_func, batch_size=2)* weights[0]
            # Cont_loss3 = softmaxes.mean()
            # softmaxes, _ = criterions[2](Ls=faet[6:8], similarity_fn=sim_func, batch_size=2)* weights[0]
            # Cont_loss4 = softmaxes.mean()
            # Cont_loss = weights_CL[0] * Cont_loss1 + weights_CL[1] * Cont_loss2 + weights_CL[2] * Cont_loss3 + weights_CL[3] * Cont_loss4
            # loss_vals.append(Cont_loss)
            # loss += Cont_loss

            grad_loss = criterions[1](output[1], y)* weights[1]
            loss_vals.append(grad_loss)
            loss += grad_loss
            loss_all.update(loss.item(), y.numel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}'.format(idx, len(train_loader), loss.item(), loss_vals[0].item(), loss_vals[1].item()))
            # print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, CL: {:.6f}, Reg: {:.6f}'.format(idx, len(train_loader), loss.item(), loss_vals[0].item(), loss_vals[1].item(),  loss_vals[2].item()))

        print('{} Epoch {} loss {:.4f}'.format(save_dir, epoch, loss_all.avg))
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg), file=f, end=' ')
        '''
        Validation
        '''
        eval_dsc = utils.AverageMeter()
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                data = [t.cuda() for t in data]
                x_val = data[0]
                y_val = data[1]
                x_seg_val = data[2]
                y_seg_val = data[3]
                seg_normalizer = trans.Seg_norm()
                x_seg_val = seg_normalizer(x_seg_val.cpu().numpy())
                x_seg_val = torch.from_numpy(x_seg_val).long().to(device)
                y_seg_val = seg_normalizer(y_seg_val.cpu().numpy())
                y_seg_val = torch.from_numpy(y_seg_val).long().to(device)
                output = model(x_val,y_val)
                # def_out = reg_model([x_seg_val.cuda().float(), output[1].cuda()])
                def_out = stn_val(x_seg_val.cuda().float(), output[1].cuda())
                dsc = utils.dice_val_VOI(def_out.long(), y_seg_val.long())
                eval_dsc.update(dsc.item(), x.size(0))
        print(epoch, ':',eval_dsc.avg)
        best_dsc = max(eval_dsc.avg, best_dsc)
        print(eval_dsc.avg, file=f)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_dsc': best_dsc,
            'optimizer': optimizer.state_dict(),
        }, save_dir='/hy-tmp/xsq/RDP-main2/experiments_sr/' + save_dir, filename='dsc{:.3f}.pth.tar'.format(eval_dsc.avg))
        loss_all.reset()

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)


def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=6):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))

if __name__ == '__main__':
    '''
    
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()