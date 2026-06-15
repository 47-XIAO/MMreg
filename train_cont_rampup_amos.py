import glob
# from torch.utils.tensorboard import SummaryWriter
import os, losses, utils
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
# from model import PR,PR_diff
# from models.models2 import RDP
# from model2 import PRWHC as network
from model import PRC as network
# from model2 import PRWHC2 as network
import random
import torch.nn.functional as F
import math
import torch.fft as fft
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
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
    train_dir = '/hy-tmp/xsq/amos22_exp/train/images/'
    val_dir = '/hy-tmp/xsq/amos22_exp/val/images/'
    weights = [1, 1]  # loss weights
    lr = 1e-4
    save_dir = 'PRC_mind_{}_reg_{}_lr_{}_AMOS_contl10_40e/'.format(*weights, lr)
    # save_dir = 'test/'.format(*weights, lr)
    if not os.path.exists('experiments_amos/' + save_dir):
        os.makedirs('experiments_amos/' + save_dir)
    if not os.path.exists('logs_amos/' + save_dir):
        os.makedirs('logs_amos/' + save_dir)
    sys.stdout = Logger('logs_amos/' + save_dir)
    f = open(os.path.join('logs_amos/'+save_dir, 'losses and dice' + ".txt"), "a")

    epoch_start = 1
    max_epoch = 40
    img_size = (208, 144, 128)
    cont_training = False
    rampup_start = 10
    rampup_length = max_epoch - rampup_start
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

# Adbomen_AMOS
    train_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16)),
                                         ])
    val_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])
    train_set = datasets.CTMRI_AMOS_Dataset(train_dir, transforms=train_composed)
    val_set = datasets.CTMRI_AMOS_Dataset(val_dir, transforms=val_composed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)

   
    criterion = losses.MIND_loss
    criterions = [criterion]
    criterions += [losses.Grad3d(penalty='l2')]
    criterions += [losses.compute_pairwise_loss]
    # criterions += [losses.frequency_loss]
    best_dsc = 0
    # weights_CL = [0.25,0.25,0.25,0.25]
    # writer = SummaryWriter(log_dir='logs/'+save_dir)
    for epoch in range(epoch_start, max_epoch + 1):
        print('Training Starts')
        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        loss_cont_all = utils.AverageMeter()
        loss_sim_all = utils.AverageMeter()
        # loss_phase_all = utils.AverageMeter()
        idx = 0
        for data in train_loader:
            idx += 1
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
    
            output = model(x,y)
            faet = output[2:] 


            loss = 0
            loss_vals = []
           
            sim_loss = criterions[0](output[0], y)* weights[0]
            loss_vals.append(sim_loss)
            loss += sim_loss
            if epoch >= rampup_start:
                lambda_cont = gaussian_rampup(epoch - rampup_start, rampup_length, target=1.0)
                softmaxes, _ = criterions[2](Ls=faet[0:2], similarity_fn=sim_func, batch_size=2)* weights[0]
                Cont_loss1 = softmaxes.mean()
                # Cont_loss = lambda_cont * Cont_loss1
                loss_vals.append(Cont_loss1)
                loss += lambda_cont * Cont_loss1
           
            grad_loss = criterions[1](output[1], y)* weights[1]
            loss_vals.append(grad_loss)
            loss += grad_loss
          
            loss_all.update(loss.item(), y.numel())
            loss_sim_all.update(sim_loss.item(), y.numel())
           
            if epoch >= rampup_start:
                loss_cont_all.update(Cont_loss1.item(), y.numel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch >= rampup_start:
                print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, CL: {:.6f}, Reg: {:.6f}'.format(idx, len(train_loader), loss.item(), loss_vals[0].item(), loss_vals[1].item(),  loss_vals[2].item()))
            else:
                print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}'.format(idx, len(train_loader), loss.item(), loss_vals[0].item(), loss_vals[1].item()))
         

        print('{} Epoch {} loss {:.4f} loss_sim {:.4f} loss_cont {:.4f}'.format(save_dir, epoch, loss_all.avg, loss_sim_all.avg, loss_cont_all.avg))
        print('Epoch {} loss {:.4f} loss_sim {:.4f} loss_cont {:.4f}'.format(epoch, loss_all.avg, loss_sim_all.avg, loss_cont_all.avg), file=f, end=' ')
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
               
                output = model(x_val,y_val)
                
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
        }, save_dir='experiments_amos/' + save_dir, filename='dsc{:.3f}.pth.tar'.format(eval_dsc.avg))
        loss_all.reset()

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)


def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=4):
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
