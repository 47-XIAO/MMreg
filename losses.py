import torch
import torch.nn.functional as F
import numpy as np
import math
import torch.nn as nn

class Grad3d(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        super(Grad3d, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred, y_true):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad


class NCC_vxm(torch.nn.Module):
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        super(NCC_vxm, self).__init__()
        self.win = win

    def forward(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)
    

def multi_class_dice_loss(soft_pred, target, num_labels, weights=None,ignore_index=[0]):
    loss = 0
    target = target.float()
    smooth = 1e-6
    valid_classes = 0
    # labels_idx[0,2,3,4,5,7,8,10,11,12,13,14,15,16,17,18,26,24,28,41,42,43,44,46,47,49,50,51,52,53,54,58,60]
    for i in range(num_labels):
        if ignore_index is not None and i in ignore_index:
            continue  
        score = soft_pred[:, i]
        target_ = target == i
        intersect = torch.sum(score * target_)
        y_sum = torch.sum(target_ * target_)
        z_sum = torch.sum(score * score)
        single_loss = ((2 * intersect + smooth) / (z_sum + y_sum + smooth))
    
        if weights is not None:
            single_loss *= weights[i]

        loss += single_loss
        valid_classes += 1

    loss = 1 - (loss / valid_classes)
    return loss


def binary_dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """

    def __init__(self, ignore_index=[0]):
        super(DiceLoss, self).__init__()
        # self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target, weight=None):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        # dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        num_valid_classes = 0
        predict = F.softmax(predict, dim=1)

        for i in range(0, target.shape[1]):
            if i in self.ignore_index:
                continue  # 跳过忽略类别
            dice_loss = binary_dice_loss(predict[:, i], target[:, i])
            if weight is not None:
                assert weight.shape[0] == target.shape[1], \
                    'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                dice_loss *= weight[i]
            total_loss += dice_loss
            num_valid_classes += 1

        # return total_loss / (target.shape[1] - len(self.ignore_index)) \
        #     if self.ignore_index is not None else total_loss / target.shape[1]
        # 避免除以0（所有类别都被忽略时）
        # if num_valid_classes == 0:
        #     raise ValueError("所有类别都被ignore_index指定忽略，无法计算Dice Loss！")
        return total_loss / num_valid_classes


class MultiClassDiceLoss(nn.Module):
    """多类Dice损失类,支持权重和忽略类别
    
    Args:
        num_labels: 
        weights: [num_labels,]
        ignore_index: 单个int或列表
        smooth: 避免分母为0,默认为1e-6
        use_softmax: 是否对输入进行softmax激活
    """
    def __init__(self, num_labels, weights=None, ignore_index=[0], smooth=1e-6, use_softmax=True):
        super(MultiClassDiceLoss, self).__init__()
        self.num_labels = num_labels
        self.smooth = smooth
        self.use_softmax = use_softmax
        # self.dice_one = DiceLoss(ignore_index=[])
        
        if weights is not None:
            self.weights = torch.tensor(weights, dtype=torch.float32)
        else:
            self.weights = None

        if ignore_index is not None:
            self.ignore_index = ignore_index if isinstance(ignore_index, list) else [ignore_index]
        else:
            self.ignore_index = []
        
        for idx in self.ignore_index:
            if idx < 0 or idx >= self.num_labels:
                raise ValueError(f"忽略的类别索引{idx}超出有效范围[0, {self.num_labels-1}]")

    def forward(self, predict, target):
        """
        Args:
            predict: 模型预测输出，形状为[N, 1, *]
                     - 如果use_softmax=True:输入为logits(未经过softmax)
                     - 如果use_softmax=False:输入为已归一化的概率分布
            target: 目标标签，形状为[N, *](非one-hot)
        
        Return:
            加权多类Dice损失值
        """
        # 验证输入形状
        assert predict.shape[0] == target.shape[0] and predict.shape[2:] == target.shape[2:], "batch和空间维度不匹配"

        # max_val = predict.max().item()
        # min_val = predict.min().item()

        # predict = predict.squeeze(1)  # [N,D,H,W]
        # predict = F.one_hot(predict.long().squeeze(1),num_classes=self.num_labels).permute(0,4,1,2,3).float()
        # soft_pred = F.softmax(predict, dim=1)  # Logits转概率
        # target = target.squeeze(1)    # [N,D,H,W]
        
        weights = self.weights.to(predict.device) if self.weights is not None else None
        
        loss_mutil = multi_class_dice_loss(
            soft_pred=predict,
            target=target,
            num_labels=self.num_labels,
            weights=weights,
            ignore_index=self.ignore_index)

        # loss_prostate = self.dice_one(predict_prostate, target_prostate)
        
        # loss = (loss_mutil + loss_prostate)/2

        return loss_mutil

def pdist_squared(x):
    xx = (x ** 2).sum(dim=1).unsqueeze(2)
    yy = xx.permute(0, 2, 1)
    dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
    dist[dist != dist] = 0
    dist = torch.clamp(dist, 0.0, 255.0)
    return dist

def MINDSSC(img, radius=3, dilation=3, device=torch.device('cuda')):
    """
    *Preliminary* pytorch implementation.
    MIND-SSC Losses for VoxelMorph
    """
    # see http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for details on the MIND-SSC descriptor

    # kernel size
    kernel_size = radius * 2 + 1

    # define start and end locations for self-similarity pattern
    six_neighbourhood = torch.tensor([[0, 1, 1],
                                      [1, 1, 0],
                                      [1, 0, 1],
                                      [1, 1, 2],
                                      [2, 1, 1],
                                      [1, 2, 1]]).long()

    # squared distances
    dist = pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)

    # define comparison mask
    x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
    mask = ((x > y).view(-1) & (dist == 2).view(-1))

    # build kernel
    idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :]
    idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :]
    mshift1 = torch.zeros(12, 1, 3, 3, 3).to(device)
    mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:, 0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
    mshift2 = torch.zeros(12, 1, 3, 3, 3).to(device)
    mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:, 0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
    rpad1 = nn.ReplicationPad3d(dilation)
    rpad2 = nn.ReplicationPad3d(radius)

    # compute patch-ssd
    ssd = F.avg_pool3d(rpad2(
        (F.conv3d(rpad1(img), mshift1, dilation=dilation) - F.conv3d(rpad1(img), mshift2, dilation=dilation)) ** 2),
        kernel_size, stride=1)

    # MIND equation
    mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
    mind_var = torch.mean(mind, 1, keepdim=True)
    mind_var = mind_var.cpu().data
    mind_var = torch.clamp(mind_var, mind_var.mean() * 0.001, mind_var.mean() * 1000)

    mind_var = mind_var.to(device)  # .to(device)
    mind /= mind_var
    mind = torch.exp(-mind)

    # permute to have same ordering as C++ code
    mind = mind[:, torch.tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]

    return mind  # Tensor: (N, 12, 192, 160, 192)


def MIND_loss(x, y):
    """
    The loss is small, even the voxel intensity distribution of fake image is so difference, loss.item < 0.14
    """
    return torch.mean((MINDSSC(x) - MINDSSC(y)) ** 2)


def compute_pairwise_loss(Ls, similarity_fn, batch_size, tau=0.5, device=None):
    """Computation of the final loss.
    
    Args:
        Ls (list): the latent spaces.
        similarity_fn (func): the similarity function between two datapoints x and y.
        tau (float): the temperature to apply to the similarities.
        device (str): the torch device to store the data and perform the computations.
    
    Returns (list of float):
        softmaxes: the loss for each positive sample (length=2N, with N=batch size).
        similarities: the similarity matrix with all pairwise similarities (2N, 2N)
    
    Note:
        This implementation works in the case where only 2 modalities are of
        interest (M=2). Please refer to the paper for the full algorithm.
    """
    # Computation of the similarity matrix
    # The matrix contains the pairwise similarities between each sample of the full batch
    # and each modalities.
    points = torch.cat([L.cuda() for L in Ls])
    N = batch_size
    similarities = torch.zeros(2*N, 2*N).cuda()
    for i in range(2*N):
        for j in range(i+1):
            s = similarity_fn(points[i], points[j])/tau  #计算相似度 
            similarities[i, j] = s
            similarities[j, i] = s

    # Computation of the loss, one row after the other.
    irange = np.arange(2*N)
    softmaxes = torch.empty(2*N).cuda()
    for i in range(2*N):
        j = (i + N) % (2 * N)  #positive sample 
        pos = similarities[i, j]
        # The negative examples are all the remaining points
        # excluding self-similarity
        neg = similarities[i][irange != i]
        softmaxes[i] = -pos + torch.logsumexp(neg, dim=0)
    return softmaxes, similarities # softmaxs:[loss0, loss1, loss2, loss3, loss4, loss5]

def frequency_loss(x, y, lambda_amp=0.1, eps=1e-8):
    """
    Frequency-domain similarity for multimodal, unaligned features.

    Args:
        x, y: feature maps (B, C, H, W) or (B, C, D, H, W)
        lambda_amp: weight for amplitude similarity
        eps: numerical stability

    Returns:
        similarity: scalar tensor
    """

    # FFT (2D or 3D)
    dims = tuple(range(-2, 0)) if x.dim() == 4 else tuple(range(-3, 0))
    Fx = torch.fft.fftn(x, dim=dims)
    Fy = torch.fft.fftn(y, dim=dims)

    # Amplitude and phase
    # Ax = torch.abs(Fx) + eps
    # Ay = torch.abs(Fy) + eps
    Px = torch.angle(Fx)
    Py = torch.angle(Fy)

    # ----- Phase similarity (geometry) -----
    phase_sim = torch.cos(Px - Py).mean()

    # # ----- Amplitude similarity (style) -----
    # Ax_log = torch.log(Ax)
    # Ay_log = torch.log(Ay)

    # amp_sim = F.cosine_similarity(
    #     Ax_log.flatten(1),
    #     Ay_log.flatten(1),
    #     dim=1
    # ).mean()

    # # ----- Combined similarity -----
    # sim = phase_sim + lambda_amp * amp_sim

    return 1 - phase_sim
