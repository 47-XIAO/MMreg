import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
from torch.distributions.normal import Normal

class Freprocess(nn.Module):
    def __init__(self, channels):
        super(Freprocess, self).__init__()

        # self.pre1 = nn.Conv3d(channels, channels, kernel_size=1, stride=1, padding=0)
        # self.pre2 = nn.Conv3d(channels, channels, kernel_size=1, stride=1, padding=0)
        
        # self.amp_fuse = nn.Sequential(
        #     nn.Conv3d(2 * channels, channels, kernel_size=1, stride=1, padding=0),
        #     nn.LeakyReLU(0.1, inplace=False),
        #     nn.Conv3d(channels, channels, kernel_size=1, stride=1, padding=0)
        # )
        
        # self.pha_fuse = nn.Sequential(
        #     nn.Conv3d(2 * channels, channels, kernel_size=1, stride=1, padding=0),
        #     nn.LeakyReLU(0.1, inplace=False),
        #     nn.Conv3d(channels, channels, kernel_size=1, stride=1, padding=0)
        # )
        
        # self.post = nn.Conv3d(channels, channels, kernel_size=1, stride=1, padding=0)
        # self.post2 = nn.Conv3d(channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, msf, panf):
        B, C, D, H, W = msf.shape

        # x1 = self.pre1(msf) + 1e-8  # [B, C, D, H, W]
        # x2 = self.pre2(panf) + 1e-8
        x1 = msf
        x2 = panf

        # 3D 实数 FFT (rfftn) —— 注意：最后一个维度被压缩为复数
        #[B, C, D, H, W//2 + 1] 
        msF = fft.rfftn(x1, dim=(-3, -2, -1), norm='backward')  # 在 D, H, W 上做 3D FFT
        panF = fft.rfftn(x2, dim=(-3, -2, -1), norm='backward')

        msF_amp = torch.abs(msF)      # [B, C, D, H, W//2+1]
        msF_pha = torch.angle(msF)
        panF_amp = torch.abs(panF)
        panF_pha = torch.angle(panF)

        amp_fuse = torch.ones_like(msF_amp)
        # pha_fuse = self.pha_fuse(torch.cat([msF_pha, panF_pha], dim=1))

        # real = amp_fuse * torch.cos(pha_fuse) + 1e-8
        # imag = amp_fuse * torch.sin(pha_fuse) + 1e-8

        # out_freq = torch.complex(real, imag)  # [B, C, D, H, W//2+1]

        # out = fft.irfftn(out_freq, s=(D, H, W), dim=(-3, -2, -1), norm='backward')  # [B, C, D, H, W]

        real_msF = amp_fuse * torch.cos(msF_pha) + 1e-8
        imag_msF = amp_fuse * torch.sin(msF_pha) + 1e-8
        out_freq_msF = torch.complex(real_msF, imag_msF)  # [B, C, D, H, W//2+1]
        out_msF = fft.irfftn(out_freq_msF, s=(D, H, W), dim=(-3, -2, -1), norm='backward')  # [B, C, D, H, W]
        out_msF = torch.real(out_msF)

        real_panF = amp_fuse * torch.cos(panF_pha) + 1e-8
        imag_panF = amp_fuse * torch.sin(panF_pha) + 1e-8
        out_freq_panF = torch.complex(real_panF, imag_panF)  # [B, C, D, H, W//2+1]
        out_panF = fft.irfftn(out_freq_panF, s=(D, H, W), dim=(-3, -2, -1), norm='backward')  # [B, C, D, H, W]
        out_panF = torch.real(out_panF)

        # return self.post(out_msF), self.post2(out_panF)
        return out_msF, out_panF

def stdv_channels_3d(x):
    """
    x: [B, C, D, H, W]
    返回: [B, 1, D, H, W] —— 每个体素位置的通道标准差
    """
    std = torch.std(x, dim=1, keepdim=True)  # [B, 1, D, H, W]
    return std


class Interaction3D(nn.Module):
    def __init__(self, channels):
        super(Interaction3D, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.contrast = stdv_channels_3d  # 局部对比度

        self.spa_att_vis = nn.Sequential(
            nn.Conv3d(channels, channels // 2, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.1),
            nn.Conv3d(channels // 2, channels, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()
        )
        self.cha_att_vis = nn.Sequential(
            nn.Conv3d(channels * 2, channels // 2, kernel_size=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv3d(channels // 2, channels * 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.post_vis = nn.Conv3d(channels * 2, channels, kernel_size=3, padding=1)

        self.spa_att_inf = nn.Sequential(
            nn.Conv3d(channels, channels // 2, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.1),
            nn.Conv3d(channels // 2, channels, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()
        )
        self.cha_att_inf = nn.Sequential(
            nn.Conv3d(channels * 2, channels // 2, kernel_size=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv3d(channels // 2, channels * 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.post_inf = nn.Conv3d(channels * 2, channels, kernel_size=3, padding=1) 


    def forward(self, fused_y, vis_y, inf):
        """
        输入:
            fused_y: [B, C, D, H, W]
            vis_y:   [B, C, D, H, W]
            inf:     [B, C, D, H, W]
        输出:
            vis_out: [B, C, D, H, W]
            inf_out: [B, C, D, H, W]
        """
        vis_diff = vis_y - fused_y  # [B, C, D, H, W]
        vis_map = self.spa_att_vis(vis_diff)  
        vis_res = vis_map * vis_y + vis_y  

        # vis_cat = torch.cat([vis_res, fused_y], dim=1)  # [B, 2C, D, H, W]

        # # 对比度 + 全局上下文
        # contrast_feat = self.contrast(vis_cat)      # [B, 1, D, H, W]
        # avg_feat = self.avgpool(vis_cat)           # [B, 2C, 1, 1, 1]
        # avg_feat = avg_feat.expand(-1, -1, vis_cat.size(2), vis_cat.size(3), vis_cat.size(4))  # [B, 2C, D, H, W]

        # cha_weight = self.cha_att_vis(contrast_feat + avg_feat)  # [B, 2C, D, H, W]
        # vis_cha = self.post_vis(cha_weight * vis_cat)  # [B, C, D, H, W]
        # vis_out = vis_cha + fused_y  # 残差连接


        inf_diff = inf - fused_y
        inf_map = self.spa_att_inf(inf_diff)
        inf_res = inf_map * inf + inf

        # inf_cat = torch.cat([inf_res, fused_y], dim=1)

        # contrast_feat_inf = self.contrast(inf_cat)
        # avg_feat_inf = self.avgpool(inf_cat)
        # avg_feat_inf = avg_feat_inf.expand(-1, -1, inf_cat.size(2), inf_cat.size(3), inf_cat.size(4))

        # cha_weight_inf = self.cha_att_inf(contrast_feat_inf + avg_feat_inf)
        # inf_cha = self.post_inf(cha_weight_inf * inf_cat)
        # inf_out = inf_cha + fused_y

        # return vis_out, inf_out
        return vis_res, inf_res

class NLBlockND_cross(nn.Module):
    # Our implementation of the attention block referenced https://github.com/tea1528/Non-Local-NN-Pytorch

    def __init__(self, in_channels, inter_channels=None, mode='embedded',
                 dimension=3, bn_layer=True):
        """Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
        """
        super(NLBlockND_cross, self).__init__()

        assert dimension in [1, 2, 3]

        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # assign appropriate convolutional, max pool, and batch norm layers for different dimensions
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.InstanceNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.InstanceNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.InstanceNorm1d

        # function g in the paper which goes through conv. with kernel size 1
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        # add BatchNorm layer after the last conv layer
        if bn_layer:
            self.W_z = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                bn(self.in_channels)
            )
            # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
            # nn.init.constant_(self.W_z[1].weight, 0)
            # nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                nn.ReLU()
            )

    def forward(self, x_thisBranch, x_otherBranch):
        #x_thisBranch for g and theta
        #x_otherBranch for phi
        """
        args
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        """
        # print(x_thisBranch.shape)

        batch_size = x_thisBranch.size(0)

        # (N, C, THW)
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation
        g_x = self.g(x_thisBranch).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == "gaussian":
            theta_x = x_thisBranch.view(batch_size, self.in_channels, -1)
            phi_x = x_otherBranch.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x_thisBranch).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x_otherBranch).view(batch_size, self.inter_channels, -1)
            # theta_x = theta_x.permute(0, 2, 1)
            phi_x = phi_x.permute(0, 2, 1)
            f = torch.matmul(phi_x, theta_x)

        # elif self.mode == "concatenate":
        else: #default as concatenate
            theta_x = self.theta(x_thisBranch).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x_otherBranch).view(batch_size, self.inter_channels, 1, -1)

            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)

            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))

        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1)  # number of position in x
            f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)

        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x_thisBranch.size()[2:])

        W_y = self.W_z(y)
        # residual connection
        z = W_y + x_thisBranch

        return z
    

class Freprocess_AMP(nn.Module):
    def __init__(self, channels):
        super(Freprocess_AMP, self).__init__()

        self.pre1 = nn.Conv3d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.pre2 = nn.Conv3d(channels, channels, kernel_size=1, stride=1, padding=0)
        
        self.amp_fuse = nn.Sequential(
            nn.Conv3d(2 * channels, channels, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv3d(channels, channels, kernel_size=1, stride=1, padding=0)
        )
        
        self.pha_fuse = nn.Sequential(
            nn.Conv3d(2 * channels, channels, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv3d(channels, channels, kernel_size=1, stride=1, padding=0)
        )
        
        self.post = nn.Conv3d(channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, msf, panf):
        B, C, D, H, W = msf.shape

        x1 = self.pre1(msf) + 1e-8  # [B, C, D, H, W]
        x2 = self.pre2(panf) + 1e-8

        # 3D 实数 FFT (rfftn) —— 注意：最后一个维度被压缩为复数
        #[B, C, D, H, W//2 + 1] 
        msF = fft.rfftn(x1, dim=(-3, -2, -1), norm='backward')  # 在 D, H, W 上做 3D FFT
        panF = fft.rfftn(x2, dim=(-3, -2, -1), norm='backward')
   
        # [B, C, D, H, W//2+1]
        msF_pha = torch.angle(msF)
        panF_amp = torch.abs(panF)

        real = panF_amp * torch.cos(msF_pha) + 1e-8
        imag = panF_amp * torch.sin(msF_pha) + 1e-8
        out_freq = torch.complex(real, imag)  # [B, C, D, H, W//2+1]

        out = fft.irfftn(out_freq, s=(D, H, W), dim=(-3, -2, -1), norm='backward')  # [B, C, D, H, W]

        return self.post(out)



# class Freattention(nn.Module):
#     def __init__(self, channels):
#         super(Freattention, self).__init__()

#         self.pre1 = nn.Conv3d(channels, channels, kernel_size=1, stride=1, padding=0)
#         self.pre2 = nn.Conv3d(channels, channels, kernel_size=1, stride=1, padding=0)

#         self.conv = nn.Conv3d(channels * 2, channels, kernel_size=1, bias=False)

#         self.fusion_conv = nn.Conv3d(channels, channels, kernel_size=1, stride=1, padding=0)
        
#         self.post = nn.Conv3d(channels, channels, kernel_size=1, stride=1, padding=0)

#     def forward(self, moving, fixed):
#         B, C, D, H, W = moving.shape

#         x1 = self.pre1(moving) + 1e-8  # [B, C, D, H, W]
#         x2 = self.pre2(fixed) + 1e-8

#         # 3D 实数 FFT (rfftn) —— 注意：最后一个维度被压缩为复数
#         #[B, C, D, H, W//2 + 1] 
#         movingF = fft.rfftn(x1, dim=(-3, -2, -1), norm='backward')  # 在 D, H, W 上做 3D FFT
#         fixedF = fft.rfftn(x2, dim=(-3, -2, -1), norm='backward')

#         movingF_amp = torch.abs(movingF)      # [B, C, D, H, W//2+1]
#         fixedF_amp = torch.abs(fixedF)

#         moving_ap = nnf.adaptive_avg_pool3d(movingF_amp, (1, 1, 1))
#         moving_mp = nnf.adaptive_max_pool3d(movingF_amp, (1, 1, 1))
#         moving_map = torch.cat([moving_ap, moving_mp], dim=1)

#         fixed_ap = nnf.adaptive_avg_pool3d(fixedF_amp, (1, 1, 1))
#         fixed_mp = nnf.adaptive_max_pool3d(fixedF_amp, (1, 1, 1))
#         fixed_map = torch.cat([fixed_ap, fixed_mp], dim=1)

#         AW_M = self.conv(moving_map) # [B, C, 1, 1, 1]
#         AW_M = AW_M.expand_as(movingF_amp)
#         att_moving = movingF * AW_M  # complex × real → complex

#         AW_F = self.conv(fixed_map)  # [B, C, 1, 1, 1]
#         AW_F = AW_F.expand_as(fixedF_amp)

#         # Apply attention to complex spectrum
#         att_fixed = fixedF * AW_F  # complex × real → complex

#         moving_out = fft.ifftn(att_moving, s=(D, H, W),dim=(-3, -2, -1), norm='ortho')  # complex
#         moving_out = torch.real(moving_out)  # take real part

#         fixed_out = fft.ifftn(att_fixed, s=(D, H, W), dim=(-3, -2, -1), norm='ortho')  # complex
#         fixed_out = torch.real(fixed_out)  # take real part

#         out_M = self.fusion_conv(moving+moving_out)
#         out_F = self.fusion_conv(fixed+fixed_out)


#         return out_M,out_F

class ProjectionLayer(nn.Module):
    def __init__(self, in_channels, dim=6, norm=nn.LayerNorm):
        super().__init__()
        self.norm = norm(dim)
        self.proj = nn.Linear(in_channels, dim)
        self.proj.weight = nn.Parameter(Normal(0, 1e-5).sample(self.proj.weight.shape))
        self.proj.bias = nn.Parameter(torch.zeros(self.proj.bias.shape))

    def forward(self, feat):
        feat = feat.permute(0, 2, 3, 4, 1)
        feat = self.norm(self.proj(feat))
        return feat

class DualGateCrossModalCA(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(1, channels // reduction)

        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        # shared semantic embedding
        self.embed = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True)
        )

        # modality-specific projections
        self.fc_m = nn.Linear(hidden, channels, bias=False)
        self.fc_f = nn.Linear(hidden, channels, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, M, F):
        B, C, _, _, _ = M.shape

        m_vec = self.avg_pool(M).view(B, C)
        f_vec = self.avg_pool(F).view(B, C)

        m_emb = self.embed(m_vec)
        f_emb = self.embed(f_vec)

        # shared discrepancy signal
        diff = torch.abs(m_emb - f_emb)

        gate_M = self.sigmoid(self.fc_m(diff)).view(B, C, 1, 1, 1)
        gate_F = self.sigmoid(self.fc_f(diff)).view(B, C, 1, 1, 1)

        M_out = M * gate_M
        F_out = F * gate_F

        return M_out, F_out

class MultiModalChannelAttention(nn.Module):
    def __init__(self, C, reduction=2):
        super().__init__()
        hidden = C // reduction

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        # modality-specific gates, cross-conditioned
        self.fc_f = nn.Sequential(
            nn.Conv3d(4 * C, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, C, 1, bias=False),
            nn.Sigmoid()
        )

        self.fc_m = nn.Sequential(
            nn.Conv3d(4 * C, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, C, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, f, m):
        # f, m: (B, C, D, H, W)

        f_avg = self.avg_pool(f)
        f_max = self.max_pool(f)

        m_avg = self.avg_pool(m)
        m_max = self.max_pool(m)

        # cross-modal descriptors
        f_desc = torch.cat([f_avg, f_max, m_avg, m_max], dim=1)
        m_desc = torch.cat([m_avg, m_max, f_avg, f_max], dim=1)

        gate_f = self.fc_f(f_desc)
        gate_m = self.fc_m(m_desc)

        return f * gate_f, m * gate_m

class WavePool3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        # x: [B, C, D, H, W]
        LL = self.pool(x)
        up = F.interpolate(LL, scale_factor=2, mode='trilinear', align_corners=False)
        HF = x - up
        return LL, HF
    
class SEBlock3D(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)  # [B,C,1,1,1]
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)   # Squeeze
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y                      # Channel-wise scaling
    
# class SpatialAttn3D(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv = nn.Conv3d(1, 1, kernel_size=3, padding=1)

#     def forward(self, hf):
#         # hf: [B,C,D,H,W]
#         attn = hf.mean(dim=1, keepdim=True)
#         attn = torch.sigmoid(self.conv(attn))
#         return attn * hf

class WaveletChannelEnhance3D(nn.Module):
    def __init__(self, channels, alpha=0.2, reduction=16):
        super().__init__()
        self.wavelet = WavePool3D()
        if reduction is None:
            self.reduction = max(channels // 2, 4)
        else:
            self.reduction = reduction
        self.ca = SEBlock3D(channels, reduction=self.reduction)
        self.alpha = alpha

    def forward(self, x):
        LL, HF = self.wavelet(x)   # HF = LH+HL+HH
        HF = self.ca(HF)

        LL_up = F.interpolate(
            LL,
            size=HF.shape[2:],
            mode='trilinear',
            align_corners=False
        )
        return HF + self.alpha * LL_up


class WaveletCrossChannelEnhance3D(nn.Module):
    def __init__(self, channels, num_heads,alpha=0.2, reduction=16):
        super().__init__()
        self.wavelet = WavePool3D()
        if reduction is None:
            self.reduction = max(channels // 2, 4)
        else:
            self.reduction = reduction
        # self.ca = SEBlock3D(channels, reduction=self.reduction)
        self.cmca = CrossModalChannelAttention(channels, num_heads)
        self.alpha = alpha

    def forward(self, m, f):
        LLm, HFm = self.wavelet(m)   # HF = LH+HL+HH
        LLf, HFf = self.wavelet(f)   # HF = LH+HL+HH
        HFm = self.cmca(HFm,HFf)
    
        LLm_up = F.interpolate(
            LLm,
            size=HFm.shape[2:],
            mode='trilinear',
            align_corners=False
        )

        M =  HFm + self.alpha * LLm_up

        return M
    
# class ChannelAttn3D(nn.Module):
#     """
#     Use on HF features only
#     """
#     def __init__(self, channels, reduction=8):
#         super().__init__()
#         mid = max(channels // reduction, 4)

#         self.mlp = nn.Sequential(
#             nn.Linear(channels, mid, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(mid, channels, bias=False)
#         )

#     def forward(self, hf):
#         """
#         hf: [B, C, D, H, W]  (high-frequency feature)
#         """
#         B, C, _, _, _ = hf.shape

#         # Avg pooling
#         avg = F.adaptive_avg_pool3d(hf, 1).view(B, C)

#         # Max pooling
#         mx = F.adaptive_max_pool3d(hf, 1).view(B, C)

#         # HFP-style fusion (sum, not concat)
#         attn = self.mlp(avg) + self.mlp(mx)
#         attn = torch.sigmoid(attn).view(B, C, 1, 1, 1)

#         return attn



# class HPSC(nn.Module):
#     def __init__(self, channels, alpha=0.2):
#         super(HPSC, self).__init__()
#         self.alpha = alpha
#         self.wavelet = WavePool3D()
#         self.spatial = SpatialAttn3D()
#         self.channel = ChannelAttn3D(channels)
#     def forward(self, x):
#         _, hf = self.wavelet(x)
#         spatial = self.spatial(hf) # output of spatial path
#         channel = self.channel(hf) # output of channel path
#         return x + self.alpha * (x * spatial + x * channel)
    
class CrossModalChannelAttention(nn.Module):
    def __init__(self, C, num_heads=4, dropout=0.1):
        super().__init__()
        
        # 1. Compact Embedding (Squeeze)
        # We fuse avg and max immediately to create a robust channel descriptor
        self.embedding = nn.Sequential(
            nn.Linear(C * 2, C),
            nn.LayerNorm(C),
            nn.ReLU()
        )

        # 2. Cross-Attention Mechanism
        # F queries M, and M queries F
        self.cross_attn = nn.MultiheadAttention(embed_dim=C, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(C)
        self.ffn = nn.Sequential(
            nn.Linear(C, 2 * C),
            nn.GELU(),
            nn.Linear(2 * C, C)
        )
        
        # 3. Excitation (Gating)
        self.gate_f = nn.Sequential(nn.Linear(C, C), nn.Sigmoid())
        self.gate_m = nn.Sequential(nn.Linear(C, C), nn.Sigmoid())

    def get_descriptor(self, x):
        # x: (B, C, D, H, W)
        # Global Average and Max Pooling
        x_avg = x.mean(dim=(2, 3, 4))
        x_max = x.amax(dim=(2, 3, 4))
        # Concatenate and project: (B, C*2) -> (B, 1, C)
        y = torch.cat([x_avg, x_max], dim=1)
        y = self.embedding(y).unsqueeze(1) 
        return y

    def forward(self, m, f):
        # f, m: (B, C, D, H, W)
        B, C, D, H, W = f.shape

        # 1. Extract Channel Descriptors (B, 1, C)
        f_token = self.get_descriptor(f)
        m_token = self.get_descriptor(m)

        # 2. Cross-Attention: Exchange Information
        # Q=f, K=m, V=m -> What parts of Moving are relevant to Fixed?
        # We process them together for efficiency by stacking
        
        # Cross Attn: F attends to M
        # attn_f, _ = self.cross_attn(query=f_token, key=m_token, value=m_token)
        # Cross Attn: M attends to F
        # attn_m, _ = self.cross_attn(query=m_token, key=f_token, value=f_token)
        attn_m, _ = self.cross_attn(query=f_token, key=m_token, value=m_token)

        # Residual + Norm + FFN
        # f_out = self.norm(f_token + attn_f)
        m_out = self.norm(m_token + attn_m)
        
        # f_out = f_out + self.ffn(f_out)
        m_out = m_out + self.ffn(m_out)

        # 3. Generate Gates
        # weight_f = self.gate_f(f_out).view(B, C, 1, 1, 1)
        weight_m = self.gate_m(m_out).view(B, C, 1, 1, 1)

        # 4. Modulate Input
        # return f, m * weight_m
        # return m * weight_m, f
        return m * weight_m




if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    a = torch.randn(1, 64, 40,40,32).to(device)
    b = torch.randn(1, 64, 40,40,32).to(device)
    # WaveletChannelEnhance3D = HPSC(64).to(device)
    output = WaveletChannelEnhance3D(a)
    print(output.shape)
    # model = Freprocess(channels=2)

    # print(out2.shape)
