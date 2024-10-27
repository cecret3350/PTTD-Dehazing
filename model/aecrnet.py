import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.nn.modules import conv
from torch.nn.modules.utils import _pair
import math
from .DCNv2.dcn_v2 import DCN
from utils import *
from einops.layers.torch import Rearrange


def isqrt_newton_schulz_autograd(A, numIters):
    dim = A.shape[0]
    normA=A.norm()
    Y = A.div(normA)
    I = torch.eye(dim,dtype=A.dtype,device=A.device)
    Z = torch.eye(dim,dtype=A.dtype,device=A.device)

    for i in range(numIters):
        T = 0.5*(3.0*I - Z@Y)
        Y = Y@T
        Z = T@Z
    #A_sqrt = Y*torch.sqrt(normA)
    A_isqrt = Z / torch.sqrt(normA) # 取的是平方根的倒数
    return A_isqrt

def isqrt_newton_schulz_autograd_batch(A, numIters):
    batchSize,dim,_ = A.shape
    normA=A.view(batchSize, -1).norm(2, 1).view(batchSize, 1, 1)
    Y = A.div(normA)
    I = torch.eye(dim,dtype=A.dtype,device=A.device).unsqueeze(0).expand_as(A)
    Z = torch.eye(dim,dtype=A.dtype,device=A.device).unsqueeze(0).expand_as(A)

    for i in range(numIters):
        T = 0.5*(3.0*I - Z.bmm(Y))
        Y = Y.bmm(T)
        Z = T.bmm(Z)
    #A_sqrt = Y*torch.sqrt(normA)
    A_isqrt = Z / torch.sqrt(normA)

    return A_isqrt



#deconvolve channels
class ChannelDeconv(nn.Module):
    def __init__(self,  block, eps=1e-2,n_iter=5,momentum=0.1,sampling_stride=3):
        super(ChannelDeconv, self).__init__()

        self.eps = eps
        self.n_iter=n_iter
        self.momentum=momentum
        self.block = block

        self.register_buffer('running_mean1', torch.zeros(block, 1))
        #self.register_buffer('running_cov', torch.eye(block))
        self.register_buffer('running_deconv', torch.eye(block))
        self.register_buffer('running_mean2', torch.zeros(1, 1))
        self.register_buffer('running_var', torch.ones(1, 1))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.sampling_stride=sampling_stride
    def forward(self, x):
        x_shape = x.shape
        if len(x.shape)==2:
            x=x.view(x.shape[0],x.shape[1],1,1)
        if len(x.shape)==3:
            print('Error! Unsupprted tensor shape.')

        N, C, H, W = x.size()
        B = self.block

        #take the first c channels out for deconv
        c=int(C/B)*B
        if c==0:
            print('Error! block should be set smaller.')

        #step 1. remove mean
        if c!=C:
            x1=x[:,:c].permute(1,0,2,3).contiguous().view(B,-1)
        else:
            x1=x.permute(1,0,2,3).contiguous().view(B,-1)

        if self.sampling_stride > 1 and H >= self.sampling_stride and W >= self.sampling_stride:
            x1_s = x1[:,::self.sampling_stride**2]
        else:
            x1_s=x1

        mean1 = x1_s.mean(-1, keepdim=True)

        if self.num_batches_tracked==0:
            self.running_mean1.copy_(mean1.detach())
        if self.training:
            self.running_mean1.mul_(1-self.momentum)
            self.running_mean1.add_(mean1.detach()*self.momentum)
        else:
            mean1 = self.running_mean1

        x1=x1-mean1

        #step 2. calculate deconv@x1 = cov^(-0.5)@x1
        if self.training:
            cov = x1_s @ x1_s.t() / x1_s.shape[1] + self.eps * torch.eye(B, dtype=x.dtype, device=x.device)
            deconv = isqrt_newton_schulz_autograd(cov, self.n_iter)

        if self.num_batches_tracked==0:
            #self.running_cov.copy_(cov.detach())
            self.running_deconv.copy_(deconv.detach())

        if self.training:
            #self.running_cov.mul_(1-self.momentum)
            #self.running_cov.add_(cov.detach()*self.momentum)
            self.running_deconv.mul_(1 - self.momentum)
            self.running_deconv.add_(deconv.detach() * self.momentum)
        else:
            # cov = self.running_cov
            deconv = self.running_deconv

        x1 =deconv@x1

        #reshape to N,c,J,W
        x1 = x1.view(c, N, H, W).contiguous().permute(1,0,2,3)

        # normalize the remaining channels
        if c!=C:
            x_tmp=x[:, c:].view(N,-1)
            if self.sampling_stride > 1 and H>=self.sampling_stride and W>=self.sampling_stride:
                x_s = x_tmp[:, ::self.sampling_stride ** 2]
            else:
                x_s = x_tmp

            mean2=x_s.mean()
            var=x_s.var()

            if self.num_batches_tracked == 0:
                self.running_mean2.copy_(mean2.detach())
                self.running_var.copy_(var.detach())

            if self.training:
                self.running_mean2.mul_(1 - self.momentum)
                self.running_mean2.add_(mean2.detach() * self.momentum)
                self.running_var.mul_(1 - self.momentum)
                self.running_var.add_(var.detach() * self.momentum)
            else:
                mean2 = self.running_mean2
                var = self.running_var

            x_tmp = (x[:, c:] - mean2) / (var + self.eps).sqrt()
            x1 = torch.cat([x1, x_tmp], dim=1)


        if self.training:
            self.num_batches_tracked.add_(1)

        if len(x_shape)==2:
            x1=x1.view(x_shape)
        return x1

#An alternative implementation
class Delinear(nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True, eps=1e-5, n_iter=5, momentum=0.1, block=512):
        super(Delinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()



        if block > in_features:
            block = in_features
        else:
            if in_features%block!=0:
                block=math.gcd(block,in_features)
                print('block size set to:', block)
        self.block = block
        self.momentum = momentum
        self.n_iter = n_iter
        self.eps = eps
        self.register_buffer('running_mean', torch.zeros(self.block))
        self.register_buffer('running_deconv', torch.eye(self.block))


    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):

        if self.training:

            # 1. reshape
            X=input.view(-1, self.block)

            # 2. subtract mean
            X_mean = X.mean(0)
            X = X - X_mean.unsqueeze(0)
            self.running_mean.mul_(1 - self.momentum)
            self.running_mean.add_(X_mean.detach() * self.momentum)

            # 3. calculate COV, COV^(-0.5), then deconv
            # Cov = X.t() @ X / X.shape[0] + self.eps * torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
            Id = torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
            Cov = torch.addmm(self.eps, Id, 1. / X.shape[0], X.t(), X)
            deconv = isqrt_newton_schulz_autograd(Cov, self.n_iter)
            # track stats for evaluation
            self.running_deconv.mul_(1 - self.momentum)
            self.running_deconv.add_(deconv.detach() * self.momentum)

        else:
            X_mean = self.running_mean
            deconv = self.running_deconv

        w = self.weight.view(-1, self.block) @ deconv
        if self.bias is None:
            b = - (w @ (X_mean.unsqueeze(1))).view(self.weight.shape[0], -1).sum(1)
        else:
            b = self.bias - (w @ (X_mean.unsqueeze(1))).view(self.weight.shape[0], -1).sum(1)

        w = w.view(self.weight.shape)
        return F.linear(input, w, b)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )



class FastDeconv(conv._ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,groups=1,bias=True, eps=1e-5, n_iter=5, momentum=0.1, block=64, sampling_stride=3,freeze=False,freeze_iter=100):
        self.momentum = momentum
        self.n_iter = n_iter
        self.eps = eps
        self.counter=0
        self.track_running_stats=True
        super(FastDeconv, self).__init__(
            in_channels, out_channels,  _pair(kernel_size), _pair(stride), _pair(padding), _pair(dilation),
            False, _pair(0), groups, bias, padding_mode='zeros')

        if block > in_channels:
            block = in_channels
        else:
            if in_channels%block!=0:
                block=math.gcd(block,in_channels)

        if groups>1:
            #grouped conv
            block=in_channels//groups

        self.block=block

        self.num_features = kernel_size**2 *block
        # for debug
        #print(f'[debug] self.num_features={self.num_features}')
        #exit()

        if groups==1:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_deconv', torch.eye(self.num_features))
        else:
            self.register_buffer('running_mean', torch.zeros(kernel_size ** 2 * in_channels))
            self.register_buffer('running_deconv', torch.eye(self.num_features).repeat(in_channels // block, 1, 1))

        self.sampling_stride=sampling_stride*stride
        self.counter=0
        self.freeze_iter=freeze_iter
        self.freeze=freeze

    def forward(self, x):
        N, C, H, W = x.shape
        B = self.block
        frozen=self.freeze and (self.counter>self.freeze_iter)
        if self.training and self.track_running_stats:
            self.counter+=1
            self.counter %= (self.freeze_iter * 10)

        if self.training and (not frozen):

            # 1. im2col: N x cols x pixels -> N*pixles x cols
            if self.kernel_size[0]>1:
                X = torch.nn.functional.unfold(x, self.kernel_size,self.dilation,self.padding,self.sampling_stride).transpose(1, 2).contiguous()
            else:
                #channel wise
                X = x.permute(0, 2, 3, 1).contiguous().view(-1, C)[::self.sampling_stride**2,:]

            if self.groups==1:
                # (C//B*N*pixels,k*k*B)
                X = X.view(-1, self.num_features, C // B).transpose(1, 2).contiguous().view(-1, self.num_features)
            else:
                X=X.view(-1,X.shape[-1])

            # 2. subtract mean
            X_mean = X.mean(0)
            X = X - X_mean.unsqueeze(0)

            # 3. calculate COV, COV^(-0.5), then deconv
            if self.groups==1:
                #Cov = X.t() @ X / X.shape[0] + self.eps * torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
                Id=torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
                Cov = torch.addmm(self.eps, Id, 1. / X.shape[0], X.t(), X)
                deconv = isqrt_newton_schulz_autograd(Cov, self.n_iter)
            else:
                X = X.view(-1, self.groups, self.num_features).transpose(0, 1)
                Id = torch.eye(self.num_features, dtype=X.dtype, device=X.device).expand(self.groups, self.num_features, self.num_features)
                Cov = torch.baddbmm(self.eps, Id, 1. / X.shape[1], X.transpose(1, 2), X)

                deconv = isqrt_newton_schulz_autograd_batch(Cov, self.n_iter)

            if self.track_running_stats:
                self.running_mean.mul_(1 - self.momentum)
                self.running_mean.add_(X_mean.detach() * self.momentum)
                # track stats for evaluation
                self.running_deconv.mul_(1 - self.momentum)
                self.running_deconv.add_(deconv.detach() * self.momentum)

        else:
            X_mean = self.running_mean
            deconv = self.running_deconv

        #4. X * deconv * conv = X * (deconv * conv)
        if self.groups==1:
            w = self.weight.view(-1, self.num_features, C // B).transpose(1, 2).contiguous().view(-1,self.num_features) @ deconv
            b = self.bias - (w @ (X_mean.unsqueeze(1))).view(self.weight.shape[0], -1).sum(1)
            w = w.view(-1, C // B, self.num_features).transpose(1, 2).contiguous()
        else:
            w = self.weight.view(C//B, -1,self.num_features)@deconv
            b = self.bias - (w @ (X_mean.view( -1,self.num_features,1))).view(self.bias.shape)

        w = w.view(self.weight.shape)
        x= F.conv2d(x, w, b, self.stride, self.padding, self.dilation, self.groups)

        return x

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class DehazeBlock(nn.Module):
    def __init__(self, conv, dim, kernel_size, norm=False):
        super(DehazeBlock, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)
        self.norm = norm

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res


class DCNBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DCNBlock, self).__init__()
        self.dcn = DCN(in_channel, out_channel, kernel_size=(3,3), stride=1, padding=1).cuda()
    def forward(self, x):
        return self.dcn(x)


class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out


class AECRNet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, use_dropout=False, padding_type='reflect', feat_cache=False):
        super(AECRNet, self).__init__()

        ###### downsample
        self.down1 = nn.Sequential(nn.ReflectionPad2d(3),
                                   nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                                   nn.ReLU(True))
        self.down2 = nn.Sequential(nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))
        self.down3 = nn.Sequential(nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))

        ###### FFA blocks
        self.block = DehazeBlock(default_conv, ngf * 4, 3)

        ###### upsample
        self.up1 = nn.Sequential(nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(ngf*2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up3 = nn.Sequential(nn.ReflectionPad2d(3),
                                 nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                 nn.Tanh())


        self.dcn_block = DCNBlock(256, 256)

        self.deconv = FastDeconv(3, 3, kernel_size=3, stride=1, padding=1)

        self.mix1 = Mix(m=-1)
        self.mix2 = Mix(m=-0.6)
        self.feat_cache = feat_cache

    def forward(self, input):
        feat_cache = []

        x_deconv = self.deconv(input) # preprocess

        x_down1 = self.down1(x_deconv) # [bs, 64, 256, 256]
        x_down2 = self.down2(x_down1) # [bs, 128, 128, 128]
        x_down3 = self.down3(x_down2) # [bs, 256, 64, 64]

        x1 = self.block(x_down3)
        feat_cache.append(x1)
        x2 = self.block(x1)
        feat_cache.append(x2)
        x3 = self.block(x2)
        feat_cache.append(x3)
        x4 = self.block(x3)
        feat_cache.append(x4)
        x5 = self.block(x4)
        feat_cache.append(x5)
        x6 = self.block(x5)
        feat_cache.append(x6)

        x_dcn1 = self.dcn_block(x6)
        x_dcn2 = self.dcn_block(x_dcn1)

        x_out_mix = self.mix1(x_down3, x_dcn2)
        x_up1 = self.up1(x_out_mix) # [bs, 128, 128, 128]
        x_up1_mix = self.mix2(x_down2, x_up1)
        x_up2 = self.up2(x_up1_mix) # [bs, 64, 256, 256] 
        out = self.up3(x_up2) # [bs,  3, 256, 256]

        # return out, feat_cache
        return out


def adaINv3(content, style, version=8):
    # tmp = adaptive_instance_normalization(fis[i], fss[i], 1, 1)
    if isinstance(version, int):
        if version == 1:
            res = adaptive_instance_normalization(content, style, 1, 1)
        elif version == 7:
            res = adaptive_instance_normalizationv7(content, style)
        elif version == 8:
            res = adaptive_instance_normalizationv8(content, style)
        elif version == 9:
            res = adaptive_instance_normalizationv9(content, style)
        elif version == 10:
            res = adaptive_instance_normalizationv10(content, style)
        elif version == 11:
            res = adaptive_instance_normalizationv11(content, style)
        elif version == 12:
            res = adaptive_instance_normalizationv12(content, style)
        elif version == 13:
            res = adaptive_instance_normalizationv13(content, style)
        elif version == 15:
            res = adaptive_instance_normalizationv15(content, style)
        elif version == 83:
            res = adaptive_instance_normalizationv83(content, style)
        else:
            tmp_m, tmp_s = calc_mean_std(content)
            new_tmp_m = tmp_m.clone()
            new_tmp_s = tmp_s.clone()
            m_index = tmp_m > 0
            if version == 101:
                new_tmp_m[m_index] = tmp_m[m_index] - 0.01
            elif version == 102:
                new_tmp_m[m_index] = tmp_m[m_index] + 0.01
            elif version == 103:
                new_tmp_m[~m_index] = tmp_m[~m_index] - 0.01
            elif version == 104:
                new_tmp_m[~m_index] = tmp_m[~m_index] + 0.01
            elif version == 105:
                new_tmp_s = new_tmp_s + 0.01
            else:
                new_tmp_s = new_tmp_s * 0.9
            res = (content - tmp_m) / tmp_s * new_tmp_s + new_tmp_m
    else:
        res = adaptive_instance_normalization_pttd(content, style, *version)
    return res


class AECRNetDouble(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, use_dropout=False, padding_type='reflect'):
        super(AECRNetDouble, self).__init__()

        ###### downsample
        self.down1 = nn.Sequential(nn.ReflectionPad2d(3),
                                   nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                                   nn.ReLU(True))
        self.down2 = nn.Sequential(nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))
        self.down3 = nn.Sequential(nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))

        ###### FFA blocks
        self.block = DehazeBlock(default_conv, ngf * 4, 3, norm=False)

        ###### upsample
        self.up1 = nn.Sequential(nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(ngf*2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up3 = nn.Sequential(nn.ReflectionPad2d(3),
                                 nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                 nn.Tanh())


        self.dcn_block = DCNBlock(256, 256)

        self.deconv = FastDeconv(3, 3, kernel_size=3, stride=1, padding=1)

        self.mix1 = Mix(m=-1)
        self.mix2 = Mix(m=-0.6)

    def forward(self, x, y, mode):
        x_deconv = self.deconv(x) # preprocess
        y_deconv = self.deconv(y)
        # x_deconv = adaIN(x_deconv, mode)
        x_down1 = self.down1(x_deconv) # [bs, 64, 256, 256]
        y_down1 = self.down1(y_deconv) # [bs, 64, 256, 256]
        x_down1 = adaINv3(x_down1, y_down1, mode)
        x_down2 = self.down2(x_down1) # [bs, 128, 128, 128]
        y_down2 = self.down2(y_down1) # [bs, 128, 128, 128]
        x_down2 = adaINv3(x_down2, y_down2, mode)
        x_down3 = self.down3(x_down2) # [bs, 256, 64, 64]
        y_down3 = self.down3(y_down2) # [bs, 256, 64, 64]
        x_down3 = adaINv3(x_down3, y_down3, mode)

        x1 = self.block(x_down3)
        y1 = self.block(y_down3)
        x1 = adaINv3(x1, y1, mode)
        x2 = self.block(x1)
        y2 = self.block(y1)
        x2 = adaINv3(x2, y2, mode)
        x3 = self.block(x2)
        y3 = self.block(y2)
        x3 = adaINv3(x3, y3, mode)
        x4 = self.block(x3)
        y4 = self.block(y3)
        x4 = adaINv3(x4, y4, mode)
        x5 = self.block(x4)
        y5 = self.block(y4)
        x5 = adaINv3(x5, y5, mode)
        x6 = self.block(x5)
        y6 = self.block(y5)
        x6 = adaINv3(x6, y6, mode)

        x_dcn1 = self.dcn_block(x6)
        y_dcn1 = self.dcn_block(y6)
        # x_dcn1 = adaINv3(x_dcn1, y_dcn1, mode)
        x_dcn2 = self.dcn_block(x_dcn1)
        y_dcn2 = self.dcn_block(y_dcn1)
        # x_dcn2 = adaINv3(x_dcn2, y_dcn2, mode)

        x_out_mix = self.mix1(x_down3, x_dcn2)
        y_out_mix = self.mix1(y_down3, y_dcn2)
        # x_out_mix = adaINv3(x_out_mix, y_out_mix, mode)
        x_up1 = self.up1(x_out_mix) # [bs, 128, 128, 128]
        y_up1 = self.up1(y_out_mix) # [bs, 128, 128, 128]
        # x_up1 = adaINv3(x_up1, y_up1, mode)
        x_up1_mix = self.mix2(x_down2, x_up1)
        y_up1_mix = self.mix2(y_down2, y_up1)
        # x_up1_mix = adaINv3(x_up1_mix, y_up1_mix, mode)
        x_up2 = self.up2(x_up1_mix) # [bs, 64, 256, 256] 
        y_up2 = self.up2(y_up1_mix) # [bs, 64, 256, 256] 
        # x_up2 = adaINv3(x_up2, y_up2, mode)
        x_out = self.up3(x_up2) # [bs,  3, 256, 256]
        y_out = self.up3(y_up2) # [bs,  3, 256, 256]
        # out = adaIN(out, 1)
        return x_out, y_out


class AECRNetStyle(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, use_dropout=False, padding_type='reflect'):
        super(AECRNetStyle, self).__init__()

        ###### downsample
        self.down1 = nn.Sequential(nn.ReflectionPad2d(3),
                                   nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                                   nn.ReLU(True))
        self.down2 = nn.Sequential(nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))
        self.down3 = nn.Sequential(nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))

        ###### FFA blocks
        self.block = DehazeBlock(default_conv, ngf * 4, 3, norm=False)

        ###### upsample
        self.up1 = nn.Sequential(nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(ngf*2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up3 = nn.Sequential(nn.ReflectionPad2d(3),
                                 nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                 nn.Tanh())


        self.dcn_block = DCNBlock(256, 256)

        self.deconv = FastDeconv(3, 3, kernel_size=3, stride=1, padding=1)

        self.mix4 = Mix(m=-1)
        self.mix5 = Mix(m=-0.6)

    def forward(self, input, mode):

        x_deconv = self.deconv(input) # preprocess
        # x_deconv = adaIN(x_deconv, mode)
        x_down1 = self.down1(x_deconv) # [bs, 64, 256, 256]
        x_down1 = adaIN(x_down1, mode)
        x_down2 = self.down2(x_down1) # [bs, 128, 128, 128]
        x_down2 = adaIN(x_down2, mode)
        x_down3 = self.down3(x_down2) # [bs, 256, 64, 64]
        x_down3 = adaIN(x_down3, mode)

        x1 = self.block(x_down3)
        x1 = adaIN(x1, mode)
        x2 = self.block(x1)
        x2 = adaIN(x2, mode)
        x3 = self.block(x2)
        x3 = adaIN(x3, mode)
        x4 = self.block(x3)
        x4 = adaIN(x4, mode)
        x5 = self.block(x4)
        x5 = adaIN(x5, mode)
        x6 = self.block(x5)
        feat_cache = x6
        x6 = adaIN(x6, mode)

        x_dcn1 = self.dcn_block(x6)
        # x_dcn1 = adaIN(x_dcn1, mode)
        x_dcn2 = self.dcn_block(x_dcn1)
        # x_dcn2 = adaIN(x_dcn2, mode)

        x_out_mix = self.mix4(x_down3, x_dcn2)
        # x_out_mix = adaIN(x_out_mix, mode)
        x_up1 = self.up1(x_out_mix) # [bs, 128, 128, 128]
        # x_up1 = adaIN(x_up1, mode)
        x_up1_mix = self.mix5(x_down2, x_up1)
        # x_up1_mix = adaIN(x_up1_mix, 7)
        x_up2 = self.up2(x_up1_mix) # [bs, 64, 256, 256] 
        # x_up2 = adaIN(x_up2, mode)
        out = self.up3(x_up2) # [bs,  3, 256, 256]
        # out = adaIN(out, 1)
        return out


class AECRNetStyleV2(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, use_dropout=False, padding_type='reflect'):
        super(AECRNetStyleV2, self).__init__()

        ###### downsample
        self.down1 = nn.Sequential(nn.ReflectionPad2d(3),
                                   nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                                   nn.ReLU(True))
        self.down2 = nn.Sequential(nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))
        self.down3 = nn.Sequential(nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))

        ###### FFA blocks
        self.block = DehazeBlock(default_conv, ngf * 4, 3)

        ###### upsample
        self.up1 = nn.Sequential(nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(ngf*2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up3 = nn.Sequential(nn.ReflectionPad2d(3),
                                 nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                 nn.Tanh())


        self.dcn_block = DCNBlock(256, 256)

        self.deconv = FastDeconv(3, 3, kernel_size=3, stride=1, padding=1)

        self.mix1 = Mix(m=-1)
        self.mix2 = Mix(m=-0.6)

        # self.attn1_mean = nn.Parameter(torch.cuda.FloatTensor(256).fill_(-5))
        # self.attn2_mean = nn.Parameter(torch.cuda.FloatTensor(256).fill_(-5))
        # self.attn3_mean = nn.Parameter(torch.cuda.FloatTensor(256).fill_(-5))
        # self.attn4_mean = nn.Parameter(torch.cuda.FloatTensor(256).fill_(-5))
        # self.attn5_mean = nn.Parameter(torch.cuda.FloatTensor(256).fill_(-5))
        # self.attn6_mean = nn.Parameter(torch.cuda.FloatTensor(256).fill_(-5))
        # self.attn1_std = nn.Parameter(torch.cuda.FloatTensor(256).fill_(-5))
        # self.attn2_std = nn.Parameter(torch.cuda.FloatTensor(256).fill_(-5))
        # self.attn3_std = nn.Parameter(torch.cuda.FloatTensor(256).fill_(-5))
        # self.attn4_std = nn.Parameter(torch.cuda.FloatTensor(256).fill_(-5))
        # self.attn5_std = nn.Parameter(torch.cuda.FloatTensor(256).fill_(-5))
        # self.attn6_std = nn.Parameter(torch.cuda.FloatTensor(256).fill_(-5))

        self.norm_num = 256

        self.attn1_mean = nn.Conv2d(self.norm_num, self.norm_num, 1, 1, 0, groups=self.norm_num, bias=False)
        nn.init.zeros_(self.attn1_mean.weight)
        self.attn2_mean = nn.Conv2d(self.norm_num, self.norm_num, 1, 1, 0, groups=self.norm_num, bias=False)
        nn.init.zeros_(self.attn2_mean.weight)
        self.attn3_mean = nn.Conv2d(self.norm_num, self.norm_num, 1, 1, 0, groups=self.norm_num, bias=False)
        nn.init.zeros_(self.attn3_mean.weight)
        self.attn4_mean = nn.Conv2d(self.norm_num, self.norm_num, 1, 1, 0, groups=self.norm_num, bias=False)
        nn.init.zeros_(self.attn4_mean.weight)
        self.attn5_mean = nn.Conv2d(self.norm_num, self.norm_num, 1, 1, 0, groups=self.norm_num, bias=False)
        nn.init.zeros_(self.attn5_mean.weight)
        self.attn6_mean = nn.Conv2d(self.norm_num, self.norm_num, 1, 1, 0, groups=self.norm_num, bias=False)
        nn.init.zeros_(self.attn6_mean.weight)
        self.attn1_std = nn.Conv2d(self.norm_num, self.norm_num, 1, 1, 0, groups=self.norm_num, bias=False)
        nn.init.zeros_(self.attn1_std.weight)
        self.attn2_std = nn.Conv2d(self.norm_num, self.norm_num, 1, 1, 0, groups=self.norm_num, bias=False)
        nn.init.zeros_(self.attn2_std.weight)
        self.attn3_std = nn.Conv2d(self.norm_num, self.norm_num, 1, 1, 0, groups=self.norm_num, bias=False)
        nn.init.zeros_(self.attn3_std.weight)
        self.attn4_std = nn.Conv2d(self.norm_num, self.norm_num, 1, 1, 0, groups=self.norm_num, bias=False)
        nn.init.zeros_(self.attn4_std.weight)
        self.attn5_std = nn.Conv2d(self.norm_num, self.norm_num, 1, 1, 0, groups=self.norm_num, bias=False)
        nn.init.zeros_(self.attn5_std.weight)
        self.attn6_std = nn.Conv2d(self.norm_num, self.norm_num, 1, 1, 0, groups=self.norm_num, bias=False)
        nn.init.zeros_(self.attn6_std.weight)
    

    def IN(self, x, attn_mean, attn_std, mean, std, screen=False):
        x_mean, x_std = calc_mean_std(x)
        x_norm = (x - x_mean) / x_std

        # score_mean = sigmoid(attn_mean).unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3)
        # score_std = sigmoid(attn_std).unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3)
        # new_mean = score_mean * x_mean +(1 - score_mean) * mean
        # new_std = score_std * x_std +(1 - score_std) * std
        # res = new_std * x_norm + new_mean

        x_mean_parts = torch.split(x_mean, self.norm_num, dim=1)
        mean_parts = torch.split(mean, self.norm_num, dim=1)
        x_std_parts = torch.split(x_mean, self.norm_num, dim=1)
        std_parts = torch.split(std, self.norm_num, dim=1)
        means = torch.concat([x_mean_parts[0], mean_parts[0]], dim=0).squeeze()
        stds = torch.concat([x_std_parts[0], std_parts[0]], dim=0).squeeze()
        means = Rearrange('a b -> b a')(means)
        stds = Rearrange('a b -> b a')(stds)
        means = Rearrange('b a -> (b a)')(means).unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3)
        stds = Rearrange('b a -> (b a)')(stds).unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3)

        # means = x_mean
        # stds = x_std

        sigmoid = nn.Sigmoid()
        # relu = nn.ReLU()
        # attn_mean_weight = sigmoid(attn_mean.weight)
        # attn_std_weight = sigmoid(attn_std.weight)
        attn_mean_weight = 1. / (1. + torch.exp(-100. * attn_mean.weight))
        attn_std_weight = 1. / (1. + torch.exp(-100. * attn_std.weight))
        attn_mean_weight = torch.concat([attn_mean_weight, 1 - attn_mean_weight], dim=1)
        attn_std_weight = torch.concat([attn_std_weight, 1 - attn_std_weight], dim=1)
        new_mean = nn.functional.conv2d(input=means, weight=attn_mean_weight, stride=1, padding=0, groups=self.norm_num)
        # new_mean = x_mean
        new_std = nn.functional.conv2d(input=stds, weight=attn_std_weight, stride=1, padding=0, groups=self.norm_num)
        # new_std = x_std
        new_mean = torch.concat([new_mean, *x_mean_parts[1:]], dim=1)
        new_std = torch.concat([new_std, *x_std_parts[1:]], dim=1)

        # attn_mean2 = attn_mean(means)
        # attn_std2 = attn_std(stds)
        # new_mean = attn_mean2 * x_mean + (1 - attn_mean2) * mean
        # new_std = attn_std2 * x_std + (1 - attn_std2) * std

        # new_mean = attn_mean(means)
        # new_std = relu(attn_std(stds))
        if screen:
            print('attn_mean: {}'.format(attn_mean_weight.squeeze()))
            # print('attn_std: {}'.format(attn_std_weight.squeeze()))
            # print('x_mean: {}'.format(torch.sum(x_mean)))
            # print('new_mean: {}'.format(torch.sum(new_mean)))
            # print('x_std: {}'.format(torch.sum(x_std)))
            # print('new_std: {}'.format(torch.sum(new_std)))
        # print(x_mean)
        # print(mean)
        # print(new_mean)
        res = new_std * x_norm + new_mean
        return res

    def forward(self, input, mean_cache, std_cache):

        feat_cache = []

        x_deconv = self.deconv(input) # preprocess
        x_down1 = self.down1(x_deconv) # [bs, 64, 256, 256]
        # x_down1 = adaIN(x_down1)
        x_down2 = self.down2(x_down1) # [bs, 128, 128, 128]
        # x_down2 = adaIN(x_down2)
        x_down3 = self.down3(x_down2) # [bs, 256, 64, 64]
        # x_down3 = adaIN(x_down3)

        x1 = self.block(x_down3)
        feat_cache.append(x1)
        x1 = self.IN(x1, self.attn1_mean, self.attn1_std, mean_cache[0], std_cache[0])
        # feat_cache.append(x1)

        x2 = self.block(x1)
        x2 = self.IN(x2, self.attn2_mean, self.attn2_std, mean_cache[1], std_cache[1])
        x3 = self.block(x2)
        x3 = self.IN(x3, self.attn3_mean, self.attn3_std, mean_cache[2], std_cache[2])
        x4 = self.block(x3)
        x4 = self.IN(x4, self.attn4_mean, self.attn4_std, mean_cache[3], std_cache[3])
        x5 = self.block(x4)
        x5 = self.IN(x5, self.attn5_mean, self.attn5_std, mean_cache[4], std_cache[4])
        x6 = self.block(x5)
        x6 = self.IN(x6, self.attn6_mean, self.attn6_std, mean_cache[5], std_cache[5])

        x_dcn1 = self.dcn_block(x6)
        # x_dcn1 = adaIN(x_dcn1, 1)
        x_dcn2 = self.dcn_block(x_dcn1)
        # x_dcn2 = adaIN(x_dcn2, 1)

        x_out_mix = self.mix1(x_down3, x_dcn2)
        # x_out_mix = adaIN(x_out_mix, 7)
        x_up1 = self.up1(x_out_mix) # [bs, 128, 128, 128]
        # x_up1 = adaIN(x_up1, 7)
        x_up1_mix = self.mix2(x_down2, x_up1)
        # x_up1_mix = adaIN(x_up1_mix, 7)
        x_up2 = self.up2(x_up1_mix) # [bs, 64, 256, 256] 
        # x_up2 = adaIN(x_up2, 1)
        out = self.up3(x_up2) # [bs,  3, 256, 256]
        # out = adaIN(out, 1)
        return out, feat_cache