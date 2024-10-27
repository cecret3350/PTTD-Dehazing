import math
import cv2
import yaml
import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F

from torch import nn
from PIL import Image
from torchvision.utils import save_image
from cv2.ximgproc import guidedFilter


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = -1
        self.min = 10000

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if val > self.max:
            self.max = val
        if val < self.min:
            self.min = val
    
    def get_max(self):
         return self.max


def read_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f) 


def white_balance(img):
    K = torch.mean(img, dim=[1, 2, 3])

    tmp = torch.mean(img, dim=[2, 3], keepdim=True)
    mr, mg, mb = torch.split(tmp, 1, dim=1)

    gr = K / mr
    gg = K / mg
    gb = K / mb

    res = torch.zeros_like(img)
    res[:, 0, :, :] = img[:, 0, :, :] * gr
    res[:, 1, :, :] = img[:, 1, :, :] * gg
    res[:, 2, :, :] = img[:, 2, :, :] * gb

    return res


def pad_img(x, patch_size):
    _, _, h, w = x.size()
    mod_pad_h = (patch_size - h % patch_size) % patch_size
    mod_pad_w = (patch_size - w % patch_size) % patch_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x


def pad_imgv2(x, crop_size, crop_step):
    _, _, h, w = x.size()
    h -= crop_size
    w -= crop_size
    mod_pad_h = (crop_step - h % crop_step) % crop_step
    mod_pad_w = (crop_step - w % crop_step) % crop_step
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x


def crop_parallel(img, crop_sz, step):
    b, c, h, w = img.shape
    h_space = np.arange(0, h - crop_sz + 1, step)
    w_space = np.arange(0, w - crop_sz + 1, step)
    index = 0
    num_h = 0
    lr_list=torch.Tensor().to(img.device)
    for x in h_space:
        num_h += 1
        num_w = 0
        for y in w_space:
            num_w += 1
            index += 1
            crop_img = img[:, :, x:x + crop_sz, y:y + crop_sz]
            lr_list = torch.cat([lr_list, crop_img])
    new_h=x + crop_sz # new height after crop
    new_w=y + crop_sz # new width  after crop
    return lr_list, num_h, num_w, new_h, new_w


def combine_parallel(sr_list, num_h, num_w, h, w, patch_size, step):
    index=0
    sr_img = torch.zeros((1, 3, h, w)).to(sr_list.device)
    for i in range(num_h):
        for j in range(num_w):
            sr_img[:, :, i*step:i*step+patch_size, j*step:j*step+patch_size] += sr_list[index]
            index+=1

    # mean the overlap region
    for j in range(1,num_w):
        sr_img[:, :, :, j*step:j*step+(patch_size-step)]/=2
    for i in range(1,num_h):
        sr_img[:, :, i*step:i*step+(patch_size-step), :]/=2

    return sr_img


def white_balance(img):
    K = torch.mean(img, dim=[1, 2, 3])

    tmp = torch.mean(img, dim=[2, 3], keepdim=True)
    mr, mg, mb = torch.split(tmp, 1, dim=1)

    gr = K / mr
    gg = K / mg
    gb = K / mb

    res = torch.zeros_like(img)
    res[:, 0, :, :] = img[:, 0, :, :] * gr
    res[:, 1, :, :] = img[:, 1, :, :] * gg
    res[:, 2, :, :] = img[:, 2, :, :] * gb

    return res


def DarkChannel(im,sz):
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    return dark

def AtmLight(im,dark):
	[h,w] = im.shape[:2]
	imsz = h*w
	numpx = int(max(math.floor(imsz/1000),1))
	darkvec = dark.reshape(imsz)
	imvec = im.reshape(imsz,3)
	indices = darkvec.argsort()
	indices = indices[imsz-numpx::]
	atmsum = np.zeros([1,3])
	for ind in range(1,numpx):
		atmsum = atmsum + imvec[indices[ind]]
	A = atmsum / numpx
	return A


def TransmissionEstimate(im,A,sz):
    omega = 0.95
    im3 = np.empty(im.shape,im.dtype)

    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/(A[0,ind] + 1e-7)

    transmission = 1 - omega*DarkChannel(im3,sz)
    return transmission


def Guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r))
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r))
    cov_Ip = mean_Ip - mean_I*mean_p

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r))
    var_I   = mean_II - mean_I*mean_I

    a = cov_Ip/(var_I + eps)
    b = mean_p - a*mean_I

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r))
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r))

    q = mean_a*im + mean_b
    return q


def TransmissionRefine(im,et):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)/255
    r = 60
    eps = 0.0001
    t = Guidedfilter(gray,et,r,eps)
    return t


def Recover(im,t,A,tx = 0.1):
    res = np.empty(im.shape,im.dtype)
    t = cv2.max(t,tx)

    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]

    return res


def guided_filter(img, coarse):
    res = Guidedfilter(img, coarse, 60, 0.0001)
    return res


def DCP(I, A=None):
    dark = DarkChannel(I,15)
    if A is None:
        A = AtmLight(I,dark)
    te = TransmissionEstimate(I,A,15)
    src = np.uint8(I * 255)
    t = TransmissionRefine(src,te)
    J = Recover(I,t,A,0.1)
    return te, t, A, J


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.contiguous().view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.contiguous().view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat, alpha_mean=1, alpha_std=1, limit=False, onechannel=False):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    # style_mean = torch.tensor([0.6906, 0.6766, 0.6749]).unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3).cuda()
    # style_std = torch.tensor([0.1955, 0.2096, 0.2236]).unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3).cuda()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)

    # if limit:
        # style_std, _ = torch.max(torch.concat([content_std, style_std], dim=0), dim=0, keepdim=True) 
        # style_mean, _ = torch.max(torch.concat([content_mean, style_mean], dim=0), dim=0, keepdim=True) 
        # style_std, _ = torch.min(torch.concat([content_std, style_std], dim=0), dim=0, keepdim=True) 
        # pass
    if onechannel:
        style_mean = torch.mean(style_mean)
        style_std = torch.mean(style_std)

    style_mean = alpha_mean * style_mean + (1 - alpha_mean) * content_mean
    style_std = alpha_std * style_std + (1 - alpha_std) * content_std
    res = normalized_feat * style_std.expand(size) + style_mean.expand(size)

    # if limit:
    #     t = style_std / content_std
    #     A = (content_std * style_mean - style_std * content_mean) / (content_std - style_std)
    #     t_mean = torch.mean(t)
    #     A_mean = torch.mean(A)
    #     if t_mean > 0 and t_mean < 1 and A_mean >0 and A_mean < 1:
    #         res = t_mean * content_feat + (1 - t_mean) * A_mean
    #     else:
    #         res = torch.ones_like(content_feat)
    # else:
        # res = normalized_feat * style_std.expand(size) + style_mean.expand(size)
    return res


def patch_adaIN(content, style, patch_size=64, step=64, alpha_mean=1., alpha_std=1., limit=False, onechannel=False):
    B, C, H, W = content.shape
    content = pad_imgv2(content, patch_size, step)
    style = pad_imgv2(style, patch_size, step)
    content_patches, num_h, num_w, new_h, new_w = crop_parallel(content, patch_size, step)
    style_patches, num_h, num_w, new_h, new_w = crop_parallel(style, patch_size, step)
    content_patches_list = torch.split(content_patches, 1, dim=0)
    # for i, tmp in enumerate(content_patches_list):
    #     save_image(tmp, 'content_{}.png'.format(i + 1))
    style_patches_list = torch.split(style_patches, 1, dim=0)
    # for i, tmp in enumerate(style_patches_list):
    #     save_image(tmp, 'style_{}.png'.format(i + 1))

    norm_content_patches_list = []
    for x, y in zip(content_patches_list, style_patches_list):
        x = adaptive_instance_normalization(x, y, alpha_mean, alpha_std, limit, onechannel)
        norm_content_patches_list.append(x)
    # for i, tmp in enumerate(norm_content_patches_list):
    #     save_image(tmp, 'norm_{}.png'.format(i + 1))

    # for i in range(len(content_patches_list)):
    #     save_image(content_patches_list[i], 'results/content_{}.png'.format(i+1))
    #     save_image(style_patches_list[i], 'results/style_{}.png'.format(i+1))
    #     save_image(norm_content_patches_list[i], 'results/norm_{}.png'.format(i+1))
    
    norm_content_patches = torch.concat(norm_content_patches_list, dim=0) 
    res = combine_parallel(norm_content_patches, num_h, num_w, new_h, new_w, patch_size, step)
    res = res[:, :, :H, :W]
    return res


def adaptive_instance_normalization_pttd(content_feat, style_feat, config, args):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)

    mean_mode = config[args.source][args.model]['mean']
    std_mode = config[args.source][args.model]['std']
    p = config[args.source][args.model]['p']
    assert mean_mode in ['min', 'abs', 'adaIN'], 'wrong mean mode! (must be min or abs or adaIN)'
    assert std_mode in ['min', 'max', 'adaIN'], 'wrong std mode! (must be min or max or adaIN)'

    if mean_mode == 'min':
        index = (((content_mean > style_mean) & (content_mean * style_mean > 0))).type(torch.float32)
    elif mean_mode == 'abs':
        index = (((torch.abs(content_mean) > torch.abs(style_mean)) & (content_mean * style_mean > 0))).type(torch.float32)
    else:
        index = 1
    new_mean = index * style_mean + (1 - index) * content_mean

    if std_mode == 'min':
        new_std, index = torch.min(torch.concat([content_std, style_std], dim=0), dim=0, keepdim=True) 
    elif std_mode == 'max':
        new_std, index = torch.max(torch.concat([content_std, style_std], dim=0), dim=0, keepdim=True) 
    else:
        new_std = style_std

    content_std_mean = torch.mean(content_std.squeeze())
    content_std_std = torch.std(content_std.squeeze())
    # mask = ((style_std >= (content_std_mean + 2 *content_std_std)) | (style_std <= (content_std_mean - 0.0 *content_std_std))).type(torch.float32)
    if std_mode == 'min':
        mask = ((new_std <= (content_std_mean - p * content_std_std))).type(torch.float32)
    elif std_mode == 'max':
        mask = ((new_std >= (content_std_mean + p * content_std_std))).type(torch.float32)
    else:
        mask = 0
    new_std = mask * content_std + (1 - mask) * new_std
    # print('2:', new_mean.mean())

    res = normalized_feat * new_std + new_mean
    return res


def kmeans(img): 
    data = img.reshape((-1, 1))
    data = np.float32(data)
    
    K = 3
    type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    max_iter = 20
    epsilon = 0.1
    criteria = (type, max_iter, epsilon)
    attempts = 10
    # flags = cv2.KMEANS_RANDOM_CENTERS
    flags = cv2.KMEANS_PP_CENTERS

    _, labels, centers = cv2.kmeans(data, K, None, criteria, attempts, flags)
    centers = centers.squeeze()
    res = centers[labels.flatten()].reshape(img.shape)
    res = torch.tensor(res, dtype=torch.float32).unsqueeze(dim=0).unsqueeze(dim=1).cuda()
    res_list = []
    centers_sort = np.sort(centers)
    for i in centers_sort:
        res_list.append((res == i).type(torch.float32))
    return res_list


def cal_mos(img, d=None, th=0.5, mode='hazy', save_mask=False):
    # d means hazy density
    if d is not None:
        d = (d - d.min()) / (d.max() - d.min())
        if th == 'kmeans':
            mask_list = kmeans(d) 
            if mode == 'hazy':
                mask = mask_list[0]
                for i in range(1, len(mask_list) - 1):
                    mask = mask + mask_list[i]
            else:
                mask = mask_list[-1]
        elif th == 'avg':
            if mode == 'hazy':
                mask = (d <= d.mean()).type(torch.float32)
            else:
                mask = (d >= d.mean()).type(torch.float32)
        else:
            if mode == 'hazy':
                mask = (d <= th).type(torch.float32)
            else:
                mask = (d >= th).type(torch.float32)
        if save_mask:
            save_image(torch.tensor(d, dtype=torch.float32), 'density.png')
            save_image(mask, 'mask_{}_{}.png'.format(th, mode))
        h, s, v = torch.split(img, 1, dim=1)
        mean = torch.sum(h * mask, dim=[2, 3], keepdim=True) / torch.sum(mask, dim=[2, 3], keepdim=True)
        h1 = (mean - h * mask) ** 2
        h2 = (1 - torch.abs(mean - h * mask)) ** 2
        h3, _ = torch.min(torch.concat([h1, h2], dim=1), dim=1, keepdim=True)
        mos = torch.sum(h3 * mask, dim=[2, 3], keepdim=True) / torch.sum(mask, dim=[2, 3], keepdim=True)
    else:
        h, s, v = torch.split(img, 1, dim=1)
        mean = torch.mean(h, dim=[2, 3], keepdim=True)
        h1 = (mean - h) ** 2
        h2 = (1 - torch.abs(mean - h)) ** 2
        h3, _ = torch.min(torch.concat([h1, h2], dim=1), dim=1, keepdim=True)
        mos = torch.mean(h3, dim=[2, 3], keepdim=True)
    return mos