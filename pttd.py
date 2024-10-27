import os, sys, math, glob, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import pyiqa
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DataParallel
from torchvision.transforms import ToTensor, Normalize, Grayscale, CenterCrop, RandomCrop, ToPILImage
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as cal_psnr
from skimage.metrics import structural_similarity as cal_ssim

from model import *
from utils import *


def pttd_dehaze(net, img, sample, style_adain='norm', mode=7, alpha_mean=1., alpha_std=1., sample_size=None, limit=False):
    B, C, H, W = img.shape
    patch_size = W // 10
    if style_adain == 'norm':
        prompt = adaptive_instance_normalization(sample, img, alpha_mean, alpha_std) 
    elif style_adain == 'patch_norm':
        prompt = patch_adaIN(sample, img, patch_size, patch_size, alpha_mean, alpha_std, limit=limit)
    elif style_adain == 'onechannel_patch_norm':
        prompt = patch_adaIN(sample, img, patch_size, patch_size, alpha_mean, alpha_std, limit=limit, onechannel=True)
    elif style_adain == 'onechannel_norm':
        prompt = adaptive_instance_normalization(sample, img, alpha_mean, alpha_std, limit=limit, onechannel=True) 
    else:
        prompt = sample
    if sample_size is not None:
        prompt = F.interpolate(prompt, size=sample_size, mode='bicubic')
    prompt = prompt.clamp(0, 1)
    img_outs = net(pad_img(img, 16), pad_img(prompt, 16), mode)
    return img_outs, prompt


parser = argparse.ArgumentParser()
parser.add_argument('--source', default='RIDCP', type=str, help='source domain')
parser.add_argument('--model', default='AECRNet', type=str, help='model name')
parser.add_argument('--pretrained_path', default='pretrained_checkpoints', type=str, help='pretrained model dir')
parser.add_argument('--ys', default='ys/0543.jpg', type=str, help='sampled image (ys in the paper) from the GTs of the source dataset')
parser.add_argument('--input', default='', type=str, help='input dir')
parser.add_argument('--output', default='./results', type=str, help='output dir')
parser.add_argument('--format', default='', type=str, help='image format')
parser.add_argument('--device', default='cuda', type=str, help='test device')
parser.add_argument('--save_all', default=False, action='store_true', help='wether save all output results (w/ CB & w/o CB)')
args = parser.parse_args()

model_list = {
    'AECRNet' : ['AECRNet(3, 3)', 'AECRNetDouble(3, 3)'],
}

to_tensor = ToTensor()

# hyper-parameter
MOS = 0.005

if __name__ == '__main__':

    pttd_config = read_yaml('pttd_config.yaml')

    pretrained_weight_name = '{}_{}.pth'.format(args.model, args.source)

    dehaze_net_pttd = eval(model_list[args.model][1])
    dehaze_net_pttd.to(args.device)
    for p in dehaze_net_pttd.parameters():
        p.requires_grad = False
    dehaze_net_pttd = DataParallel(dehaze_net_pttd)
    dehaze_net_pttd.load_state_dict(torch.load(os.path.join(args.pretrained_path, pretrained_weight_name))['state_dict'])
    dehaze_net_pttd.eval()

    if args.save_all:
        dehaze_net = eval(model_list[args.model][0])
        dehaze_net.to(args.device)
        for p in dehaze_net.parameters():
            p.requires_grad = False
        dehaze_net = DataParallel(dehaze_net)
        dehaze_net.load_state_dict(torch.load(os.path.join(args.pretrained_path, pretrained_weight_name))['state_dict'])
        dehaze_net.eval()

    if os.path.isdir(args.input):
        hazy_list = glob.glob(os.path.join(args.input, '*.{}*'.format(args.format)))
    else:
        hazy_list = glob.glob(args.input)
    hazy_list.sort()

    if not os.path.exists(args.output):
        os.mkdir(args.output)
    
    sample = Image.open(args.ys).convert('RGB')
    sample_W, sample_H = sample.size
    for fp in tqdm(hazy_list):
        hazy_name = fp.split('/')[-1].split('.png')[0]
        hazy = Image.open(fp).convert('RGB')
        hazy_hsv = Image.open(fp).convert('HSV')
        sample_resize = sample.resize(hazy.size)
        hazy = to_tensor(hazy).unsqueeze(dim=0).cuda()
        hazy_hsv = to_tensor(hazy_hsv).unsqueeze(dim=0).cuda()
        sample_t = to_tensor(sample_resize).unsqueeze(dim=0).cuda()
        B, C, H, W = hazy.shape

        hazy_np = hazy.clamp(0, 1).squeeze().cpu().numpy().transpose(1, 2, 0)
        _, hazy_density, _, dcp_out = DCP(hazy_np)
        hazy_density = torch.tensor(hazy_density, dtype=torch.float32).unsqueeze(dim=0).unsqueeze(dim=1).cuda()
        mos_hazy = cal_mos(hazy_hsv, hazy_density, 'avg', 'hazy', False)

        if args.save_all:
            try:
                img_out0 = dehaze_net(pad_img(hazy, 16))[:, :, :H, :W]
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, skip image {}'.format(hazy_name))
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

            try:
                img_outs, prompt1 = pttd_dehaze(dehaze_net_pttd, hazy, sample_t, 'patch_norm', [pttd_config, args], sample_size=[sample_H, sample_W])
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, skip image {}'.format(hazy_name))
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            img_out1, _ = img_outs
            img_out1 = img_out1[:, :, :H, :W]

            try:
                img_outs, prompt2 = pttd_dehaze(dehaze_net_pttd, hazy, sample_t, 'onechannel_patch_norm', [pttd_config, args], sample_size=[sample_H, sample_W])
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, skip image {}'.format(hazy_name))
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            img_out2, _ = img_outs
            img_out2 = img_out2[:, :, :H, :W]

            if mos_hazy >= 0.005:
                img_out_final = img_out1
            else:
                img_out_final = img_out2

            # save_image(hazy, os.path.join(args.output, '{}_hazy.png'.format(hazy_name, args.model)))
            save_image(img_out0, os.path.join(args.output, '{}_{}.png'.format(hazy_name, args.model)))
            save_image(img_out1, os.path.join(args.output, '{}_{}-PTTD1.png'.format(hazy_name, args.model)))
            save_image(img_out2, os.path.join(args.output, '{}_{}-PTTD2.png'.format(hazy_name, args.model)))
            save_image(img_out_final, os.path.join(args.output, '{}_{}-PTTD.png'.format(hazy_name, args.model)))
            # save_image(prompt1, os.path.join(args.output, '{}_{}-prompt1.png'.format(hazy_name, args.model)))
            # save_image(prompt2, os.path.join(args.output, '{}_{}-prompt2.png'.format(hazy_name, args.model)))
        else:
            if mos_hazy >= 0.005:
                img_outs, _ = pttd_dehaze(dehaze_net_pttd, hazy, sample_t, 'patch_norm', [pttd_config, args], sample_size=[sample_H, sample_W])
                img_out_final, _ = img_outs
                img_out_final = img_out_final[:, :, :H, :W]
            else:
                img_outs, _ = pttd_dehaze(dehaze_net_pttd, hazy, sample_t, 'onechannel_patch_norm', [pttd_config, args], sample_size=[sample_H, sample_W])
                img_out_final, _ = img_outs
                img_out_final = img_out_final[:, :, :H, :W]
            save_image(img_out_final, os.path.join(args.output, '{}_{}-PTTD.png'.format(hazy_name, args.model)))