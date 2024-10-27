import os, glob, argparse
import pyiqa
import torch
import cv2

from tqdm import tqdm
from torchvision.transforms import ToTensor
from skimage.metrics import peak_signal_noise_ratio as val_psnr
from skimage.metrics import structural_similarity as val_ssim
from PIL import Image

from utils import AverageMeter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', default='./results', type=str, help='results directory')
    parser.add_argument('--gt_dir', default='png', type=str, help='GTs directory')
    args = parser.parse_args()
    to_tensor = ToTensor()

    out_list = glob.glob(os.path.join(args.result_dir, '*AECRNet-PTTD.png'))
    out_list.sort()

    psnrs = AverageMeter()
    ssims = AverageMeter()

    for out_path in tqdm(out_list):
        name = out_path.split('/')[-1].split('_AECRNet-PTTD.png')[0] + '.png'
        out = cv2.imread(out_path) / 255.
        gt = cv2.imread(os.path.join(args.gt_dir, name)) / 255.

        psnr_tmp = val_psnr(gt, out)
        ssim_tmp = val_ssim(gt, out, win_size=11, data_range=1., channel_axis=2, gaussian_weights=True)
        psnrs.update(psnr_tmp)
        ssims.update(ssim_tmp)
    
    print('PSNR: {}, SSIM: {}'.format(psnrs.avg, ssims.avg))