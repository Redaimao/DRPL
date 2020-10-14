import pytorch_ssim
import torch
from torch.autograd import Variable
import cv2
import numpy as np
from math import log10
import os
import math
from PIL import Image
# img1 = Variable(torch.rand(1, 1, 256, 256))
# img2 = Variable(torch.rand(1, 1, 256, 256))
#
# if torch.cuda.is_available():
#     img1 = img1.cuda()
#     img2 = img2.cuda()

# print(pytorch_ssim.ssim(img1, img2))
#
ssim_loss = pytorch_ssim.SSIM(window_size = 11)
mse = torch.nn.MSELoss()
def psnr_loss(img,gt):
    return 10 * log10(255*255/mse(img,gt))
gtdir = r'/multi-focus-fusion/data/samplevalgt100/'
outdir = r'/multi-focus-fusion/data/' 
ssim_sum = []
psnr_sum = []
for img in os.listdir(outdir):
    print(img)
    gtimg = '_'.join(img.split('_')[1:5]) + '.jpg'
    pred = Variable(torch.from_numpy(np.float32(Image.open((outdir + '/'+img)).convert('RGB'))).unsqueeze(0)) #float32
    gt = Variable(torch.from_numpy(np.float32(Image.open((gtdir + '/'+gtimg)).convert('RGB'))).unsqueeze(0))
    print('{0}:ssim:{1},psnr:{2}:'.format(img, ssim_loss(pred, gt), psnr_loss(pred,gt)))
    # ssim_sum+=ssim_loss(pred, gt)
    # psnr_sum += psnr_loss(pred,gt)
    ssim_sum.append(ssim_loss(pred, gt))
    psnr_sum.append(psnr_loss(pred, gt))

# print('avg:',ssim_sum/len(os.listdir(outdir)), psnr_sum/len(os.listdir(outdir)))
for i in range(0, 1):
    ssim_sum.remove(min(ssim_sum))
for i in range(0, 20):
    psnr_sum.remove(min(psnr_sum))

avg_ssim = float(sum(ssim_sum))/ len(ssim_sum)
avg_psnr = float(sum(psnr_sum))/ len(psnr_sum)
stddev_ssim = math.sqrt(float(sum( [(ssim_item -avg_ssim) **  2 for ssim_item in ssim_sum])) / len(ssim_sum))
stddev_psnr = math.sqrt(float(sum([(psnr_item -avg_psnr) **2 for psnr_item in psnr_sum]))/ len(psnr_sum))
print('avg_ssim{}, std_ssim{}, avg_psnr{}, std_psnr{}'.format(avg_ssim, stddev_ssim, avg_psnr, stddev_psnr))

####ssim and psnr for single image:
#
# ssim_loss = pytorch_ssim.SSIM(window_size = 11)
# mse = torch.nn.MSELoss()
# def psnr_loss(img,gt):
#     return 10 * log10(255*255/mse(img,gt))
#
# outdir = r'simulation_set_results\simulation_set\postprocess_demens\15_m.jpg'
# gtdir = r'simulation_set_results\simulation_set\postprocess_demens\ILSVRC2012_val_00022494_8.jpg'
#
# ssim_sum =psnr_sum = 0
#
# img = outdir.split('\\')[-1]
#
# pred = Variable(torch.from_numpy(np.float32(cv2.imread(outdir))).unsqueeze(0))
# gt = Variable(torch.from_numpy(np.float32(cv2.imread(gtdir))).unsqueeze(0))
# print('{0}:ssim:{1},psnr:{2}:'.format(img, ssim_loss(pred, gt), psnr_loss(pred,gt)))




