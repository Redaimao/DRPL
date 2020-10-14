# coding: utf-8


import torch
import argparse
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import numpy as np
import os
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
sys.path.append(".")
from utils.utils import MyDataset, MyTestDataset
from tensorboardX import SummaryWriter
from datetime import datetime
# from net import MemNet as Net
from net import Generator as Generator
import cv2
import time
#----------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# parser.add_argument('--testfile', type=str, default='./data/sample_val_100.pkl', help='test image list') #sample_val_100 new_val_3k
# parser.add_argument('--test_path', type=str, default='./data/sampleval100/', help='test image file') #sampleval100 images_prev
parser.add_argument('--testfile', type=str, default='./data/lytro_pairs.pkl', help='test image list')
parser.add_argument('--test_path', type=str, default='./data/lytro', help='test image file')
# parser.add_argument('--testfile', type=str, default='./data/disp100.pkl', help='test image list')
# parser.add_argument('--test_path', type=str, default='./data/dispimages/', help='test image file')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--imageSize', type=int, default=520, help='the low resolution image size')
parser.add_argument('--nGPU', type=int, default=2, help='number of GPUs to use')
parser.add_argument('--load_ckpt', type=str, default=None, help="path to finetune weights (to continue training)")
parser.add_argument('--out', type=str, default='./data/', help='output image dir')
parser.add_argument('--thread', type=float, default=0.5, help='thread for mask')

opt = parser.parse_args()

#-----------------------------output file-----------------------------------------------------------

# ./data/12-25_12-00-15_model_20/
result_dir = opt.out + opt.load_ckpt.split('/')[-2] + '_model_' + opt.load_ckpt.split('/')[-1].split('_')[0]+ '/'+ opt.test_path.split('/')[-1] +'/'
# result_dir = r'/data/guoxiaobao/Project/multi-focus-fusion/data/simul-convention/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)


##------------------------check if cudnn and cuda are avaliable-----------
cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# device = torch.device("cuda:0,1" if opt.cuda else "cpu")
ngpu = int(opt.nGPU)
# ------------------------------------ step 1/5 ------------------------------------

trainTransform = transforms.Compose([
    transforms.ToTensor()
])

validTransform = transforms.Compose([
    transforms.ToTensor()
])

test_data = MyTestDataset(path=opt.test_path,pkfile = opt.testfile, transform=trainTransform)
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)


# ------------------------------------ step 2/5 ------------------------------------
n_res_blk = 12
net = Generator(n_res_blk)
print(net)

def _load_ckpt(model, ckpt):
    load_dict = {}
    for name in ckpt:
        load_dict[name.split('module.')[1]]= ckpt[name]
    model.load_state_dict(load_dict, strict=False)
    return model

if opt.load_ckpt is not None:
    # load params
    pretrained_dict = torch.load(opt.load_ckpt)
    net_state_dict = net.state_dict()
    pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
    net_state_dict.update(pretrained_dict_1)
    net.load_state_dict(net_state_dict)
else: #train from scratch
    raise Exception('none model!! set --load_ckpt')

if opt.cuda:
    net = torch.nn.DataParallel(net)
    net = net.cuda()
# -----------------------------step 3/5:define post processing---------------------------------

def postmerge(output1, output2, img1name, img2name,thread = 0.5):
    im1 = cv2.imread(opt.test_path + '/' +img1name)
    im2 = cv2.imread(opt.test_path + '/' + img2name)
    # im1 = cv2.imread(result_dir + '/' +img1name)
    # im2 = cv2.imread(result_dir + '/' + img2name)

    output1 = (output1 + 1 -output2) / 2
    #output1 = np.where(output1 >= thread, 1, 0).repeat(3, -1)#0210：non-binaray
    #output1 = np.sign((output1 + 1 - output2) / 2)
    output2 = 1 - output1
   # output2 = np.where(output2 >= thread, 1, 0).repeat(3,-1)
    #print(output1)
    res = im1 * output1 + im2 * output2
    return  res
# def postmerge_single(output1, img1name, img2name,thread = 0.5):
#     im1 = cv2.imread(opt.test_path + '/' +img1name)
#     im2 = cv2.imread(opt.test_path + '/' + img2name)
#
#     # output1 = (output1 + 1 -output2) / 2
#     #output1 = np.where(output1 >= thread, 1, 0).repeat(3, -1)#0210：non-binaray
#     #output1 = np.sign((output1 + 1 - output2) / 2)
#     output2 = 1 - output1
#    # output2 = np.where(output2 >= thread, 1, 0).repeat(3,-1)
#     #print(output1)
#     res = im1 * output1 + im2 * output2
#     return  res

def sliding_patch_preds(net, image, h_s, w_s, k_size):
    #b, c, h, w = image.size()
    # print(image.size())
    b, c, h, w = image.size()
    # print(image.size())
    arr_cont = np.zeros((b,1,h,w),dtype=np.float32) # to record counts of patch for each pixel
    arr_value = np.zeros((b,1,h,w),dtype=np.float32)
    for x in range(0, h, h_s):
        for y in range(0, w , w_s):
            sub_img = image[:,:, x:np.clip((x + k_size),0,h), y:np.clip((y + k_size),0,w)]  # slicing
            sub_img = net(sub_img)

            arr_value[:,:, x:np.clip((x + k_size),0,h), y:np.clip((y + k_size),0,w)] += sub_img.cpu().data.numpy()
            arr_cont[:,:, x:np.clip((x + k_size),0,h), y:np.clip((y + k_size),0,w)] += 1

    arr_value = arr_value/arr_cont
    return arr_value
# ------------------------------------ step 4/5  --------------------------------------------------



net.eval()
for i, data in enumerate(test_loader):

    image1, image2, img1name, img2name = data
    image1, image2 = Variable(image1.cuda()), Variable(image2.cuda())
   # print('image1size:',image1.size(),type(image1)) #image1size: torch.Size([1, 3, 520, 520]) <class 'torch.Tensor'>
    mergeimg = torch.cat((image1,image2),1)
    img1name = img1name[0]
    img2name = img2name[0]
   # print(img1name[0], img2name[0])
    mergename = img1name.split('.')[0] + '_m.jpg'  #id.jpg
    print(mergename)


    #tic = time.time()
    output = net(mergeimg)
    # print('mergeshape:',output.shape)
    # print(img1name.shape)
    # output = output.squeeze().cpu().data.numpy()[:,:,np.newaxis]
    # result_img = postmerge_single(output, img1name, img2name, thread = opt.thread)
    output1 = output[:,0,:,:].squeeze(0).cpu().data.numpy()[:,:,np.newaxis]
    output2 = output[:,1,:,:].squeeze(0).cpu().data.numpy()[:,:,np.newaxis]

    result_img = postmerge(output1, output2, img1name, img2name, thread = opt.thread)
    # toc = time.time()
    # print('time:',toc - tic)

 # ------------------------------------ step5/5------------------------------------
 #    result_dir = r"./data/new_val_3k_result/"
 #    if not os.path.exists(result_dir):
 #        os.makedirs(result_dir)
    cv2.imwrite(result_dir + '/' + mergename, result_img)
    maskdir = result_dir + '/mask/'
    if not os.path.exists(maskdir):
        os.makedirs(maskdir)
    # cv2.imwrite(maskdir + str(img1name), output.repeat(3,-1)*255) #single

    cv2.imwrite(maskdir + str(img1name), output1.repeat(3,-1)*255)
    cv2.imwrite(maskdir + str(img2name), output2.repeat(3,-1)*255)
    cv2.imwrite(maskdir + str(mergename), (output1 + output2).repeat(3,-1)*255)

print('Finished testing')

