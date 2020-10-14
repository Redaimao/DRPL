# -*- coding:utf-8 -*-
# sliding patch for testing then get average for pixel value
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


def sliding_patch_preds(net, image, h_s, w_s, k_size):
    #b, c, h, w = image.size()
    c, h, w = image.size()
    print(c,h,w)
    arr_cont = np.zeros((1,h,w),dtype=np.float32) # to record counts of patch for each pixel
    arr_value = np.zeros((1,h,w),dtype=np.float32)

    # img_seq = []  # img1 patch sequence
    # coor_seq = []  # left-up coordinates of each pach
    # print(img1.size())
    # print(img2.size())
    for x in range(0, h , h_s):
        for y in range(0, w, w_s):
            print(np.clip((x + k_size),0,h), np.clip((y + k_size),0,w))
            sub_img = image[:, x:np.clip((x + k_size),0,h), y:np.clip((y + k_size),0,w)]  # slicing
            #sub_img = net(sub_img)
            arr_value[:, x:np.clip((x + k_size),0,h), y:np.clip((y + k_size),0,w)] += sub_img[0,:,:].cpu().data.numpy() #sub_img.cpu().data.numpy() #
            arr_cont[:, x:np.clip((x + k_size),0,h), y:np.clip((y + k_size),0,w)] += 1
    print(arr_value)
    arr_value = arr_value/arr_cont

    print(arr_cont)
    print(arr_value)
    return arr_value





if __name__ == '__main__':
    img1 = Variable(torch.rand(3, 8, 8))
    img2 = Variable(torch.rand(3, 8, 8))

    if torch.cuda.is_available():
        img1 = img1.cuda()
        img2 = img2.cuda()

    arr = sliding_patch_preds(None, img1, 2, 2, 3)
