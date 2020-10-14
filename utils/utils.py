# coding: utf-8
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import pickle as pkl
import cv2

class MyDataset(Dataset):
    def __init__(self, path, transform = None, target_transform = None):
        with open(path,'rb') as file:
            imgs = pkl.load(file)

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn = self.imgs[index]
        img = Image.open('./data/images/'+fn).convert('RGB')
        # print('img.size',img.size)
        label = fn.split('.')[0].split('_')[-1] #groundtruth name
        gt = cv2.imread('./data/template2/' + str(label) + '.bmp') #read groundtruth
        gt = np.fabs(np.float32((cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY) // 255.0)[np.newaxis,:,:])) #convert to binaray gray

        if self.transform is not None:
            img = self.transform(img)

        return img, gt, label

    def __len__(self):
        return len(self.imgs)

class MyDataset_Gradient(Dataset):
    def __init__(self, path, transform = None, target_transform = None):
        with open(path,'rb') as file:
            imgs = pkl.load(file)

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        p1 = self.imgs[index][0]
        p2 = self.imgs[index][1]
        pgt = '_'.join(p2.split('_')[1:5]) + '.jpg'

        img1 = Image.open('./data/images/' + p1).convert('RGB')
        img2 = Image.open('./data/images/' + p2).convert('RGB')
        gtimg = Image.open('./data/raw/' + pgt).convert('RGB')

        label1 = p1.split('.')[0].split('_')[-1] #groundtruth name
        label2 = p2.split('.')[0].split('_')[-1] #groundtruth name
        gt1 = Image.open('./data/template2/' + str(label1) + '.bmp')
        gt2 = Image.open('./data/template2/' + str(label2) + '.bmp')

        if self.transform is not None:
            #print('self.transform',self.transform)
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            gt1 = self.transform(gt1)[0,:,:].unsqueeze(0).clamp_(0,1)
            gt2 = self.transform(gt2)[0,:,:].unsqueeze(0).clamp_(0,1)
            gtimg = self.transform(gtimg)

        return img1,img2,gt1,gt2,label1,label2,gtimg

    def __len__(self):
        return len(self.imgs)

class MyTestDataset(Dataset):
    def __init__(self, path, pkfile, transform = None, target_transform = None):
        imgs = []  #for each element represent a pair tuple

        with open(pkfile,'rb') as file:
            imgs = pkl.load(file)
        print(imgs)
        print(len(imgs))
        self.imgs = imgs
        # print(self.imgs)
        self.transform = transform
        self.target_transform = target_transform
        self.path = path

    def __getitem__(self, index):
        fn = self.imgs[index]

        img1 = Image.open(self.path +'/' + fn[0]).convert('RGB')
        img2 = Image.open(self.path +'/' + fn[1]).convert('RGB')

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1,img2,fn[0],fn[1]

    def __len__(self):
        return len(self.imgs)


def normalize_invert(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor