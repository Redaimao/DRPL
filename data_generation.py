# -*- coding:utf-8 -*-
import numpy as np
import cv2
import os
import pickle as pkl
from PIL.Image import Image
import  matplotlib.pyplot as plt
from shutil import copyfile
import random

#global threshold
def threshold_demo(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  #to binary
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
    print(binary.shape)
    print("threshold value %s"%ret)
    cv2.namedWindow("binary0", cv2.WINDOW_NORMAL)
    cv2.imshow("binary0", binary)
    cv2.waitKey(0)
    return binary

def crop_img_percent(imgdir,hper,wper,savedir):
    imname = imgdir.split('/')[-1].split('.')[0]
    print(imname)
    img = cv2.imread(imgdir)
    h,w,c = img.shape
    nh, nw = int(h * (1-hper)//2), int(w * (1-wper)//2)
    hcen, wcen = int(h//2), int(w//2)
    nimg = img[hcen-nh:hcen+nh,wcen-nw:wcen+nw]
    cv2.imwrite(savedir + '/' + imname + '.jpg', nimg)
#
# def crop_box(image,height,width,dir):
#     h = height//2  #half the height and width
#     w = width//2
#
#     # bin_region = threshold_demo(region) #binary
#     # print(bin_region)
#     cv2.imwrite(dir + '\\' + str(cont) + '.jpg', region)  
#     reverse_mask = ~region // 255  # mask reverse
#     cv2.imwrite(dir + '\\' + 're' + str(cont) + '.jpg', reverse_mask*255) 
#     # print('bin_region:',bin_region.shape)
#     # bin_region, contours, hierarchy = cv2.findContours(bin_region, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     # print(contours)
#     # imag = cv2.drawContours(region,contours,1,(0,255,0),1)
#     # cv2.imshow('imag',imag)
#     # cv2.waitKey(0)
#     # cv2.imshow('region',region)
#     # cv2.waitKey(0)
#    # boxes.append(box)
# #print(boxes)

def pair_generation(imgdir, maskdir, remaskdir,savedir,noisetype ='GaussianBlur'):
    # given a mask and an image, resize image to the size of mask,and save image, traversing all the pixels in image,if
    # the coordinates of image in mask ==1,then
    mask_name =  maskdir.split('/')[-1].split('.')[0]
    img_name = imgdir.split('/')[-1].split('.')[0]
    pair_name = '_'.join([img_name,mask_name])
    twin_pair_name = '_'.join([img_name,'re'+mask_name])
    blur_name = '_'.join([img_name,'g'])

    #print(img_name)
    mask = cv2.imread(maskdir) 
    reverse_mask = cv2.imread(remaskdir)
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)
    # print(mask.shape)#(256, 256, 3)
    mask = mask//255  
    reverse_mask = reverse_mask//255
    # print('mask',mask)
    #reverse_mask = ~mask//255  
    # print('reverse_mask',reverse_mask)
    # print(reverse_mask.shape)
    img = cv2.imread(imgdir) 
    print('img',img.shape)
    w = img.shape[0] // 2
    h = img.shape[1] // 2
    print(w ,h)
    if img.shape[0] <= 256 or img.shape[1] <=256:
        return
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    img = img[w-128:w+128,h-128:h+128] #crop from center
    print(img.shape)
    # if img.shape[0] <= 256 or img.shape[1] <= 256:
    #     return
    #cv2.imwrite(savedir + 'raw/' + img_name + '.jpg',img)#save raw image
    blur = img
    if noisetype is 'GaussianBlur':
        blur = blur_mask(img)  # (src, ksize, sigmaX[, dst[, sigmaY[, borderType]]]) -> dst
    # cv2.imshow('blur', blur)
    # cv2.waitKey(0)
   # cv2.imwrite(savedir + 'blur/' + blur_name + '.jpg', blur)#save blur image
    #print(img.shape)
    clear_pair = img * mask #element wise product
    blur_pair = blur * reverse_mask
    pair = clear_pair + blur_pair
    # cv2.imshow('clear_pair', clear_pair)
    # cv2.waitKey(0)
    # cv2.imshow('blur_pair', blur_pair)
    # cv2.waitKey(0)
    # cv2.imshow('pair', pair)
    # cv2.waitKey(0)


    clear_pair = img * reverse_mask  # element wise product
    blur_pair = blur * mask
    twin_pair = clear_pair + blur_pair
    # cv2.imshow('clear_pair', clear_pair)
    # cv2.waitKey(0)
    # cv2.imshow('blur_pair', blur_pair)
    # cv2.waitKey(0)
    # cv2.imshow('twin_pair', twin_pair)
    # cv2.waitKey(0)

    print(savedir+'raw/'+img_name+'.jpg')
    cv2.imwrite(savedir+'raw/'+img_name+'.jpg' ,img) #save cropped image
    cv2.imwrite(savedir+'blur/'+blur_name+'.jpg', blur)  # save blur image
    cv2.imwrite(savedir+'images/'+pair_name +'.jpg', pair)  # save image pairs
    cv2.imwrite(savedir+'images/'+twin_pair_name+'.jpg', twin_pair)  # save twin pair images
    print('ff')
    return


def blur_mask(img):
   # kernel = cv2.getGaussianKernel(ksize= 5,sigma= 0) #(ksize, sigma[, ktype]) -> retval
    blur = cv2.GaussianBlur(img,(7,7),sigmaX=0,sigmaY=0)#(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]]) -> dst

    # In this approach, instead of a box filter consisting of equal filter coefficients,
    #  a Gaussian kernel is used. It is done with the function, cv2.GaussianBlur().
    #  We should specify the width and height of the kernel which should be positive and odd.
    #   We also should specify the standard deviation in the X and Y directions, sigmaX and sigmaY respectively.
    #   If only sigmaX is specified, sigmaY is taken as equal to sigmaX. If both are given as zeros,
    #   they are calculated from the kernel size. Gaussian filtering is highly effective in removing Gaussian noise from the image.

    return blur

def blur_img_lvl(img,lvl): #give blur according to level
    pass

def _rand_pair_generation(imgdir, blurimgdir,maskdir, remaskdir,savedir):
    # given a mask and an image, resize image to the size of mask,and save image, traversing all the pixels in image,if
    # the coordinates of image in mask ==1,then
    mask_name =  maskdir.split('/')[-1].split('.')[0]
    img_name = imgdir.split('/')[-1].split('.')[0]
    blurname = blurimgdir.split('/')[-1].split('_')[0]
    pair_name = '_'.join([blurname,img_name,mask_name])
    twin_pair_name = '_'.join([blurname,img_name,'re'+mask_name])

    mask = cv2.imread(maskdir) 
    reverse_mask = cv2.imread(remaskdir)
    mask =np.fabs(mask//255.0)  
    reverse_mask = np.fabs(reverse_mask//255.0)
    # print(mask)
    # print(mask[:,:,0].sum())
    # print(mask[:,:,1].sum())
    # print(mask[:,:,2].sum())
    img = cv2.imread(imgdir) 
    # print(img.shape)
    blur = cv2.imread(blurimgdir)
    # print(blur.shape)
    #print(blur)

    clear_pair = img * mask #element wise product
    blur_pair = blur * reverse_mask
    pair = clear_pair + blur_pair


    clear_pair = img * reverse_mask  # element wise product
    blur_pair = blur * mask
    twin_pair = clear_pair + blur_pair

    cv2.imwrite(savedir+'dispimages/'+pair_name +'.jpg', pair)  
    cv2.imwrite(savedir+'dispimages/'+twin_pair_name+'.jpg', twin_pair) 
    # print('done')
    return pair_name,twin_pair_name

#for final display,generate images of size 512
def crop_img_percent(imgdir,hper,wper,savedir):
    imname = imgdir.split('/')[-1].split('.')[0]
    print(imname)
    img = cv2.imread(imgdir)
    print(img.shape)
    h,w,c = img.shape
    # nh, nw = int(h * (1-hper)//2), int(w * (1-wper)//2)
    nh = hper//2
    nw = wper//2
    hcen, wcen = int(h//2), int(w//2)
    nimg = img[hcen-nh:hcen+nh,wcen-nw:wcen+nw]
    cv2.imwrite(savedir + '/' + imname + '.jpg', nimg)

def cut_nine_img(imgdir,savdir, t=256):
    imname = imgdir.split('/')[-1].split('.')[0]
    print(imgdir)
    print(imname)
    img = cv2.imread(imgdir)
    h,w,c = img.shape
    hcen, wcen = int(h // 2), int(w // 2)
    if h < t or w < t:
        return
    im1 = img[hcen - t//2: hcen, wcen - t//2: wcen]
    im2 = img[hcen - t//2: hcen, wcen - t//4: wcen + t//4]
    im3 = img[hcen - t//2: hcen, wcen : wcen + t//2]
    im4 = img[hcen - t//4: hcen + t//4, wcen : wcen + t//2]
    im5 = img[hcen : hcen + t//2, wcen : wcen + t//2]
    im6 = img[hcen : hcen + t//2, wcen - t//4 : wcen + t//4]
    im7 = img[hcen : hcen + t//2, wcen - t//2: wcen]
    im8 = img[hcen - t//4:hcen + t//4, wcen - t//2:wcen]
    im9 = img[hcen - t//4:hcen + t//4, wcen - t//4:wcen + t//4]

    # if h < 256 or w < 256:
    #     return
    # im1 = img[hcen - 128: hcen, wcen - 128: wcen]
    # im2 = img[hcen - 128: hcen, wcen - 64: wcen + 64]
    # im3 = img[hcen - 128: hcen, wcen: wcen + 128]
    # im4 = img[hcen - 64: hcen + 64, wcen: wcen + 128]
    # im5 = img[hcen: hcen + 128, wcen: wcen + 128]
    # im6 = img[hcen: hcen + 128, wcen - 64: wcen + 64]
    # im7 = img[hcen: hcen + 128, wcen - 128: wcen]
    # im8 = img[hcen - 64:hcen + 64, wcen - 128:wcen]
    # im9 = img[hcen - 64:hcen + 64, wcen - 64:wcen + 64]


    cv2.imwrite(savdir + imname + '_1.jpg', im1)
    cv2.imwrite(savdir + imname + '_2.jpg', im2)
    cv2.imwrite(savdir + imname + '_3.jpg', im3)
    cv2.imwrite(savdir + imname + '_4.jpg', im4)
    cv2.imwrite(savdir + imname + '_5.jpg', im5)
    cv2.imwrite(savdir + imname + '_6.jpg', im6)
    cv2.imwrite(savdir + imname + '_7.jpg', im7)
    cv2.imwrite(savdir + imname + '_8.jpg', im8)
    cv2.imwrite(savdir + imname + '_9.jpg', im9)






if __name__ == '__main__':
  #  dir = r'D:\programming_project\image_fusion_data\template'
    #dir1= r'D:\programming_project\image_fusion_data\template2'

    #reverse mask
    #
    # for i in range(1,5):
    #     img1 = cv2.imread(dir + '\\'+ str(i) + '.bmp')
    #     #img1 = img1.sum(-1).clip(0,255)
    #     img1 = cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY).clip(0,255)
    #     print(img1.shape)
    #
    #
    #     img1 = np.fabs((img1 //255.0)[:,:,np.newaxis].repeat(3,-1))
    #     #print(img1.shape)
    #     # print(img1)
    #     #print(img1.sum())
    #     # print('img1',img1[:,:,0])
    #     # print('img11', img1[:,:,1])
    #     # print('img12', img1[:,:,2])
    #     img2 = 1-img1
    #     # print(img2.shape)
    #     # print(img2.sum())
    #     # # print('mask',mask)
    #     # #reverse_img = ~img1 // 255  
    #     # # re
    #     # cv2.imshow('img', np.array(img1 *255))
    #     # cv2.waitKey(0)
    #     # cv2.imshow('img', np.array(img2 * 255))
    #     # cv2.waitKey(0)
    #     cv2.imwrite(dir1 + '\\' + str(i) + '.bmp', np.array(img1 * 255))
    #     cv2.imwrite(dir1 + '\\re' + str(i) + '.bmp',np.array(img2 * 255))


    # dir = r'D:\programming_project\image_fusion_data\all_results\disptemplate1'
    # for i in os.listdir(dir):
    #
    #     img = cv2.imread(dir + '\\' + i)  
    #     img = cv2.resize(img, (256, 256))#resize images to 256
    #     print(type(img))
    #     print(img.shape)
    #     print(img)
    #
    #     img = (img//255.0)[:,:,0][:,:,np.newaxis].repeat(3,-1)
    #     print(img.shape)
    #     img1 = 1 - img
    #     cv2.imwrite(dir + '\\' +i, img * 255)
    #     cv2.imwrite(dir + '\\re' + i, img1 * 255)


    # mdir = dir + 'raw/'
    # ndir = dir + 'select/'
    # # hper, wper = 0.15,0.15
    # for img in os.listdir(ndir):
    #     imgdir = ndir + img
    #     cut_nine_img(imgdir,mdir)
    # print('done')
    #     imgdir = mdir + '/' + img
    #     crop_img_percent(imgdir, hper, wper, ndir)


    #1.generate templates:

    #crop_box(img,contours[1],128,128,10,mdir) 
    #


#-----------------------
    # #2.combine templates,clear images and blur images to generate training and validation images pairs:
    # imgdir = r'./data/raw100/' #clear images
    # savedir =  r'./data/'  # root dir of data
    # blurimg = r'./data/blur100/' #blur images
    # mdir = r'./data/disptemplate/' #template dir
    #
    # imgnames = []
    # for imgs in os.listdir(imgdir):
    #     imgnames.append(imgs.split('.')[0].rstrip())
    # print(len(imgnames))#9360
    # print(imgnames[10])
    # #
    # templs = []
    # temll = []
    # for tem in os.listdir(mdir):
    #     if tem.split('re')[-1] not in templs:
    #         templs.append(tem.split('re')[-1])
    #         temll.append((tem.split('re')[-1],'re'+tem.split('re')[-1]))
    # print(len(templs))
    # print(len(temll))#213
    # print(temll)
    # #
    # cnt = 10#20 # number of images // number of template pairs
    # namlst = []
    # for i in range(0,len(temll)):
    #     maskdir = mdir + temll[i][0]
    #     remaskdir = mdir +temll[i][1]
    #     for k in range(i*cnt,(i+1)*cnt):
    #         imdir = imgdir + imgnames[k] + '.jpg'
    #         rls = [1,2,3]#np.random.randint(1, 4, 2)
    #         for j in rls:
    #             print('j', j)
    #             blurimgdir = blurimg + str(j) + '_blur_' + imgnames[k] + '.jpg'
    #             # print(blurimg)
    #             pair_name, twin_pair_name = _rand_pair_generation(imgdir=imdir, blurimgdir=blurimgdir, maskdir=maskdir,
    #                                                               remaskdir=remaskdir, savedir=savedir)
    #             namlst.append((pair_name + '.jpg',twin_pair_name + '.jpg'))
    # print(len(namlst))#27477
    # with open('./data/disp100.pkl', 'wb') as trainfile:
    #     pkl.dump(namlst,trainfile)

  #  --------------------------------
    #
    #
    # train_img =random.sample(namlst,20000)
    #
    # print(train_img[0])
    # with open('./data/train_pair_20k.pkl', 'wb') as trainfile:
    #     pkl.dump(train_img, trainfile)
    # val_img = []
    # for i in namlst:
    #     if i not in train_img:
    #         val_img.append(i)
    # print(len(val_img))
    # with open('./data/val_pair.pkl','wb') as valfile:
    #     pkl.dump(val_img,valfile)
    # print(val_img[0])
    # print(len(train_img),len(val_img))

    #3.generate 500 pairs of validation images to test
    #vali = []
    # with open('./data/val_pair.pkl','rb') as trainfile:
    #     vali = pkl.load(trainfile)
    # print(len(vali))
    # print(vali[0])
    #
    # nals = random.sample(vali,100)
    # with open('./data/sample_val_100.pkl','rb') as valfile:
    #         vali = pkl.load(valfile)
    # for item in vali:
    #     copyfile('./data/images/{}'.format(item[0]),'./data/sampleval100/{}'.format(item[0]))
    #     copyfile('./data/images/{}'.format(item[1]), './data/sampleval100/{}'.format(item[1]))
    #     gtname = '_'.join(item[0].split('_')[1:5]) + '.jpg'
    #     print(gtname)
    #     copyfile('./data/raw/{}'.format(gtname), './data/samplevalgt100/{}'.format(gtname))
    #########################


    # train = []
    # train2 = []
    # val = []
    # for i in range(0,20000):
    #     train.append(imgls[rands2[i]])
    # for k in range(20000,30000):
    #     train2.append(imgls[rands2[k]])
    # for j in rands1:
    #     val.append(imgls[j])
    # print(len(train))
    # print(len(val))
    #
    # with open('./data/train_pair_20k.pkl', 'wb') as trainfile:
    #     pkl.dump(train, trainfile)
    # with open('./data/train_pair_10k.pkl', 'wb') as trainfile1:
    #     pkl.dump(train2, trainfile1)
    # with open('./data/val_pair.pkl', 'wb') as valfile:
    #     pkl.dump(val, valfile)

    # imgs = []
    # im = []
    # with open('./data/train20k.pkl','rb') as file:
    #     imgs = pkl.load(file)
    #     print(len(imgs))
    #     print(imgs[2])
    # with open('./data/train7k.pkl','rb') as file:
    #     im = pkl.load(file)
    #     print(len(im))
    #     print(im[2])
    # imgs= [imgs[it] for it in range(0,len(imgs),2)]
    # im = [im[it] for it in range(0,len(im)-1096,3)]
    # im = im +imgs
    # print(len(im))
    # print(im[2])
    # with open('./data/train27k.pkl', 'wb') as valfile:
    #     pkl.dump(im, valfile)


    # imgs = []
    # with open('./data/val5k.pkl','rb') as file:
    #     imgs = pkl.load(file)
    #     print(len(imgs))
    #     print(imgs[5])
    # with open('./data/val5k.pkl', 'wb') as valfile:
    #     pkl.dump(imgs[:5000], valfile)

 # p1 = self.imgs[index][0]
 #        p2 = self.imgs[index][1]
 #        pgt = '_'.join(p2.split('_')[1:5]) + '.jpg'
 #
 #        img1 = Image.open('./data/images/' + p1).convert('RGB')
 #        img2 = Image.open('./data/images/' + p2).convert('RGB')
 #        gtimg = Image.open('./data/raw/' + pgt).convert('RGB')
 #    all_gts = []
 #    for im in os.listdir('./data/images_prev/'):
 #        all_gts.append(im)
 #    # print(all_gts)
 #
 #    with open('./data/train_pair_10k.pkl','rb') as file:  # [('lytro-15-A.jpg', 'lytro-15-B.jpg'), ('lytro-10-A.jpg', 'lytro-10-B.jpg'),
 #        imgs = pkl.load(file)
 #        # print(imgs)
 #        new_val = []
 #        from PIL import Image
 #        for tpl in imgs:
 #            p1 = tpl[0]
 #            p2 = tpl[1]
 #            print(p2)
 #            gt = '_'.join(p2.split('_')[1:5]) + '.jpg'
 #            if os.path.exists('./data/images_prev/' + str(p2)) and os.path.exists('./data/images_prev/' + str(tpl[0])):
 #
 #                try:
 #
 #
 #                   img1 = Image.open('./data/images_prev/' + p1).convert('RGB')
 #                   img2 = Image.open('./data/images_prev/' + p2).convert('RGB')
 #                   gtimg = Image.open('./data/raw/' + gt).convert('RGB')
 #                   print('true')
 #                   new_val.append(tpl)
 #                   if len(new_val) == 3000:
 #                       break
 #                except:
 #                    continue
 #
 #
 #        with open('./data/new_val_3k.pkl','wb') as wfile:
 #            pkl.dump(new_val, wfile)
 #
 #        # os.mkdir('./data/new_val_3k/')
 #    with open('./data/new_val_3k.pkl', 'rb') as rfile:
 #        imgs = pkl.load(rfile)
 #        print(imgs)



#test sesf method on simu-imgs
    # imgfile = r'D:\programming_project\image_fusion_data\all_results\showimg'
    #
    # imgls = []
    # for img in os.listdir(imgfile):
    #     if 're' not in img:
    #         imgls.append((img, str('_'.join(img.split('_')[:-1])) + '_re' + str(img.split('_')[-1])))
    # imgls.append(('book-A.jpg','book-B.jpg'))
    # print(len(imgls))
    #
    #
    # with open(r'D:\programming_project\image_fusion_data\all_results\show10pair.pkl', 'wb') as rfile:
    #     pkl.dump(imgls, rfile)

###20191125：generate raw&blur pairs with gt 0 or 1    ILSVRC2012_val_00004831_1.jpg//2_blur_ILSVRC2012_val_00029095_7.jpg
    import shutil
    imgdir = r'./data/raw/' #clear images
    savedir =  r'./data/raw_blur_pair/'  # root dir of data
    blurimg = r'./data/blur/' #blur images
    rb_ls = []
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    for img in os.listdir(imgdir):
        print(img)
        # b_id = random.randint(1, 7)
        for i in range(1, 8):
            blurname = str(i) + '_blur_' + str(img)
            # print(b_id)
            # print(blurname)
            if blurname in os.listdir(blurimg):
                # print('ok')
                rb_ls.append(tuple((img,blurname)))
    # print(rb_ls)
    rb_ls = random.sample(rb_ls, 20000)
    print(len(rb_ls))
    # print(rb_ls)
    for tu in rb_ls:
        img = tu[0]
        blurname = tu[1]
        shutil.copy(os.path.join(imgdir, img), os.path.join(savedir, img))
        shutil.copy(os.path.join(blurimg, blurname), os.path.join(savedir, blurname))

    with open('./data/train_raw_blur_pair_20k.pkl', 'wb') as trainfile:
            pkl.dump(rb_ls, trainfile)


