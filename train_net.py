# coding: utf-8


import torch
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import numpy as np
import os
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append(".")
from utils.utils import MyDataset, MyTestDataset,MyDataset_Gradient,MyDataset_Raw_Blur
from tensorboardX import SummaryWriter
from datetime import datetime
from net import Generator as Generator
import cv2
import gradient_loss as GL
import pytorch_ssim

#-----------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--train_path', type=str, default='./data/train_raw_blur_pair_20k.pkl', help='train image list')#train_pair_20k
parser.add_argument('--valid_path', type=str, default='./data/val_pair.pkl', help='validation image list')
parser.add_argument('--test_path', type=str, default='./data/lytro', help='test image list')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--train_bs', type=int, default=50, help='train batch size')
parser.add_argument('--valid_bs', type=int, default=50, help='validate batch size')
parser.add_argument('--test_bs', type=int, default=1, help='validate batch size')
parser.add_argument('--lr_init', type=float, default=0.05, help='learning rate for generator')
parser.add_argument('--max_epoch', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--ckpt_step', type=int, default=2, help='cheakpoint step')
parser.add_argument('--warm_up', action='store_true', default=False, help='warm_up_network')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--imageSize', type=int, default=128, help='the low resolution image size')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--load_ckpt', type=str, default=None, help="path to finetune weights (to continue training)")
parser.add_argument('--out', type=str, default='./Result/', help='folder to output logs and models')
parser.add_argument('--outtest', type=str, default='./data/test/', help='output image dir')
parser.add_argument('--outval', type=str, default='./data/val/', help='output image dir')
parser.add_argument('--outtrain', type=str, default='./data/train/', help='output image dir')
parser.add_argument('--thread', type=float, default=0.5, help='thread for mask')
opt = parser.parse_args()

#-----------------------------log-----------------------------------------------------------

# log
result_dir = opt.out
now_time = datetime.now()
time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')

log_dir = os.path.join(result_dir, time_str)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

writer = SummaryWriter(log_dir=log_dir)

##------------------------check if cudnn and cuda are avaliable-----------
cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

torch.cuda.empty_cache()
# device = torch.device("cuda:{0,1}" if opt.cuda else "cpu")
# device = torch.cuda.set_device(id= 0,1)
ngpu = int(opt.nGPU)#torch.cuda.device_count()
# ------------------------------------ step 1/5 ------------------------------------

trainTransform = transforms.Compose([
    transforms.ToTensor()
])

validTransform = transforms.Compose([
    transforms.ToTensor()
])

testTransform = transforms.Compose([
    transforms.ToTensor()
])


#train_data = MyDataset_Raw_Blur(path=opt.train_path, transform=trainTransform)
#valid_data = MyDataset_Gradient(path=opt.valid_path, transform=validTransform)

train_data = MyDataset_Gradient(path=opt.train_path, transform=trainTransform)
valid_data = MyDataset_Gradient(path=opt.valid_path, transform=validTransform)

#test_data = MyTestDataset(path=opt.test_path, transform = testTransform)

train_loader = DataLoader(dataset=train_data, batch_size=opt.train_bs, shuffle=True, drop_last=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=opt.valid_bs, shuffle=False)
#test_loader = DataLoader(dataset=test_data, batch_size=opt.test_bs)

# ------------------------------------ step 2/5 -----------------------------------




n_res_blk = 12
net = Generator(n_res_blk)#.to(device)

print(net)
#-----multi-gpu training---------
# def load_network(network):
#     save_path = os.path.join('./model',name,'net_%s.pth'%opt.which_epoch)
#     state_dict = torch.load(save_path)
#     # create new OrderedDict that does not contain `module.`
#     from collections import OrderedDict
#     new_state_dict = OrderedDict()
#     for k, v in state_dict.items():
#         namekey = k[7:] # remove `module.`
#         new_state_dict[namekey] = v
#     # load params
#     network.load_state_dict(new_state_dict)
#     return network

def _load_ckpt(model, ckpt):
    load_dict = {}
    for name in ckpt:
        print(name.split('module.')[1])
        load_dict[name.split('module.')[1]]= ckpt[name]
    model.load_state_dict(load_dict, strict=False)
    return model

if opt.load_ckpt is not None: #finetune
    print(opt.load_ckpt)
    pretrained_dict = torch.load(opt.load_ckpt)
    net_state_dict = net.state_dict()
    # from collections import OrderedDict
    # net_state_dict = OrderedDict()
    pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
    net_state_dict.update(pretrained_dict_1)
    net.load_state_dict(net_state_dict)
    print('loading pretrained params')
else: #train from scratch
    net.initialize_weights()

if opt.cuda:
    net = torch.nn.DataParallel(net, device_ids=[0, 1])
    net.cuda()
    # net = net.to(device)



print('warm_up:',opt.warm_up)
if opt.warm_up:
    print('warming up network..')
    criterion = nn.L1Loss()
    criterion_bet = nn.L1Loss()
    ssim_loss = pytorch_ssim.SSIM(window_size=11)  # to maximize this loss
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, dampening=0.1)
    for epoch in range(1):
        loss_sigma = loss_gradients_sum = loss_between_pair_sum = loss_pair_lable_sum = loss_ssim_sum = 0.0  # loss

        for i, data in enumerate(train_loader):
            # images and labels
            input1, input2, gt1, gt2, lb1, lb2, gtimg = data
            input1, input2, gt1, gt2, gtimg = Variable(input1).cuda(), Variable(input2).cuda(), Variable(
                gt1).cuda(), Variable(gt2).cuda(), Variable(gtimg).cuda().requires_grad_(False)
            inputs = torch.cat((input1, input2), 1)  # concat input pair image tensor
            labels = torch.cat((gt1, gt2), 1).requires_grad_(False)  # concat ground truth image tensor
            # forward, backward, update weights
            optimizer.zero_grad()
            outputs = net(inputs)

            loss_pair_lable = criterion(outputs, labels)  # loss for each image prediction and gt
            g1 = outputs[:, 0, :, :].unsqueeze(1) * (input1.data)  # predict feature_map1
            g2 = outputs[:, 1, :, :].unsqueeze(1) * (input2.data)  # predict feature_map2
            mergeimg = g1 + g2
            loss_ssim = 1.0 - ssim_loss(mergeimg.cuda(), gtimg.cuda())
            loss_gradients = GL.gradient_loss_merge(mergeimg, gtimg, opt.cuda, device =0)
            # loss of 1-A-B
            sumpreds = outputs[:, 0, :, :] + outputs[:, 1, :, :]
            sumpreds = 1.0 - sumpreds.unsqueeze(1)
            zeroimg = torch.zeros(sumpreds.size())
            loss_between_pair = criterion_bet(sumpreds.cuda(), zeroimg.cuda())

            loss = 0.8 * loss_pair_lable + 0.1 * loss_ssim + 0.1 * loss_gradients
            loss.backward()

            torch.nn.utils.clip_grad_norm(net.parameters(), 0.5)
            optimizer.step()

            loss_sigma += loss.item()
            loss_gradients_sum += loss_gradients.item()
            loss_between_pair_sum += loss_between_pair.item()
            loss_pair_lable_sum += loss_pair_lable.item()
            loss_ssim_sum += loss_ssim.item()
            # print information for each 10 iteration
            if i % 10 == 9:
                loss_avg = loss_sigma / 10
                loss_pair_lable_avg = loss_pair_lable_sum / 10
                loss_between_pair_avg = loss_between_pair_sum / 10
                loss_gradients_avg = loss_gradients_sum / 10
                loss_ssim_avg = loss_ssim_sum / 10
                loss_sigma = loss_pair_lable_sum = loss_between_pair_sum = loss_gradients_sum = loss_ssim_sum = 0.0
                print(
                    "Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] lr:{} Loss: {:.4f} Loss_pair: {:.4f} Loss_bt: {:.4f} Loss_grads: {:.4f} Loss_ssim: {:.4f} ".format(
                        epoch + 1, opt.max_epoch, i + 1, len(train_loader), '0.001', loss_avg,
                        loss_pair_lable_avg,
                        loss_between_pair_avg, loss_gradients_avg, loss_ssim_avg))

        loss_sigma = loss_gradients_sum = loss_between_pair_sum = loss_pair_lable_sum = 0.0

    net_save_path = os.path.join(log_dir, str('warm') +'_net_params.pkl')
    torch.save(net.state_dict(), net_save_path)

    net.load_state_dict(net.state_dict())
    # print(net_save_path)
    print(net)


print('train network begin..')
# ------------------------------------ step 3/5  ------------------------------------

criterion = nn.L1Loss() # loss function
# criterion = nn.BCELoss() # loss function to compare
criterion_bet = nn.MSELoss(size_average=False)
ssim_loss = pytorch_ssim.SSIM(window_size = 11)#to maximize this loss
#optimizer = optim.SGD(net.parameters(), lr=opt.lr_init, momentum=0.9, dampening=0.1)
optimizer = optim.Adam(net.parameters(), lr=opt.lr_init)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
# ------------------------------------ step 4/5  --------------------------------------------------

for epoch in range(opt.max_epoch):

    loss_sigma = loss_gradients_sum = loss_between_pair_sum = loss_pair_lable_sum = loss_ssim_sum = 0.0  # loss
    scheduler.step()
    for i, data in enumerate(train_loader):
        # images and labels
        input1, input2, gt1, gt2, lb1, lb2, gtimg = data
        input1, input2, gt1, gt2, gtimg = Variable(input1).cuda(), Variable(input2).cuda(), Variable(gt1).cuda(), Variable(gt2).cuda(),Variable(gtimg).cuda().requires_grad_(False)
        inputs = torch.cat((input1, input2), 1) #concat input pair image tensor
        labels = torch.cat((gt1, gt2), 1).requires_grad_(False)#concat ground truth image tensor
        # forward, backward, update weights
        optimizer.zero_grad()
        outputs = net(inputs)

        loss_pair_lable = criterion(outputs,labels)# loss for each image prediction and gt
        g1 = outputs[:,0,:,:].unsqueeze(1) * (input1.data )# predict feature_map1
        g2 = outputs[:, 1, :, :].unsqueeze(1) * (input2.data)#predict feature_map2
        mergeimg = g1 + g2
        loss_ssim = 1.0 - ssim_loss(mergeimg.cuda(),gtimg.cuda())
        loss_gradients = GL.gradient_loss_merge(mergeimg,gtimg,opt.cuda,device=0)
        #loss of 1-A-B
        sumpreds = outputs[:,0,:,:] + outputs[:,1,:,:]
        sumpreds = 1.0-sumpreds.unsqueeze(1)
        zeroimg = torch.zeros(sumpreds.size())
        loss_between_pair = criterion_bet(sumpreds.cuda(),zeroimg.cuda())

        loss = 0.8*loss_pair_lable +0.1*loss_ssim +0.1*loss_gradients

        loss.backward()

        torch.nn.utils.clip_grad_norm(net.parameters(), 0.5)
        optimizer.step()

        loss_sigma += loss.item()
        loss_gradients_sum += loss_gradients.item()
        #loss_between_pair_sum += loss_between_pair.item()
        loss_pair_lable_sum += loss_pair_lable.item()
        loss_ssim_sum += loss_ssim.item()
        # print information for each 10 iteration
        if i % 10 == 9:
            loss_avg = loss_sigma / 10
            loss_pair_lable_avg = loss_pair_lable_sum / 10
            loss_between_pair_avg = loss_between_pair_sum / 10
            loss_gradients_avg = loss_gradients_sum / 10
            loss_ssim_avg = loss_ssim_sum / 10
            loss_sigma = loss_pair_lable_sum = loss_between_pair_sum = loss_gradients_sum = loss_ssim_sum = 0.0
            print(
                "Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] lr:{} Loss: {:.4f} Loss_pair: {:.4f} Loss_bt: {:.4f} Loss_grads: {:.4f} Loss_ssim: {:.4f} ".format(
                    epoch + 1, opt.max_epoch, i + 1, len(train_loader),  scheduler.get_lr()[0], loss_avg, loss_pair_lable_avg,
                    loss_between_pair_avg, loss_gradients_avg, loss_ssim_avg))

            # record loss
            writer.add_scalars('Loss_group', {'train_loss': loss_avg}, epoch)
            # record learning rate
            writer.add_scalar('learning rate', scheduler.get_lr()[0], epoch)

        if epoch % 10 == 0:  # save mask for each 10 epoch
            train_mask_result_dir = os.path.join(opt.outtrain, str(time_str), str(epoch))
            if not os.path.exists(train_mask_result_dir):
                os.makedirs(train_mask_result_dir)
            for i in range(0,outputs.size(0)):
                outmask1 = outputs[i,0,:,:].squeeze().unsqueeze(-1).repeat(1,1,3)
                mask_id1 = lb1[i]
                outmask2 = outputs[i, 1, :, :].squeeze().unsqueeze(-1).repeat(1, 1, 3)
                mask_id2 = lb2[i]
                #--------------------------
                output = outputs[i,0,:,:].squeeze().unsqueeze(-1).repeat(1,1,3)
                cv2.imwrite(train_mask_result_dir + '/' + str(mask_id1) + '.jpg', output.cpu().data.numpy() * 255)
                #---------------------------
                cv2.imwrite(train_mask_result_dir +'/' + str(mask_id1) + '.jpg', outmask1.cpu().data.numpy()*255)
                cv2.imwrite(train_mask_result_dir + '/' + str(mask_id2) + '.jpg', outmask2.cpu().data.numpy() * 255)
                cv2.imwrite(train_mask_result_dir + '/' + str('m_'+ mask_id1) + '.jpg', np.clip((outmask1.cpu().data.numpy()*255 + outmask2.cpu().data.numpy() * 255),0,255))

    # for each epoch, record
    for name, layer in net.named_parameters():
        writer.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(), epoch)
        writer.add_histogram(name + '_data', layer.cpu().data.numpy(), epoch)



    # ------------------------------------validation ------------------------------------
    if epoch == 50:

        loss_sigma = loss_gradients_sum =loss_between_pair_sum=loss_pair_lable_sum=loss_ssim_sum=0.0


        net.eval()
        for i, data in enumerate(valid_loader):
            input1, input2, gt1, gt2, lb1, lb2, gtimg = data
            input1, input2, gt1, gt2, gtimg = Variable(input1).cuda(), Variable(input2).cuda(), Variable(
                gt1).cuda(), Variable(gt2).cuda(), Variable(gtimg).cuda().requires_grad_(False)

            inputs = torch.cat((input1, input2), 1)  # concat input pair image tensor

            labels = torch.cat((gt1, gt2), 1).requires_grad_(False) # concat ground truth image tensor
            # print(labels.size())#torch.Size([16, 2, 128, 128]

            # forward, backward, update weights
            optimizer.zero_grad()
            outputs = net(inputs)

            loss_pair_lable = criterion(outputs, labels)  # loss for each image prediction and gt


            g1 = outputs[:, 0, :, :].unsqueeze(1) * (input1.data)  # predict feature_map1
            g2 = outputs[:, 1, :, :].unsqueeze(1) * (input2.data) # predict feature_map2

            mergeimg = g1 + g2

            loss_ssim = 1.0 - ssim_loss(mergeimg.cuda(),gtimg.cuda())
            loss_gradients = GL.gradient_loss_merge(mergeimg, gtimg, opt.cuda, device=0)

            #loss of 1-A-B
            sumpreds = outputs[:, 0, :, :] + outputs[:, 1, :, :]
            sumpreds = 1.0 - sumpreds.unsqueeze(1)
            zeroimg = torch.zeros(sumpreds.size())
            loss_between_pair = criterion_bet(sumpreds.cuda(), zeroimg.cuda())

            #loss = loss_pair_lable +2.0* loss_ssim+0.1* loss_gradients +0.1*loss_between_pair
            loss = 0.8 * loss_pair_lable + 0.1 * loss_ssim + 0.1 * loss_gradients
            loss.backward()
            optimizer.step()

            loss_sigma += loss.item()
            loss_gradients_sum += loss_gradients.item()
            #loss_between_pair_sum += loss_between_pair.item()
            loss_pair_lable_sum += loss_pair_lable.item()
            loss_ssim_sum += loss_ssim.item()

        if epoch % 10 == 0:
            val_mask_result_dir = os.path.join(opt.outval, str(time_str), str(epoch))
            if not os.path.exists(val_mask_result_dir):
                os.makedirs(val_mask_result_dir)
            for i in range(0,outputs.size(0)):
                outmask1 = outputs[i, 0, :, :].squeeze().unsqueeze(-1).repeat(1, 1, 3)
                mask_id1 = lb1[i]
                outmask2 = outputs[i, 1, :, :].squeeze().unsqueeze(-1).repeat(1, 1, 3)
                mask_id2 = lb2[i]
                # --------------------------
                output = outputs[i, 0, :, :].squeeze().unsqueeze(-1).repeat(1, 1, 3)
                cv2.imwrite(train_mask_result_dir + '/' + str(mask_id1) + '.jpg', output.cpu().data.numpy() * 255)
                # ---------------------------

                cv2.imwrite(val_mask_result_dir + '/' + str(mask_id1) + '.jpg', outmask1.cpu().data.numpy() * 255)
                cv2.imwrite(val_mask_result_dir + '/' + str(mask_id2) + '.jpg', outmask2.cpu().data.numpy() * 255)
                cv2.imwrite(val_mask_result_dir + '/' + str('m_' + mask_id1) + '.jpg',
                          np.clip((outmask1.cpu().data.numpy() * 255 + outmask2.cpu().data.numpy() * 255),0,255))


        print('epoch {},{} set valid_loss:{:.2%},Loss_pair: {:.4f} Loss_bt: {:.4f} Loss_grads: {:.4f} Loss_ssim: {:.4f}'.format(epoch, 'Valid', loss_sigma / len(valid_loader), \
                                                                                                              loss_pair_lable_sum/len(valid_loader),\
                                                                                                              loss_between_pair_sum/len(valid_loader), \
                                                                                                              loss_gradients_sum / len(valid_loader),\
                                                                                                              loss_ssim_sum / len(valid_loader)))
        # Loss, accuracy
        writer.add_scalars('Loss_group', {'valid_loss': loss_sigma / len(valid_loader)}, epoch)

        #--------------------------------------save model----------------

    if epoch % opt.ckpt_step == 0:
        net_save_path = os.path.join(log_dir, str(epoch) +'_net_params.pkl')
        torch.save(net.state_dict(), net_save_path)


print('Finished Training')

# ------------------------------------ step5: save model ------------------------------------
net_save_path = os.path.join(log_dir, str(opt.max_epoch) + '_net_params.pkl')
torch.save(net.state_dict(), net_save_path)
