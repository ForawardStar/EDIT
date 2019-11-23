import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
from PIL import Image
import random

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torchvision.models as models

from models import *
from datasets import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=26, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=3000, help='number of epochs of training')
parser.add_argument('--dataset_name', type=str, default="models_full", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=25, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=256, help='size of image height')
parser.add_argument('--img_width', type=int, default=256, help='size of image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=200, help='interval between sampling images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=10, help='interval between saving model checkpoints')
parser.add_argument('--n_residual_blocks', type=int, default=9, help='number of residual blocks in generator')
parser.add_argument('--clip_value', type=int, default=0.02, help='lower and upper clip value for disc, weights')
opt = parser.parse_args()

#vgg16 = models.vgg16(pretrained=True).features
#vgg16 = vgg16.cuda()
# G_AB = TransformNet(32)
# G_BA = TransformNet(32)
# metanet = MetaNet(G_AB.get_param_dict()).cuda()
# D_A = Discriminator()
# D_B = Discriminator()
#
# G_AB = G_AB.cuda()
# G_BA = G_BA.cuda()
# D_A = D_A.cuda()
# D_B = D_B.cuda()

# G_AB.load_state_dict(torch.load('saved_models/%s/G_AB_%d.pth' % (opt.dataset_name, 70)))
# G_BA.load_state_dict(torch.load('saved_models/%s/G_BA_%d.pth' % (opt.dataset_name, 70)))
# vgg16.load_state_dict(torch.load('saved_models/%s/vgg16%d.pth' % (opt.dataset_name,70)))
# metanet.load_state_dict(torch.load('saved_models/%s/metanet%d.pth' % (opt.dataset_name, 70)))
# D_A.load_state_dict(torch.load('saved_models/%s/D_A_%d.pth' % (opt.dataset_name, 70)))
# D_B.load_state_dict(torch.load('saved_models/%s/D_B_%d.pth' % (opt.dataset_name, 70)))

domains_num = 4
G = torch.load('saved_models/%s/G_AB_%d.pth' % (opt.dataset_name, opt.epoch))
#G_BA = torch.load('saved_models/%s/entire_G_BA_%d.pth' % (opt.dataset_name, opt.epoch))
vgg16 = torch.load('saved_models/%s/vgg16%d.pth' % (opt.dataset_name,opt.epoch))
metanet = torch.load('saved_models/%s/metanet%d.pth' % (opt.dataset_name, opt.epoch))
#D_A = torch.load('saved_models/%s/entire_D_A_%d.pth' % (opt.dataset_name, 71))
#D_B = torch.load('saved_models/%s/entire_D_B_%d.pth' % (opt.dataset_name, 71))

G.cuda()
vgg16.cuda()
metanet.cuda()

G.eval()
vgg16.eval()
metanet.eval()
# G_AB.eval()
# G_BA.eval()
#vgg16.eval()
# metanet.eval()
# D_A.eval()
# D_B.eval()
Tensor = torch.cuda.FloatTensor

transforms_ = [ transforms.Resize(int(opt.img_height), Image.BICUBIC),
                transforms.RandomCrop((opt.img_height, opt.img_width)),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
transforms_ = transforms.Compose(transforms_)
one_hot_label = torch.tensor(np.zeros((domains_num + 1, 256, 256))).unsqueeze(0).type(Tensor)
one_hot_label[0][2] = 1
one_hot_label[0][domains_num] = 1 


# val_dataloader = DataLoader(ImageDataset("test_images", transforms_=transforms_, unaligned=True, mode='test'),
#                         batch_size=1, shuffle=True, num_workers=1)


root_path = "/home/vector/fuyuanbin/monet2photo/testB/"
#ref_path = "../test_images/reference/"
ref_path = "reference/"
files = os.listdir(root_path)
ref_files = os.listdir(ref_path)
count = 0 
for file in files:
    file_path = root_path + file
    #file_path = "cmp_b0017_process.png"
    #ref_file = "ref_1.png"
    
    for ref_file in ref_files: 
    #ref_file = ref_files[random.randint(0,len(ref_files) -

        ref_file_path = ref_path + ref_file 
        ref_count = ref_file.split(".")[0] 
        img = transforms_(Image.open(file_path).convert("RGB")).unsqueeze(0).type(Tensor)
        ref = transforms_(Image.open(ref_file_path).convert("RGB")).unsqueeze(0).type(Tensor)
        img.cuda()
        ref.cuda()

        ref_features = vgg16(ref)
        val_weight = metanet(mean_std(ref_features), one_hot_label)
        G.set_weights(val_weight, 0)

        val_fake = G(img)
    #save_image(img, 'test_images/test_result/input_%d.png' % (count), nrow=5,
    #           normalize=True)
        save_image(img, 'test_input/input_%d.png' % (count), nrow=5,
               normalize=True)
        #save_image(ref, 'test_result/ref_%d.png' % (ref_count), nrow=5,


        #       normalize=True)
        save_image(val_fake, 'test_result_full/result_%d_%s.png' % (count, ref_count), nrow=5,
               normalize=True)

        #ref_count = ref_count + 1
    #ref_file = "684_AB.jpg"
    #ref_file_path = ref_path + ref_file
    
    #ref = transforms_(Image.open(ref_file_path).convert("RGB")).unsqueeze(0).type(Tensor)
    #img.cuda()
    #ref.cuda()

    #ref_features = vgg16(ref)
    #val_weight = metanet(mean_std(ref_features))
    #G.set_weights(val_weight, 0)

    #val_fake = G(img, one_hot_label)
    
    #save_image(val_fake, 'test_images/test_result/result_%d_2.png' % (count), nrow=5,
    #           normalize=True)
    count = count + 1
    print("count: ",count)




def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    val_real_A = Variable(imgs['A'].type(Tensor))
    val_real_B = Variable(imgs['B'].type(Tensor))
    label_AB = Variable(imgs['label_AB'].type(Tensor))
    label_BA = Variable(imgs['label_BA'].type(Tensor))
    r1 = Variable(imgs['reference1'].type(Tensor))
    r2 = Variable(imgs['reference2'].type(Tensor))
    label = imgs['label_name']

    val_real_A_features = vgg16(r1)
    val_real_B_features = vgg16(r2)
    
    val_weights_AB = metanet(mean_std(val_real_B_features))
    val_weights_BA = metanet(mean_std(val_real_A_features))

    G.set_weights(val_weights_AB, 0)
    #G_BA.set_weights(val_weights_BA, 0)
    val_fake_B = G(val_real_A, label_AB)
    #val_reconv_A = G_BA(val_fake_B)
    G.set_weights(val_weights_BA, 0)
    val_fake_A = G(val_real_B, label_BA)
    
    for key in val_weights_AB.keys():
        val_weights_BA[key] = (val_weights_AB[key] + val_weights_BA[key]) / 2

    G.set_weights(val_weights_BA, 0)
    #G_BA.set_weights(val_weights_BA, 0)

    val_fake_B_add = G(val_real_A, label_AB)
    #val_reconv_A = G_BA(val_fake_B)
    #G.set_weights(val_weights_BA, 0)
    val_fake_A_add = G(val_real_B, label_BA)
    # img_sample = torch.cat((val_real_A.data, val_fake_B.data,
    #                         val_real_B.data, val_fake_A.data), 0)
    # save_image(img_sample, 'images/%s/%s.png' % (opt.dataset_name, batches_done), nrow=5, normalize=True)
    # img_sample1 = torch.cat((val_real_A.data,
    #                         val_real_B.data,val_fake_B.data), 0)
    img_sample2 = torch.cat((val_real_A.data,
                             val_real_B.data, val_fake_B.data), 0)

    #save_image(img_sample1, 'images/%s/test_result/%s_mobnet2photo.png' % (opt.dataset_name, batches_done), nrow=5, normalize=True)
    #save_image(img_sample2, 'images/%s/test_result/%s_photo2monet.png' % (opt.dataset_name, batches_done), nrow=5, normalize=True)
    # save_image(val_fake_B, 'images/%s/test_result/%s_photo2monet.png' % (opt.dataset_name, batches_done), nrow=5,
    #            normalize=True)
    save_image(r1, 'images/test_result/%s_reference1_%d.png' % (batches_done, label), nrow=5,
               normalize=True)
    save_image(r2, 'images/test_result/%s_reference2_%d.png' % (batches_done, label), nrow=5,
                           normalize=True)
    save_image(val_real_B, 'images/test_result/%s_photo_%d.png' % (batches_done, label), nrow=5,
               normalize=True)
    save_image(val_real_A, 'images/test_result/%s_painting_%d.png' % (batches_done, label), nrow=5,
               normalize=True)
    save_image(val_fake_B, 'images/test_result/%s_monet2photo_%s.png' % (batches_done, label), nrow=5,
               normalize=True)
    save_image(val_fake_A, 'images/test_result/%s_photo2monet_%s.png' % (batches_done, label), nrow=5,
               normalize=True)
    save_image(val_fake_B_add, 'images/test_result/%s_monet2photoadd_%s.png' % (batches_done, label), nrow=5,
                           normalize=True)
    save_image(val_fake_A_add, 'images/test_result/%s_photo2monetadd_%s.png' % (batches_done, label), nrow=5,
                           normalize=True)

# #length = 2*len(val_dataloader)
# length = 12
# for i in range(length):
#     sample_images(i)
