import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time

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
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs of training')
parser.add_argument('--dataset_name', type=str, default="monet2photo", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=25, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=256, help='size of image height')
parser.add_argument('--img_width', type=int, default=256, help='size of image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=1000, help='interval between sampling images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=2, help='interval between saving model checkpoints')
parser.add_argument('--n_residual_blocks', type=int, default=9, help='number of residual blocks in generator')
parser.add_argument('--clip_value', type=int, default=0.02, help='lower and upper clip value for disc, weights')
opt = parser.parse_args()
print(opt)

# Create sample and checkpoint directories
os.makedirs('images/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('saved_models/%s' % opt.dataset_name, exist_ok=True)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_style = torch.nn.MSELoss()

cuda = True if torch.cuda.is_available() else False

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2**4, opt.img_width // 2**4)

# Initialize generator and discriminator
# G_AB = GeneratorResNet(res_blocks=opt.n_residual_blocks)
# G_BA = GeneratorResNet(res_blocks=opt.n_residual_blocks)

vgg16 = VGG(models.vgg16(pretrained=True).features[:26])
vgg16 = vgg16.cuda()
G = TransformNet(32)
#G_BA = TransformNet(32)
metanet = MetaNet(G.get_param_dict()).cuda()
D = Discriminator()
#D_B = Discriminator()


if cuda:
    G = G.cuda()
    #G_BA = G_BA.cuda()
    D = D.cuda()
    #D_B = D_B.cuda()

    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_style.cuda()
 
if opt.epoch != 0:
    # Load pretrained models
    # G_AB.load_state_dict(torch.load('saved_models/%s/G_AB_%d.pth' % (opt.dataset_name, opt.epoch)))
    # G_BA.load_state_dict(torch.load('saved_models/%s/G_BA_%d.pth' % (opt.dataset_name, opt.epoch)))
    metanet.load_state_dict(torch.load('saved_models/%s/metanet%d.pth' % (opt.dataset_name, opt.epoch-1)))
    D.load_state_dict(torch.load('saved_models/%s/D_A_%d.pth' % (opt.dataset_name, opt.epoch-1)))
    #D_B.load_state_dict(torch.load('saved_models/%s/D_B_%d.pth' % (opt.dataset_name, opt.epoch-1)))
else:
    # Initialize weights
    metanet.apply(weights_init_normal)
    G.apply(weights_init_normal)
    #G_BA.apply(weights_init_normal)
    D.apply(weights_init_normal)
    #D_B.apply(weights_init_normal)

# Loss weights
lambda_cyc = 10
lambda_sty = 0.1

# Optimizers
trainable_params = {}
trainable_param_shapes = {}
#for model in [vgg16, G, metanet]:
for model in [G, metanet]:
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params[name] = param
            trainable_param_shapes[name] = param.shape

optimizer_G = torch.optim.Adam(trainable_params.values(),
                                lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(D.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
#optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
#lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Image transformations
# transforms_ = [ transforms.Resize(int(opt.img_height*1.12), Image.BICUBIC),
#                 transforms.RandomCrop((opt.img_height, opt.img_width)),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
transforms_ = [ transforms.Resize(int(opt.img_height), Image.BICUBIC),
                transforms.RandomCrop((opt.img_height, opt.img_width)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
# Training data loader
dataloader = DataLoader(ImageDataset("/home/vector/fuyuanbin", transforms_=transforms_, unaligned=True),
                        batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
# Test data loader
val_dataloader = DataLoader(ImageDataset("/home/vector/fuyuanbin", transforms_=transforms_, unaligned=True, mode='test'),
                        batch_size=1, shuffle=True, num_workers=1)

def ColorToByte(tensor, threshold):
    # R = tensor[0][1]
    # G = tensor[0][1]
    # B = tensor[0][2]
    #
    #tensor[0][0] = 0.299 * R + 0.587 * G + 0.114 * B
    tensor[0][1] = tensor[0][0]
    tensor[0][2] = tensor[0][0]

    return torch.where(tensor>threshold,torch.ones(tensor.shape).cuda(),torch.zeros(tensor.shape).cuda())

def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    val_real_A = Variable(imgs['A'].type(Tensor))
    val_real_B = Variable(imgs['B'].type(Tensor))
    val_label_AB = Variable(imgs['label_AB'].type(Tensor))
    val_label_BA = Variable(imgs['label_BA'].type(Tensor))
    label_name = imgs['label_name']

    val_real_A_features = vgg16(val_real_A)
    val_real_B_features = vgg16(val_real_B)

    val_weights_AB = metanet(mean_std(val_real_B_features), val_label_AB)
    val_weights_BA = metanet(mean_std(val_real_A_features), val_label_BA)

    G.set_weights(val_weights_BA,0)
    val_fake_A = G(val_real_B)
    G.set_weights(val_weights_AB,0)
    val_fake_B = G(val_real_A)
    img_sample = torch.cat((val_real_A.data, val_fake_B.data,
                            val_real_B.data, val_fake_A.data), 0)
    save_image(img_sample, 'images/%s/%s_%s.png' % (opt.dataset_name, batches_done, label_name), nrow=5, normalize=True)

# ----------
#  Training
# ----------

prev_time = time.time()
count = 0
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):


        # Set model input

        real_A = Variable(batch['A'].type(Tensor))
        real_B = Variable(batch['B'].type(Tensor))
        label_AB = Variable(batch['label_AB'].type(Tensor))
        label_BA = Variable(batch['label_BA'].type(Tensor))
        label = batch['label_name']

        real_A_features = vgg16(real_A)
        real_B_features = vgg16(real_B)

        realA_mean_std = mean_std(real_A_features)
        realB_mean_std = mean_std(real_B_features)

        weights_AB = metanet(realB_mean_std,label_AB)
        weights_BA = metanet(realA_mean_std,label_BA)

        #G_BA.set_weights(weights_BA, 0)

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        G.set_weights(weights_AB, 0)
        fake_B = G(real_A)
        G.set_weights(weights_BA, 0)
        fake_A = G(real_B)
        recov_A = G(fake_B)
        G.set_weights(weights_AB, 0)
        recov_B = G(fake_A)


        # Identity loss
        # loss_id_A = criterion_identity(G_BA(real_A), real_A)
        # loss_id_B = criterion_identity(G_AB(real_B), real_B)
        #
        # loss_identity = (loss_id_A + loss_id_B) / 2

        # GAN loss
        #fake_B = G_AB(real_A,label)
        loss_GAN_AB = criterion_GAN(D(fake_B,label_AB), valid)
        #fake_A = G_BA(real_B,label)
        loss_GAN_BA = criterion_GAN(D(fake_A,label_BA), valid)

        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        # Cycle loss
        #recov_A = G_BA(fake_B,label)
        loss_cycle_A = criterion_cycle(recov_A, real_A)
        #recov_B = G_AB(fake_A,label)
        loss_cycle_B = criterion_cycle(recov_B, real_B)

        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        fake_A_features = vgg16(fake_A)
        fake_B_features = vgg16(fake_B)

        fakeA_mean_std = mean_std(fake_A_features)
        fakeB_mean_std = mean_std(fake_B_features)

        #loss_style_AB = torch.mean(torch.abs(1-fake_B))
        loss_style_BA = criterion_style(fakeA_mean_std, realA_mean_std)
        if label == 3:
            loss_style_AB = criterion_style(fakeB_mean_std, realB_mean_std)
        else:
            loss_style_AB = 0
        
        loss_style = loss_style_AB + loss_style_BA

        # Total loss
        # loss_G =    loss_GAN + \
        #             lambda_cyc * loss_cycle + \
        #             lambda_id * loss_identity
        loss_G =    loss_GAN + \
                    lambda_cyc * loss_cycle + lambda_sty * loss_style

        loss_G.backward()
        optimizer_G.step()

        #if i % opt.sample_interval == 0:
            # save_image(real_B, 'images/%s/realedge%s_%s.png' % (opt.dataset_name, epoch,i),
            #            nrow=5,normalize=True)
            # save_image(fake_A, 'images/%s/fakeimage%s_%s.png' % (opt.dataset_name, epoch, i),
            #            nrow=5, normalize=True)
        #    save_image(real_A, 'images/%s/realimage%s_%s.png' % (opt.dataset_name, epoch, i),
        #           nrow=5, normalize=True)
        #    save_image(ColorToByte(fake_B, 0.5), 'images/%s/fakeedge%s_%s.png' % (opt.dataset_name, epoch, i),
        #           nrow=5, normalize=True)
        #    save_image(fake_B, 'images/%s/fakeedge%s_%s_withoutbyte.png' % (opt.dataset_name, epoch, i),
        #           nrow=5, normalize=True)
        #    print("colortobyte", ColorToByte(fake_B, 0.5))

        # -----------------------
        #  Train Discriminator A
        # -----------------------

        optimizer_D.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D(real_A,label_BA), valid)
        # Fake loss (on batch of previously generated samples)
        fake_A_, label_ = fake_A_buffer.push_and_pop(fake_A, label_BA)

        loss_fake = criterion_GAN(D(fake_A_.detach(), label_), fake)
        # Total loss
        loss_D_A = (loss_real + loss_fake) / 2

        #loss_D_A.backward()
        #optimizer_D_A.step()

        #for P_A in D_A.parameters():
        #    P_A.data.clamp_(-opt.clip_value,opt.clip_value)

        # -----------------------
        #  Train Discriminator B
        # -----------------------

        #optimizer_D_B.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D(real_B,label_AB), valid)
        # Fake loss (on batch of previously generated samples)
        fake_B_,label_ = fake_B_buffer.push_and_pop(fake_B, label_AB)
        loss_fake = criterion_GAN(D(fake_B_.detach(), label_), fake)
        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2


        #if count >= 3:
        #    count = 0

        #    loss_D_A.backward()
        #    optimizer_D_A.step()

        #    for P_A in D_A.parameters():
        #        P_A.data.clamp_(-opt.clip_value,opt.clip_value)


        #    loss_D_B.backward()
        #    optimizer_D_B.step()

        #    for P_B in D_B.parameters():
        #         P_B.data.clamp_(-opt.clip_value,opt.clip_value)
        #else:
        #    count = count + 1

        loss_D = (loss_D_A + loss_D_B) / 2
        if count >= 2: 
             loss_D.backward() 
             optimizer_D.step()
             count=0  
        else:

             count = count + 1  



        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, style: %f] ETA: %s" %
                                                        (epoch, opt.n_epochs,
                                                        i, len(dataloader),
                                                        loss_D.item(), loss_G.item(),
                                                        loss_GAN.item(), loss_cycle.item(),loss_style.item(),
                                                         time_left))

        #If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

        del valid , fake ,real_A, real_B, real_A_features , real_B_features ,  weights_AB , weights_BA

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D.step()
    #lr_scheduler_D_B.step()

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(G, 'saved_models/%s/G_AB_%d.pth' % (opt.dataset_name, epoch))
        #torch.save(G_BA.state_dict(), 'saved_models/%s/G_BA_%d.pth' % (opt.dataset_name, epoch))
        torch.save(vgg16, 'saved_models/%s/vgg16%d.pth' % (opt.dataset_name, epoch))
        torch.save(metanet, 'saved_models/%s/metanet%d.pth' % (opt.dataset_name, epoch))
        torch.save(D, 'saved_models/%s/D_%d.pth' % (opt.dataset_name, epoch))
        #torch.save(D_B, 'saved_models/%s/D_B_%d.pth' % (opt.dataset_name, epoch))
