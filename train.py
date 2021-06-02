#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 16:55:21 2021

@author: maggie
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision

import os
import time 
import argparse

from dataset import InpaintDataset
from model import Generator, Discriminator
from model_stage2 import UAE_noFC_AfterNoise
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--imgpath", default = ".")
parser.add_argument("--txtpath", default = ".")
parser.add_argument("--year", default = "")
parser.add_argument("--model", default = 'ResNet')
parser.add_argument("--w", default = 224)
parser.add_argument("--h", default = 224)
parser.add_argument("--batch_size", default = 64)
parser.add_argument("--checkpoint_path", default = "checkpoints")
parser.add_argument("--save_dir",default = "output")
parser.add_argument("--lr", default= 0.0002, help='initial learning rate for adam')
parser.add_argument("--beta1", default= 0.5, help='momentum term of adam')
parser.add_argument("--beta2", default= 0.999)
parser.add_argument("--lambda_bce", default= 100)
parser.add_argument("--epochs", default= 100)
parser.add_argument("--display_count", type=int, default = 10, help='steps to show')
parser.add_argument("--save_count", type=int, default = 10, help='epoch to save')
opt = parser.parse_args()

if not os.path.exists(os.path.dirname(os.path.join('.', opt.checkpoint_path))):
        os.makedirs(os.path.dirname(os.path.join('.', opt.checkpoint_path)))
if not os.path.exists(os.path.dirname(os.path.join('.', opt.save_dir))):
        os.makedirs(os.path.dirname(os.path.join('.', opt.save_dir))) 
        
        
train_dataset = InpaintDataset(opt, mode = 'train')
train_dataloader = DataLoader(train_dataset, batch_size = opt.batch_size, shuffle = True, pin_memory = True, num_workers = 8)
print('Size of the training dataset: %d, dataloader: %d' % (len(train_dataset), len(train_dataloader)))
val_dataset = InpaintDataset(opt, mode = 'val')
val_dataloader = DataLoader(val_dataset, batch_size = opt.batch_size, shuffle = True, pin_memory = True, num_workers = 8)
print('Size of the validation dataset: %d, dataloader: %d' % (len(val_dataset), len(val_dataloader)))

generator = Generator()
#generator = UAE_noFC_AfterNoise(ch_in=3, repeat_num = None, hidden_num=64)
discriminator = Discriminator()

generator.cuda()


discriminator.cuda()

Loss_G = []
Loss_D = []
val_Loss_G = []
val_Loss_D = []

lossBCE = nn.BCELoss()
lossMSE = nn.MSELoss()
lossL1 = nn.L1Loss()


optimizer_G = torch.optim.Adam(generator.parameters(), lr = opt.lr, betas = (opt.beta1, opt.beta2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = opt.lr, betas = (opt.beta1, opt.beta2))


for epoch in range(opt.epochs):
    
    generator.train()
    discriminator.train()
    for step, batch in enumerate(train_dataloader):
        start_time = time.time()
        
        img = batch["image"].cuda()
        N, C, H, W = img.shape
        #print(N)
        mask = batch['crop_mask'].cuda()
        point_r = batch['point_r']
        point_c = batch['point_c']
        #print(point_r, point_c)
        #print(img.shape)
        x = img*mask
        y = img*(1-mask)
        
        #train generator
        x = x.float()
        output = generator(x)
        #print('output: ', output.shape)
        #GAN loss
        x = torch.cat([img, output.detach()], dim = 0)
        d_z = discriminator(x)
        #print(d_z)
        d_z = torch.clamp(d_z, 0.0, 1.0)
        #print(d_z)
        d_z_pos = d_z[:N]
        d_z_neg = d_z[N:]
        #print(d_z_pos)
        #print(d_z_neg)
        
        L1 = lossL1(output, img)
        bce = lossBCE(d_z_neg, torch.ones((N, 1)).cuda())
        loss_G = 50*L1 + bce
        
        
        optimizer_G.zero_grad()
        loss_G.backward(retain_graph = True)
        optimizer_G.step()
        Loss_G.append(loss_G)
        
        
        loss_real = lossBCE(d_z_pos, torch.ones((N,1)).cuda())
        loss_fake = lossBCE(d_z_neg, torch.zeros((N, 1)).cuda())
        loss_D = (loss_real+loss_fake)/2
        
        
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()
        Loss_D.append(loss_D)
        
    
    
    generator.eval()
    discriminator.eval()
    for step, batch in enumerate(val_dataloader):
        img = batch["image"].cuda()
        N, C, H, W = img.shape
        mask = batch['crop_mask'].cuda()
        point_r = batch['point_r']
        point_c = batch['point_c']
        #print(point_r, point_c)
        #print(img.shape)
        x = img*mask
        y = img*(1-mask)
        
        #train generator
        x = x.float()
        output = generator(x)
        d_z = discriminator(torch.cat([img, output.detach()], dim = 0))
        #print(d_z)
        d_z = torch.clamp(d_z, 0.0, 1.0)
        #print(d_z)
        d_z_pos = d_z[:N] #d_z[0]
        d_z_neg = d_z[N:]#d_z[1]
        
        L1 = lossL1(output, img)
        bce = lossBCE(d_z_neg, torch.ones((N,1)).cuda())
        val_loss_G = 50*L1 + bce
        
        loss_real = lossBCE(d_z_pos, torch.ones((N,1)).cuda())
        loss_fake = lossBCE(d_z_neg, torch.zeros((N,1)).cuda())
        val_loss_D = (loss_real+loss_fake)/2
        
        val_Loss_G.append(val_loss_G)
        val_Loss_D.append(val_loss_D)
        
    if (epoch+1) % opt.display_count == 0:
        t = time.time() - start_time
        print('step: %6d, time: %.3f, loss: %4f' % (step+1, t, loss_G.item()), flush = True)
        utils.imgsave(output, os.path.join(opt.save_dir, '{}_{}.jpg'.format(epoch+1, step+1)))
        with open("loss_trainG.txt", "a") as f:
            for i in Loss_G:
                f.write("{}\n".format(i))
            Loss_G = []
        with open("loss_trainD.txt", "a") as f:
            for i in Loss_D:
                f.write("{}\n".format(i))
            Loss_D = []
        with open("loss_valG.txt", "a") as f:
            for i in val_Loss_G:
                f.write("{}\n".format(i))
            val_Loss_G = []
        with open("loss_valD.txt", "a") as f:
            for i in val_Loss_D:
                f.write("{}\n".format(i))
            val_Loss_D = []
    if (epoch+1) % opt.save_count == 0:
        utils.save_checkpoint(generator, os.path.join(opt.checkpoint_path, 'g_%04d.pth' % (epoch+1)))
        utils.save_checkpoint(discriminator, os.path.join(opt.checkpoint_path, 'd_%04d.pth' % (epoch+1)))   














