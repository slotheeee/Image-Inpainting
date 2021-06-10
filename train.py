#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 23:22:55 2021

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
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--imgpath", default = ".")
parser.add_argument("--txtpath", default = ".")
parser.add_argument("--year", default = "")
parser.add_argument("--model", default = 'ResNet')
parser.add_argument("--w", type = int, default = 224)
parser.add_argument("--h", type = int, default = 224)
parser.add_argument("--batch_size", type = int, default = 1)
parser.add_argument("--checkpoint_path", default = "checkpoints")
parser.add_argument("--checkpoint_loadpath", default = "checkpoints_best")
parser.add_argument("--checkpoint_num", type = int, default = 500)
parser.add_argument("--save_dir",default = "output")
parser.add_argument("--lr", default= 0.0002, help='initial learning rate for adam')
parser.add_argument("--beta1", default= 0.5, help='momentum term of adam')
parser.add_argument("--beta2", default= 0.999)
parser.add_argument("--epochs", type = int, default= 10000)
parser.add_argument("--display_count", type=int, default = 10, help='steps to show')
parser.add_argument("--save_count", type=int, default = 500, help='epoch to save')
opt = parser.parse_args()

if not os.path.isdir(os.path.join('.', opt.checkpoint_path)):
    os.makedirs(os.path.join('.', opt.checkpoint_path))
    print('checkpoint folder create')
if not os.path.isdir(os.path.join('.', opt.save_dir)):
    os.makedirs(os.path.join('.', opt.save_dir))
    print('output folder create')

        
train_dataset = InpaintDataset(opt, mode = 'train')
train_dataloader = DataLoader(train_dataset, batch_size = opt.batch_size, shuffle = True, pin_memory = True)#, num_workers = 8)
print('Size of the training dataset: %d, dataloader: %d' % (len(train_dataset), len(train_dataloader)))

generator = Generator()#(ch_in=3, repeat_num = None, hidden_num=64)
discriminator = Discriminator()

utils.load_checkpoint(generator, os.path.join(opt.checkpoint_path, 'g_%04d.pth' % (opt.checkpoint_num)))
utils.load_checkpoint(generator, os.path.join(opt.checkpoint_path, 'g_%04d.pth' % (opt.checkpoint_num)))


generator.cuda()
discriminator.cuda()

Loss_G = []
Loss_D = []

lossBCE = nn.BCELoss()
lossMSE = nn.MSELoss()
lossL1 = nn.L1Loss()


optimizer_G = torch.optim.Adam(generator.parameters(), lr = opt.lr, betas = (opt.beta1, opt.beta2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = opt.lr, betas = (opt.beta1, opt.beta2))
generator.train()
discriminator.train()

for epoch in range(opt.epochs):
    
    #generator.train()
    #discriminator.train()
    for step, batch in enumerate(train_dataloader):
        start_time = time.time()
        
        img = batch["image"].cuda()
        N, C, H, W = img.shape
        #print(N)
        mask = batch['crop_mask'].cuda()
        sr = batch['point_r']
        sc = batch['point_c']
        crop_size = batch['crop_size']
        #print(point_r, point_c)
        #print(img.shape)
        x = img*mask
        y = img*(1-mask)
        #print(x.shape)
        #train generator
        x = x.float()
        y = y.float()
        r = generator(y)
        #print(r.shape)
        #print('output: ', output.shape)
        #GAN loss
        output = r+y
        tup = torch.cat([img, output.detach()], dim = 0)
        d_z = discriminator(tup)
        #print(d_z)
        d_z = torch.clamp(d_z, 0.0, 1.0)
        #print(d_z)
        d_z_pos = d_z[:N]
        d_z_neg = d_z[N:]
        #print(d_z_pos)
        #print(d_z_neg)
        #r = Variable(r)
        a = r[:,:, sr:sr+crop_size, sc:sc+crop_size]
        #print('x: ', x.shape)
        #x = Variable(x)
        #print('var: ', x.shape)
        x = x[:,:, sr:sr+crop_size, sc:sc+crop_size]
        #print(a.shape, x.shape)
        L1 = lossL1(output, img)+lossL1(a, x)*50
        #print('L1: ', L1)
        bce = lossBCE(d_z_neg, torch.ones((N, 1)).cuda())
        #print('bce: ', bce)
        loss_G = 2000*L1 + bce
        
        
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
        if (step+1) % 10 == 0:
            t = time.time() - start_time
            print('Train: step: %6d, time: %.3f, loss: %4f' % (step+1, t, loss_G.item()), flush = True)
            
            with open("loss_trainG.txt", "a") as f:
                for i in Loss_G:
                    f.write("{}\n".format(i))
                Loss_G = []
            with open("loss_trainD.txt", "a") as f:
                for i in Loss_D:
                    f.write("{}\n".format(i))
                Loss_D = []
        if (step+1) % opt.display_count == 0:
            utils.imgsave(output, os.path.join(opt.save_dir, '{}_{}.jpg'.format((epoch+1)%3, step+1)))
            utils.imgsave(a, os.path.join(opt.save_dir, '{}_{}_a.jpg'.format((epoch+1)%3, step+1)))
            utils.imgsave(x, os.path.join(opt.save_dir, '{}_{}_x.jpg'.format((epoch+1)%3, step+1)))
        
    
    if (epoch+1) % opt.save_count == 0:
        utils.save_checkpoint(generator, os.path.join(opt.checkpoint_path, 'g_%04d.pth' % (epoch+1)))
        utils.save_checkpoint(discriminator, os.path.join(opt.checkpoint_path, 'd_%04d.pth' % (epoch+1)))   














