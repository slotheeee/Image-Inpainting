# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from PIL import Image
from PIL import ImageDraw

import os
import numpy as np
import random
import utils

class InpaintDataset(Dataset):
    def __init__(self, opt, mode):
        super(InpaintDataset, self).__init__()
        self.opt = opt
        self.mode = mode
        self.imgpath = opt.imgpath
        self.txtpath = opt.txtpath
        self.year = opt.year
        self.w = opt.w
        self.h = opt.h
        self.transforms = transforms.Compose([\
                            transforms.Resize((self.h, self.w)),\
                            transforms.ToTensor(),\
                            #ransforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                            ])
        names = []
        with open(os.path.join(self.txtpath, self.mode+self.year+".txt"), "r") as f:
            for line in f.readlines():
                name = line.strip("\n")
                names.append(name)
        self.names = names
        
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, index):
        name = self.names[index]
        img = Image.open(os.path.join(self.imgpath, self.mode+self.year, name))
        imgnp = np.asarray(img)
        imgnp = np.resize(imgnp,(self.h, self.w, 3))
        img = self.transforms(img)
        cropout_size = random.randint(20, int(min(self.w, self.h)/2))
        #start point of cropout part(r, c)
        sr = random.randint(0, self.h-cropout_size-1)
        sc = random.randint(0, self.w-cropout_size-1)
        
        crop_mask = np.zeros((1, self.h, self.w))
        #print("crop_mask",crop_mask.shape)
        crop_mask[:,sr:sr+cropout_size-1, sc:sc+cropout_size-1] = 1
        
        crop_mask = torch.from_numpy(crop_mask)
        
        
        result = {
            "image": img,
            "crop_mask": crop_mask,
            "point_r": sr, 
            "point_c":sc, 
            "crop_size": cropout_size
            }
        return result
    
if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgpath", default = ".")
    parser.add_argument("--txtpath", default = ".")
    parser.add_argument("--year", default = "2017")
    parser.add_argument("--w", default = 224)
    parser.add_argument("--h", default = 224)
    parser.add_argument("--batch_size", default = 4)
    opt = parser.parse_args()
    
    dataset = InpaintDataset(opt, mode = 'val')
    dataloader = DataLoader(dataset, batch_size = opt.batch_size, shuffle = False, pin_memory = True)
    print('Size of the dataset: %d, dataloader: %d' % (len(dataset), len(dataloader)))
    item = dataset.__getitem__(0)
    
        
    
    data_iter = iter(dataloader)
    print("1")
    batch = data_iter.next()
    print("2")
    
    img = batch['image'].cuda()
    mask = batch['crop_mask'].cuda()
    point_r = batch['point_r']
    point_c = batch['point_c']
    print(point_r, point_c)
    print(img.shape)
    x = img*mask
    print(img[0][1][50][40])
    y = img*(1-mask)
    
    add = x+y
    
    
    utils.imgsave(mask, './mask.png')
    utils.imgsave(x, './x.png')
    utils.imgsave(y, './y.png')
    utils.imgsave(add, './xy.png')






