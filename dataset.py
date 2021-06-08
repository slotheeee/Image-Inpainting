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
        self.transforms = transforms.Compose([
                            transforms.Resize((self.h, self.w)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = (0.485,0.456,0.406),std = (0.229,0.224,0.225))
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
        #print(name)
        img = Image.open(os.path.join(self.imgpath, self.mode+self.year, name))
        img = img.convert('RGB')
        #imgnp = np.asarray(img)
        #imgnp = self.transforms(imgnp)#np.resize(imgnp,(self.h, self.w, 3))
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
            "image_name": name,
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
    parser.add_argument("--year", default = "")
    parser.add_argument("--w", default = 224)
    parser.add_argument("--h", default = 224)
    parser.add_argument("--batch_size", default = 1)
    opt = parser.parse_args()
    
    dataset = InpaintDataset(opt, mode = 'train')
    dataloader = DataLoader(dataset, batch_size = opt.batch_size, shuffle = False, pin_memory = True)#, num_workers = 8)

    print('Size of the dataset: %d, dataloader: %d' % (len(dataset), len(dataloader)))
    item = dataset.__getitem__(0)
    
        
    
    data_iter = iter(dataloader)
    print("1")
    batch = data_iter.next()
    print("2")
    
    img = batch['image'].cuda()
    mask = batch['crop_mask'].cuda()
    sr = batch['point_r']
    sc = batch['point_c']
    crop_size = batch['crop_size']
    print(crop_size)
    print(img.shape)
    x = img*mask
    x2 = x[:,:, sr:sr+crop_size-1, sc:sc+crop_size-1]
    print(img[0][1][50][40])
    y = img*(1-mask)
    
    add = x+y
    
    
    utils.imgsave(mask, './mask.png')
    utils.imgsave(x2, './x.png')
    utils.imgsave(y, './y.png')
    utils.imgsave(add, './xy.png')
    
    
    
    counter = 0
    for step, batch in enumerate(dataloader):
        counter+=opt.batch_size
        img = batch['image'].cuda()
        #print(img.shape)
        print('counter: ',counter)





