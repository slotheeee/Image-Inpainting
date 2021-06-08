#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 29 13:35:32 2021

@author: maggie
"""

import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

def save_checkpoint(model, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
        
    torch.save(model.cpu().state_dict(), save_path)
    model.cuda()
    
def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return
    model.load_state_dict(torch.load(checkpoint_path))
    print("success in loading checkpoint")
    model.cuda()
    
def imgsave(img, img_name):
    img = torchvision.utils.make_grid(img.detach().cpu())
    #img = torchvision.transforms.ToPILImage()(img)
    #img.save(img_name)
    im_numpy = tensor2im(img)
    im_array = Image.fromarray(im_numpy)
    im_array.save(img_name)
    
def tensor2im(input_image, imtype=np.uint8):
   
    mean = [0.485,0.456,0.406] 
    std = [0.229,0.224,0.225]  
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        for i in range(len(mean)):
            image_numpy[i] = image_numpy[i] * std[i] + mean[i]
        image_numpy = image_numpy * 255
        image_numpy = np.transpose(image_numpy, (1, 2, 0))  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

