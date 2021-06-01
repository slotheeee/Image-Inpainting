#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 30 12:49:35 2021

@author: maggie
"""

import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch import cat
import numpy as np

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class Generator(nn.Module):
    
    def upscale(self, input_dim, mid_dim, output_dim, kernel = 1, stride = 1, padding = 0):
        return nn.Sequential(
            nn.Conv2d(input_dim, mid_dim, kernel, stride, padding),
            nn.ReLU(),
            nn.Conv2d(mid_dim, output_dim, kernel, stride, padding),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
            )
    def set_parameter_require_grad(self, model, feature_extract):
        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False
                
    def initial_model(self, model_name, feature_extract):
        model_ft = None
        if model_name == 'ResNet':
            model_ft = models.resnet50(pretrained = True)
            self.set_parameter_require_grad(model_ft, feature_extract)
            in_ftrs = model_ft.fc.in_features
            model_ft.avgpool = Identity()
            model_ft.fc = Identity()
        return model_ft, in_ftrs
    

    def __init__(self, model_name = 'ResNet', input_dim = 3, output_dim = 3, feature_extract = True ):

        super(Generator, self).__init__()
        self.model_name = model_name
        self.init_model, self.in_ftrs = self.initial_model(self.model_name, feature_extract)
        
        self.up1 = self.upscale(self.in_ftrs, int(self.in_ftrs/2), int(self.in_ftrs/2))#7->14
        self.up2 = self.upscale(int(self.in_ftrs/2), int(self.in_ftrs/4), int(self.in_ftrs/4))#14->28
        self.up3 = self.upscale(int(self.in_ftrs/4), int(self.in_ftrs/8), int(self.in_ftrs/16))#28->56
        self.up4 = self.upscale(int(self.in_ftrs/16), int(self.in_ftrs/32), int(self.in_ftrs/64))#56->112
        self.up5 = self.upscale(int(self.in_ftrs/64), int(self.in_ftrs/128), output_dim)#112->224
    

    def forward(self, input):
        x = self.init_model(input)
        if self.model_name == 'ResNet':
            x = x.view(-1, self.in_ftrs, 7, 7)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        
        return x
    
class Discriminator(nn.Module):

    def uniform(self, stdev, size):
        return np.random.uniform(
            low=-stdev * np.sqrt(3),
            high=stdev * np.sqrt(3),
            size=size
        ).astype('float32')
    
    def LeakyReLU(self, x, alpha=0.2):
        #self.lrelu = nn.LeakyReLU(alpha, inplace=False)
        #return self.lrelu(x)
        return torch.max(alpha*x, x).clone()

    def conv2d(self, x, input_dim, filter_size, output_dim, gain=1, stride=1, padding=2):
        filter_values = self.uniform(
                self._weights_stdev,
                (output_dim, input_dim, filter_size, filter_size)
            )
        filter_values *= gain
        filters = torch.from_numpy(filter_values)
        biases = torch.from_numpy(np.zeros(output_dim, dtype='float32'))
        if self.use_gpu:
            filters = filters.cuda()
            biases = biases.cuda()
        result = nn.functional.conv2d(x, filters, biases, stride, padding)
        return result
        
    def LayerNorm(self, ch):
        return nn.BatchNorm2d(ch)
        
    def __init__(self, bn=True, input_dim=3, dim=64, _weights_stdev=0.02, use_gpu=True):
        super(Discriminator, self).__init__()
        self.bn = bn
        self.input_dim = input_dim
        self.dim = dim
        self._weights_stdev = _weights_stdev
        self.use_gpu = use_gpu

        self.bn1 = self.LayerNorm(2*self.dim)
        self.bn2 = self.LayerNorm(4*self.dim)
        self.bn3 = self.LayerNorm(8*self.dim)
        self.bn4 = self.LayerNorm(8*self.dim)
        
        self.fc1 = nn.Linear(7*7*8*self.dim, 1)
        
    def forward(self, x):
        output = x
#        print("in D: ",output.size())
        output = self.conv2d(output, self.input_dim, 5, self.dim, stride=2)
#        print("1 conv: ", output.size())
        output = self.LeakyReLU(output)
#        print("ReLU: ", output.size())
        
        output = self.conv2d(output, self.dim, 5, 2*self.dim, stride=2)
#        print("1 conv: ", output.size())
        
        if self.bn:
            output = self.bn1(output)
#            print("2 in bn: ", output.size())
        
        output = self.LeakyReLU(output)
#        print("2 ReLU: ", output.size())
        
        output = self.conv2d(output, 2*self.dim, 5, 4*self.dim, stride=2)
#        print("2 conv: ", output.size())
        
        if self.bn:
            output = self.bn2(output)
#            print("3 in bn: ", output.size())
        
        output = self.LeakyReLU(output)
#        print("3 ReLU: ", output.size())
        
        output = self.conv2d(output, 4*self.dim, 5, 8*self.dim, stride=2)
#        print("3 conv: ", output.size())
        
        if self.bn:
            output = self.bn3(output)
#            print("4 in bn: ", output.size())
        
        output = self.LeakyReLU(output)
#        print("4 ReLU: ", output.size())
        
        output = self.conv2d(output, 8*self.dim, 5, 8*self.dim, stride=2)
        #print("4 conv: ", output.size())
        
        if self.bn:
            output = self.bn4(output)
            #print("5 in bn: ", output.size())
        
        output = self.LeakyReLU(output)
        #print("5 ReLU: ", output.size())
        
        #print("self.dim", self.dim)
        output = output.view(-1, 7*7*8*self.dim)
        #print("view: ", output.shape)
        output = self.fc1(output)
        #print('discriminator: ', output.shape)
        return output
    
if __name__ == '__main__':
    generator = Generator(model_name='ResNet')
    generator.cuda()
    
    x = np.zeros((4, 3, 224, 224))
    x = torch.from_numpy(x)
    print('brfore: ', x.shape)
    
    x = generator(x)
    print('after:', x.shape )
