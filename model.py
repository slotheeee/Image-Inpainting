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
'''
class Generator(nn.Module):#(21, config.z_num, config.repeat_num, config.hidden_num)
    
    def block(self, ch_in, ch_out, kernel, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel, stride, padding),
            nn.ReLU(),
            nn.Conv2d(ch_out, ch_out, kernel, stride, padding),
            nn.ReLU()
        )
    
    def block_one(self, ch_in, ch_out, kernel, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel, stride, padding),
            nn.ReLU()
        )
    
    def conv(self, ch_in, ch_out, kernel, stride=1, padding=1):
        return nn.Conv2d(ch_in, ch_out, kernel, stride, padding)
        
    def fc(self, ch_in, ch_out):
        return nn.Linear(ch_in, ch_out)
    
    def __init__(self, ch_in = 3, z_num = 64, repeat_num = 6, hidden_num=128):
        super(Generator, self).__init__()
        self.min_fea_map_H = 16
        self.min_fea_map_W = 16
        self.z_num = z_num 
        self.hidden_num = hidden_num 
        self.repeat_num = repeat_num
        
        self.block_1 = self.block_one(ch_in, self.hidden_num, 3, 1)

        self.block1 = self.block(self.hidden_num, 128, 3, 1)
        self.block2 = self.block(256, 256, 3, 1)
        self.block3 = self.block(384, 384, 3, 1)
        self.block4 = self.block(512, 512, 3, 1)
        self.block5 = self.block(640, 640, 3, 1)
        self.block6 = self.block(768, 768, 3, 1)
            
        self.block_one1 = self.block_one(128, 256, 3, 2)
        self.block_one2 = self.block_one(256, 384, 3, 2)
        self.block_one3 = self.block_one(384, 512, 3, 2)
        self.block_one4 = self.block_one(512, 640, 3, 2)
        self.block_one5 = self.block_one(640, 768, 3, 2)
        
        self.fc1 = self.fc(self.min_fea_map_H * self.min_fea_map_W * 768, self.z_num)
        self.fc2 = self.fc(self.z_num, self.min_fea_map_H * self.min_fea_map_W * self.hidden_num)
        
        self.block7 = self.block(896, 896, 3, 1)
        self.block8 = self.block(1280, 1280, 3, 1)
        self.block9 = self.block(1024, 1024, 3, 1)
        self.block10 = self.block(768, 768, 3, 1)
        self.block11 = self.block(512, 512, 3, 1)
        self.block12 = self.block(256, 256, 3, 1)
        
        self.block_one6 = self.block_one(896, 640, 1, 1, padding=0)
        self.block_one7 = self.block_one(1280, 512, 1, 1, padding=0)
        self.block_one8 = self.block_one(1024, 384, 1, 1, padding=0)
        self.block_one9 = self.block_one(768, 256, 1, 1, padding=0)
        self.block_one10 = self.block_one(512, 128, 1, 1, padding=0)
        
        self.conv_last = self.conv(256, 3, 3, 1) 
        
        self.upscale = nn.Upsample(scale_factor=2)
        
    def forward(self, x):
        encoder_layer_list = []
#        print("in G: ",x.size())
        x = self.block_1(x)
#        print("block_1: ",x.size())
        
        # 1st encoding layer
        res = x
        x = self.block1(x)
#        print("en1 block_1: ",x.size())
        
        x = x + res
#        print("en1 x+res: ",x.size())
        
        encoder_layer_list.append(x)
        x = self.block_one1(x)
#        print("en1 block_1: ",x.size())
        
        # 2nd encoding layer
        res = x
        x = self.block2(x)
#        print("en2 block_2: ",x.size())
        
        x = x + res
#        print("en2 block_2+res: ",x.size())
        
        encoder_layer_list.append(x)
        x = self.block_one2(x)
#        print("en2 block_2: ",x.size())
        
        # 3rd encoding layer
        res = x
        x = self.block3(x)
#        print("en3 block_3: ",x.size())
        
        x = x + res
#        print("en3 block_3+res: ",x.size())
        
        encoder_layer_list.append(x)
        x = self.block_one3(x)
#        print("en3 block_3: ",x.size())
        
        # 4th encoding layer
        res = x
        x = self.block4(x)
#        print("en4 block_4: ",x.size())
        
        x = x + res
#        print("en4 block_4+res: ",x.size())
        
        encoder_layer_list.append(x)
        x = self.block_one4(x)
#        print("en4 block_4: ",x.size())
        
        # 5th encoding layer
        res = x
        x = self.block5(x)
#        print("en5 block_5: ",x.size())
        
        x = x + res
#        print("en5 block_5+res: ",x.size())
        
        encoder_layer_list.append(x)
        x = self.block_one5(x)
#        print("en5 block_5: ",x.size())
        
        # 6th encoding layer
        res = x
        x = self.block6(x)
#        print("en6 block_6: ",x.size())
        
        x = x + res
#        print("en6 block_6+res: ",x.size())
        
        encoder_layer_list.append(x)
            
        x = x.view(-1, self.min_fea_map_H * self.min_fea_map_W * 768)
#        print("x.view: ",x.size())
        x = self.fc1(x)
#        print("fc1(x): ", x.size())
        z = x
        
        x = self.fc2(z)
#        print("fc2(z)= ", x.size())
        x = x.view(-1, self.hidden_num, self.min_fea_map_H, self.min_fea_map_W)
#        print("x.view: ", x.size())
        # 1st decoding layer
        x = torch.cat([x, encoder_layer_list[5]], dim=1)
#        print("de1 torch cat: ",x.size())
        res = x
        x = self.block7(x)
#        print("de1 block_7: ",x.size())
        
        x = x + res
#        print("de1 x+res: ",x.size())
        
        x = self.upscale(x)
#        print("de1 upsample: ",x.size())
        
        x = self.block_one6(x)
#        print("de1 block_one6: ",x.size())
        
        # 2nd decoding layer
        x = torch.cat([x, encoder_layer_list[4]], dim=1)
        res = x
        x = self.block8(x)
#        print("de2 block_8: ",x.size())
        x = x + res
        x = self.upscale(x)
        x = self.block_one7(x)
#        print("de2 block_one7: ", x.size())
        # 3rd decoding layer
        x = torch.cat([x, encoder_layer_list[3]], dim=1)
        res = x
        x = self.block9(x)
#        print("de3 block_9: ",x.size())
        
        x = x + res
        x = self.upscale(x)
        x = self.block_one8(x)
#        print("de3 block_one8: ", x.size())
        # 4th decoding layer
        x = torch.cat([x, encoder_layer_list[2]], dim=1)
        res = x
        x = self.block10(x)
#        print("de4 block_10: ",x.size())
        x = x + res
#        print("de4 x+res: ", x.size())
        x = self.upscale(x)
#        print("de4 upscale: ", x.size())
        x = self.block_one9(x)
#        print("de4 block_one9: ", x.size())
        # 5th decoding layer
        x = torch.cat([x, encoder_layer_list[1]], dim=1)
        res = x
        x = self.block11(x)
#        print("de5 block_11: ",x.size())
        x = x + res
        x = self.upscale(x)
        x = self.block_one10(x)
        # 6th decoding layer
        x = torch.cat([x, encoder_layer_list[0]], dim=1)
        res = x
        x = self.block12(x)
#        print("de6 block_12: ",x.size())
        x = x + res
        
        output = self.conv_last(x)
#        print("output: ", output.size())
        return output
'''    
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
