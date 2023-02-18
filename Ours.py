from mimetypes import init
from turtle import forward

from torch import conv2d
import torch
import torch.nn as nn
import common
#from model import common
import torch.nn.functional as F
import math 
import cv2
import os
import datetime
import scipy.io as io

import numpy as np

def EzConv(in_channel,out_channel,kernel_size):
    return nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=kernel_size,stride=1,padding=kernel_size//2,bias=True)

class Upsample(nn.Sequential):
    def __init__(self, scale, n_feats, bn=False, act=False, bias=True, conv=EzConv):
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsample, self).__init__(*m)
        
class CA(nn.Module):
    '''CA is channel attention'''
    def __init__(self,n_feats,kernel_size=3,bias=True, bn=False, act=nn.ReLU(True),res_scale=1,conv=EzConv,reduction=16):

        super(CA, self).__init__()
        
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)
        self.body = nn.Sequential(*m)
        
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(n_feats, n_feats // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_feats // reduction, n_feats, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.body(x)
        CA = self.conv_du(y)
        CA = torch.mul(y, CA)
        x = CA + x
        return x
                 
class SCconv(nn.Module):
    def __init__(self,n_feats,kernel_size,pooling_r):
        super(SCconv,self).__init__()
        self.half_feats = n_feats//2
        self.f1 = nn.Sequential(
            nn.Conv2d(n_feats//2,n_feats//2,kernel_size,padding=kernel_size//2),
            nn.ReLU(True)
        )
        self.f2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=pooling_r,stride=pooling_r),
            nn.Conv2d(n_feats//2,n_feats//2,kernel_size,padding=kernel_size//2),
        )
        self.f3 = nn.Conv2d(n_feats//2,n_feats//2,kernel_size,padding=kernel_size//2)
        self.f4 = nn.Sequential(
            nn.Conv2d(n_feats//2,n_feats//2,kernel_size,padding=kernel_size//2),
            nn.ReLU(True)
        )
    
    def forward(self,x):
        x1 = x[:, 0:self.half_feats, :, :]
        x2 = x[:, self.half_feats:, :, :]
        identity_x1 = x1
        out_x1 = torch.sigmoid(torch.add(identity_x1,F.interpolate(self.f2(x1),identity_x1.size()[2:])))
        out_x1 = torch.mul(self.f3(x1),out_x1)
        out_x1 = self.f4(out_x1)
        out_x2 = self.f1(x2)
        out = torch.cat([out_x1,out_x2],dim=1)
        return out

class SSELB(nn.Module):
    def __init__(self,n_feats,kernel_size,pooling_r):
        super(SSELB,self).__init__()
        self.body = nn.Sequential(
            SCconv(n_feats,kernel_size,pooling_r),
            CA(n_feats),
        )

    def forward(self,x):
        res = self.body(x)
        return res + x
        
        
class NGIM(nn.Module):
    def __init__(self,n_feats,scale):
        super(NGIM,self).__init__()
          
        if scale == 4:
            self.TrunkUp = nn.Sequential(
                nn.ConvTranspose2d(n_feats,n_feats,kernel_size=8,stride=4,padding=2),
                nn.PReLU(n_feats)
            )
            self.MultiUp = nn.Sequential(
                nn.Conv2d(n_feats*3,n_feats//2,kernel_size=3,padding=1),
                nn.Conv2d(n_feats//2,n_feats,kernel_size=3,padding=1),
                nn.ConvTranspose2d(n_feats,n_feats,kernel_size=8,stride=4,padding=2),
                nn.PReLU(n_feats)
            )
        elif scale == 8:
            self.TrunkUp = nn.Sequential(
                nn.ConvTranspose2d(n_feats,n_feats,kernel_size=12,stride=8,padding=2),
                nn.PReLU(n_feats)
            )
            self.MultiUp = nn.Sequential(
                nn.Conv2d(n_feats*3,n_feats//2,kernel_size=3,padding=1),
                nn.Conv2d(n_feats//2,n_feats,kernel_size=3,padding=1),
                nn.ConvTranspose2d(n_feats,n_feats,kernel_size=12,stride=8,padding=2),
                nn.PReLU(n_feats)
            )            
        
        self.error_resblock = nn.Sequential(
            nn.Conv2d(n_feats,n_feats,kernel_size=3,padding=1),
        )
    def forward(self,xl,xi,xn):
        
        h1 = self.TrunkUp(xi)
        h2 = self.MultiUp(torch.cat([xl,xi,xn],dim=1))
        e = h2 - h1
        e = self.error_resblock(e)
        h1 = h1 + e
        return h1

class SSELM(nn.Module):
    def __init__(self, n_colors, n_feats, n_blocks,pooling_r):
        super(SSELM, self).__init__()
        kernel_size = 3
        self.head = nn.Conv2d(n_colors, n_feats, kernel_size,padding=kernel_size//2)
        body = []
        for i in range(n_blocks):
            body.append(SSELB(n_feats,kernel_size,pooling_r))

        self.body = nn.Sequential(*body)

        #self.recon = nn.Conv2d(n_feats, n_colors, kernel_size=3,padding=kernel_size//2)

    def forward(self, x):
        x = self.head(x)
        
        y = self.body(x) + x

        return y
   
class GELIN(nn.Module):
    def __init__(self,n_feats,n_colors,kernel_size,pooling_r,n_subs, n_ovls,blocks,scale):
        super(GELIN,self).__init__()

        # calculate the group number (the number of branch networks)
        # 向上取整计算组的数量 G 
        self.n_feats = n_feats
        self.n_subs = n_subs
        self.G = math.ceil((n_colors - n_ovls) / (n_subs - n_ovls))
        # calculate group indices
        self.start_idx = []
        self.end_idx = []        
        self.scale = scale
        for g in range(self.G):
            sta_ind = (n_subs - n_ovls) * g
            end_ind = sta_ind + n_subs
            if end_ind > n_colors:
                end_ind = n_colors
                sta_ind = n_colors - n_subs
            # 把每一组的开始 idx 与结束 idx 存入 list
            self.start_idx.append(sta_ind)
            self.end_idx.append(end_ind)

        self.branch = SSELM(n_subs,n_feats,blocks,pooling_r)
        

        self.branch_up = NGIM(n_feats,scale)
        self.branch_recon = nn.Conv2d(n_feats, n_subs, kernel_size=3,padding=kernel_size//2)

    def forward(self,x,lms):
    
        b, c, h, w = x.shape
        m = []
        y = torch.zeros(b, c,  h*self.scale,  w*self.scale).cuda() 

        channel_counter = torch.zeros(c).cuda()    

        for g in range(self.G):

            sta_ind = self.start_idx[g]
            end_ind = self.end_idx[g]

            xi = x[:, sta_ind:end_ind, :, :]
            xi = self.branch(xi)
            m.append(xi)
            
        for g in range(self.G):

            sta_ind = self.start_idx[g]
            end_ind = self.end_idx[g]

            if g==0:
                xl = m[self.G-1]
                xi = m[g]
                xn = m[g+1]
            elif g==self.G-1:
                xl = m[g-1]
                xi = m[g]
                xn = m[0]
            else:
                xl = m[g-1]
                xi = m[g]
                xn = m[g+1]  

            xi = self.branch_up(xl,xi,xn)
            xi = self.branch_recon(xi)
            y[:, sta_ind:end_ind, :, :] += xi
            # 用 channel_counter 记录某一个位置被加了几次，然后再除这个数字取平均
            channel_counter[sta_ind:end_ind] = channel_counter[sta_ind:end_ind] + 1

        # intermediate “result” is averaged according to their spectral indices
        y = y / channel_counter.unsqueeze(1).unsqueeze(2)
        y = y + lms
        return y
