import einops
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from ldm.modules.diffusionmodules.openaimodel import TimestepEmbedSequential

from cldm.encoders.MPRnet import *
from cldm.encoders.AIRNet import *
from cldm.encoders.NAFnet import *
from cldm.encoders.Restormer import *

from ldm.modules.diffusionmodules.util import (
    conv_nd,
)

def get_encoder(encoder_type, in_channels, out_channels, level, cntr):
    # print(encoder_type)
    if encoder_type == 0:
        return CLDM_enc_block(in_channels, out_channels, level, cntr)
    elif encoder_type == 1:
        return MPRNet_enc_block(in_channels, out_channels, level, cntr)
    elif encoder_type == 10:
        return Restormer_enc_type_1(in_channels, out_channels, level, cntr)
    elif encoder_type == 4:
        return MPRNet_enc_block_L(in_channels, out_channels, level, cntr)
    elif encoder_type == 5:
        return AIRNet_enc_block_S(in_channels, out_channels, level, cntr)
    elif encoder_type == 6:
        return AIRNet_enc_block_9(in_channels, out_channels, level, cntr)
    elif encoder_type == 7:
        return NAFNet_enc_block_9(in_channels, out_channels, level, cntr)
    elif encoder_type == 11:
        return Restormer_enc_type(in_channels, out_channels, level, cntr)
    elif encoder_type == 2:
        return AIRNet_enc_block(in_channels, out_channels, level, cntr)
    elif encoder_type == 3:
        return NAFNet_enc_block(in_channels, out_channels, level, cntr)
    elif encoder_type == 9:
        return NAFNet_enc_block_7(in_channels, out_channels, level, cntr)
    elif encoder_type == 8:
        return NAFNet_enc_block_old(in_channels, out_channels, level, cntr)
    else:
        assert "Encoder option not present"
        
def CLDM_enc_block(in_channels, out_channels, level, cntr):
    if cntr == 4:
        # print('hello')
        layers = [conv_nd(2, in_channels, out_channels, 3, padding=1, stride=2),
                  nn.SiLU()]
    elif cntr == 1:
        # print('hello1')
        layers = [conv_nd(2, in_channels, out_channels, 3, padding=1),
                nn.SiLU(),
                # conv_nd(2, out_channels, out_channels, 3, padding=1),
                # nn.SiLU(),
                conv_nd(2, out_channels, out_channels, 3, padding=1),
                nn.SiLU()]
    else:
        # print('hello1')
        layers = [conv_nd(2, in_channels, out_channels, 3, padding=1, stride=2),
                nn.SiLU(),
                # conv_nd(2, out_channels, out_channels, 3, padding=1),
                # nn.SiLU(),
                conv_nd(2, out_channels, out_channels, 3, padding=1),
                nn.SiLU()]
    # if level == cntr:
    #     layers.append()
    return TimestepEmbedSequential(*layers)

def Restormer_enc_type(in_channels, out_channels, level, cntr):
    if cntr == 1:
        # layers = [ResBlock(in_feat=in_channels, out_feat=out_channels, stride=1), # 10
        #           ResBlock(in_feat=out_channels, out_feat=out_channels, stride=1),
        #           ResBlock(in_feat=out_channels, out_feat=out_channels, stride=1)]
        layers = [RestormerBlock(in_channels, out_channels, 1, 2, downsample=False)]
    elif cntr == 4:
        layers = [RestormerBlock(in_channels, out_channels, 1, 4, downsample=True)]
        # layers = [ResBlock(in_feat=in_channels, out_feat=out_channels, stride=2), # 10
        #             ResBlock(in_feat=out_channels, out_feat=out_channels, stride=1)]
    else:
        layers = [RestormerBlock(in_channels, out_channels, 2, 6, downsample=True)]
        # layers = [ResBlock(in_feat=in_channels, out_feat=out_channels, stride=2), # 10
        #           ResBlock(in_feat=out_channels, out_feat=out_channels, stride=1),
        #           ResBlock(in_feat=out_channels, out_feat=out_channels, stride=1)]
        # layers = [ResBlock(in_feat=in_channels, out_feat=out_channels, stride=2),]
    return TimestepEmbedSequential(*layers)

def Restormer_enc_type_1(in_channels, out_channels, level, cntr):
    if cntr == 1:
        # layers = [ResBlock(in_feat=in_channels, out_feat=out_channels, stride=1), # 10
        #           ResBlock(in_feat=out_channels, out_feat=out_channels, stride=1),
        #           ResBlock(in_feat=out_channels, out_feat=out_channels, stride=1)]
        layers = [RestormerBlock(in_channels, out_channels, 1, 1, downsample=False)]
    elif cntr == 4:
        layers = [RestormerBlock(in_channels, out_channels, 1, 1, downsample=True)]
        # layers = [ResBlock(in_feat=in_channels, out_feat=out_channels, stride=2), # 10
        #             ResBlock(in_feat=out_channels, out_feat=out_channels, stride=1)]
    else:
        layers = [RestormerBlock(in_channels, out_channels, 1, 1, downsample=True)]
        # layers = [ResBlock(in_feat=in_channels, out_feat=out_channels, stride=2), # 10
        #           ResBlock(in_feat=out_channels, out_feat=out_channels, stride=1),
        #           ResBlock(in_feat=out_channels, out_feat=out_channels, stride=1)]
        # layers = [ResBlock(in_feat=in_channels, out_feat=out_channels, stride=2),]
    return TimestepEmbedSequential(*layers)

def MPRNet_enc_block(in_channels, out_channels, level, cntr):
    layers = []
    if cntr != 1:
        layers = [
            DownSample(in_channels, out_channels - in_channels)
        ]
    else:
        layers = [
            conv(in_channels, out_channels, 3, bias=False), 
            CAB(out_channels, 3, 4, bias=False, act=nn.PReLU()),
        ]
    num_blocks = 2
    if cntr == 4:
        layers.extend([
            *[CAB(out_channels, 3, 4, bias=False, act=nn.PReLU()) for _ in range(1)],
        ])
    elif level != cntr:
        layers.extend([
            *[CAB(out_channels, 3, 4, bias=False, act=nn.PReLU()) for _ in range(num_blocks)],
        ])
    else:
        layers.extend([
            *[CAB(out_channels, 3, 4, bias=False, act=nn.PReLU()) for _ in range(num_blocks)],
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        ])
    return TimestepEmbedSequential(*layers)

def MPRNet_enc_block_L(in_channels, out_channels, level, cntr):
    layers = []
    if cntr != 1:
        layers = [
            DownSample(in_channels, out_channels - in_channels)
        ]
    else:
        layers = [
            conv(in_channels, out_channels, 3, bias=False), 
            CAB(out_channels, 3, 4, bias=False, act=nn.PReLU()),
        ]
    num_blocks = 2
    if cntr == 4:
        layers.extend([
            *[CAB(out_channels, 3, 4, bias=False, act=nn.PReLU()) for _ in range(2)],
        ])
    elif level != cntr:
        layers.extend([
            *[CAB(out_channels, 3, 4, bias=False, act=nn.PReLU()) for _ in range(num_blocks)],
        ])
    else:
        layers.extend([
            *[CAB(out_channels, 3, 4, bias=False, act=nn.PReLU()) for _ in range(num_blocks)],
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        ])
    return TimestepEmbedSequential(*layers)
    
def AIRNet_enc_block(in_channels, out_channels, level, cntr):
    if cntr == 1:
        # layers = [ResBlock(in_feat=in_channels, out_feat=out_channels, stride=1), # 10
        #           ResBlock(in_feat=out_channels, out_feat=out_channels, stride=1),
        #           ResBlock(in_feat=out_channels, out_feat=out_channels, stride=1)]
        layers = [ResBlock(in_feat=in_channels, out_feat=out_channels, stride=1),
                  ResBlock(in_feat=out_channels, out_feat=out_channels, stride=1)]
    elif cntr == 4:
        layers = [ResBlock(in_feat=in_channels, out_feat=out_channels, stride=2)]
        # layers = [ResBlock(in_feat=in_channels, out_feat=out_channels, stride=2), # 10
        #             ResBlock(in_feat=out_channels, out_feat=out_channels, stride=1)]
    else:
        # layers = [ResBlock(in_feat=in_channels, out_feat=out_channels, stride=2), # 10
        #           ResBlock(in_feat=out_channels, out_feat=out_channels, stride=1),
        #           ResBlock(in_feat=out_channels, out_feat=out_channels, stride=1)]
        layers = [ResBlock(in_feat=in_channels, out_feat=out_channels, stride=2),
                  ResBlock(in_feat=out_channels, out_feat=out_channels, stride=1),
                  ResBlock(in_feat=out_channels, out_feat=out_channels, stride=1)]
    return TimestepEmbedSequential(*layers)

def AIRNet_enc_block_S(in_channels, out_channels, level, cntr):
    if cntr == 1:
        # layers = [ResBlock(in_feat=in_channels, out_feat=out_channels, stride=1), # 10
        #           ResBlock(in_feat=out_channels, out_feat=out_channels, stride=1),
        #           ResBlock(in_feat=out_channels, out_feat=out_channels, stride=1)]
        layers = [ResBlock(in_feat=in_channels, out_feat=out_channels, stride=1),]
    elif cntr == 4:
        layers = [ResBlock(in_feat=in_channels, out_feat=out_channels, stride=2)]
        # layers = [ResBlock(in_feat=in_channels, out_feat=out_channels, stride=2), # 10
        #             ResBlock(in_feat=out_channels, out_feat=out_channels, stride=1)]
    else:
        # layers = [ResBlock(in_feat=in_channels, out_feat=out_channels, stride=2), # 10
        #           ResBlock(in_feat=out_channels, out_feat=out_channels, stride=1),
        #           ResBlock(in_feat=out_channels, out_feat=out_channels, stride=1)]
        layers = [ResBlock(in_feat=in_channels, out_feat=out_channels, stride=2),]
    return TimestepEmbedSequential(*layers)


def AIRNet_enc_block_9(in_channels, out_channels, level, cntr):
    if cntr == 1:
        # layers = [ResBlock(in_feat=in_channels, out_feat=out_channels, stride=1), # 10
        #           ResBlock(in_feat=out_channels, out_feat=out_channels, stride=1),
        #           ResBlock(in_feat=out_channels, out_feat=out_channels, stride=1)]
        layers = [ResBlock(in_feat=in_channels, out_feat=out_channels, stride=1),]
    elif cntr == 4:
        # layers = [ResBlock(in_feat=in_channels, out_feat=out_channels, stride=2),
        #           ResBlock(in_feat=in_channels, out_feat=out_channels, stride=1),]
        layers = [ResBlock(in_feat=in_channels, out_feat=out_channels, stride=2)]
    else:
        # layers = [ResBlock(in_feat=in_channels, out_feat=out_channels, stride=2), # 10
        #           ResBlock(in_feat=out_channels, out_feat=out_channels, stride=1),
        #           ResBlock(in_feat=out_channels, out_feat=out_channels, stride=1)]
        layers = [ResBlock(in_feat=in_channels, out_feat=out_channels, stride=2),]
    return TimestepEmbedSequential(*layers)

def NAFNet_enc_block_9(in_channels, out_channels, level, cntr):
    # num_blocks = 2 (4, 2 for large batch model 10)
    num_blocks = 2 # 
    if cntr == 1:
        layers = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                      kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True),
            *[NAFBlock(out_channels) for _ in range(num_blocks)]
        ]
    elif cntr == 4:
        layers = [
            conv_nd(2, in_channels, out_channels, 3, padding=1, stride=2),
            *[NAFBlock(out_channels) for _ in range(num_blocks*4)] # 11
            # *[NAFBlock(out_channels) for _ in range(2)] # 7
        ]
    else:
        layers = [
            conv_nd(2, in_channels, out_channels, 3, padding=1, stride=2),
            *[NAFBlock(out_channels) for _ in range(num_blocks*2)]
        ]
    if level != cntr and cntr != 4:
        layers.append(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, 
                                kernel_size=1, padding=0, stride=1, groups=1,
                              bias=True))
    return TimestepEmbedSequential(*layers)
    
def NAFNet_enc_block(in_channels, out_channels, level, cntr):
    # num_blocks = 2 (4, 2 for large batch model 10)
    num_blocks = 2 # 
    if cntr == 1:
        layers = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                      kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True),
            *[NAFBlock(out_channels) for _ in range(1)]
        ]
    elif cntr == 4:
        layers = [
            conv_nd(2, in_channels, out_channels, 3, padding=1, stride=2),
            *[NAFBlock(out_channels) for _ in range(num_blocks)] # 11
            # *[NAFBlock(out_channels) for _ in range(2)] # 7
        ]
    else:
        layers = [
            conv_nd(2, in_channels, out_channels, 3, padding=1, stride=2),
            *[NAFBlock(out_channels) for _ in range(num_blocks)]
        ]
    if level != cntr and cntr != 4:
        layers.append(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, 
                                kernel_size=1, padding=0, stride=1, groups=1,
                              bias=True))
    return TimestepEmbedSequential(*layers)

def NAFNet_enc_block_old(in_channels, out_channels, level, cntr):
    # num_blocks = 2 (4, 2 for large batch model 10)
    num_blocks = 2 # 
    if cntr == 1:
        layers = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                      kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True),
            *[NAFBlock(out_channels) for _ in range(1)]
        ]
    elif cntr == 4:
        layers = [
            conv_nd(2, in_channels, out_channels, 3, padding=1, stride=2),
            *[NAFBlock(out_channels) for _ in range(num_blocks*2)] # 11
            # *[NAFBlock(out_channels) for _ in range(2)] # 7
        ]
    else:
        layers = [
            conv_nd(2, in_channels, out_channels, 3, padding=1, stride=2),
            *[NAFBlock(out_channels) for _ in range(num_blocks)]
        ]
    if level != cntr and cntr != 4:
        layers.append(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, 
                                kernel_size=1, padding=0, stride=1, groups=1,
                              bias=True))
    return TimestepEmbedSequential(*layers)

def NAFNet_enc_block_7(in_channels, out_channels, level, cntr):
    # num_blocks = 2 (4, 2 for large batch model 10)
    num_blocks = 2 # 
    if cntr == 1:
        layers = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                      kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True),
            *[NAFBlock(out_channels) for _ in range(num_blocks)]
        ]
    elif cntr == 4:
        layers = [
            conv_nd(2, in_channels, out_channels, 3, padding=1, stride=2),
            *[NAFBlock(out_channels) for _ in range(2)] # 11
            # *[NAFBlock(out_channels) for _ in range(2)] # 7
        ]
    else:
        layers = [
            conv_nd(2, in_channels, out_channels, 3, padding=1, stride=2),
            *[NAFBlock(out_channels) for _ in range(num_blocks)]
        ]
    if level != cntr and cntr != 4:
        layers.append(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, 
                                kernel_size=1, padding=0, stride=1, groups=1,
                              bias=True))
    return TimestepEmbedSequential(*layers)
