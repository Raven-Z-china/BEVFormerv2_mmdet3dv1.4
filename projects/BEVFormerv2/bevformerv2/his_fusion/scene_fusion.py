import torch
from torch import nn
from .instance_extraction import TransformerDecoderLayer
from mmcv.cnn import ConvModule

class SceneFusion(torch.nn.Module):
    def __init__(self,dim=512,lks=7,sks=1,group=1):
        self.lkp = LKP(dim, lks=7, sks=1, groups=1)

    def background_extraction(self):
        pass

    def temporal_fusion(self):
        pass

    def forward(self,x):
        x_stk = torch.stack(x,dim=1)
        x_cat = torch.cat(x,dim=1)
        self.lkp(x)
        x_mean = torch.mean(x_stk, dim=1)
        self.bn(self.ska(x_mean, _))
        return

class Cross_Temporal_Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=14):
        super().__init__()



    def forward(self, query, key):

        return x


class LKP(nn.Module):
    def __init__(self, dim, lks, sks, groups):
        super().__init__()
        self.cv1 = ConvModule(dim, dim // 2, norm_cfg = 'BN')
        self.cv2 = ConvModule(dim // 2, dim // 2, ks=lks, pad=(lks - 1) // 2, groups=dim // 2,norm_cfg = 'BN',act_cfg = None)
        self.cv3 = ConvModule(dim // 2, dim // 2,norm_cfg = 'BN')
        self.cv4 = nn.Conv2d(dim // 2, sks ** 2 * dim // groups, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=dim // groups, num_channels=sks ** 2 * dim // groups)

        self.sks = sks
        self.groups = groups
        self.dim = dim

    def forward(self, x):
        x = self.cv3(self.cv2(self.cv1(x)))
        w = self.norm(self.cv4(x))
        b, _, h, width = w.size()
        w = w.view(b, self.dim // self.groups, self.sks ** 2, h, width)
        return w


class LSConv(nn.Module):
    def __init__(self, dim):
        super(LSConv, self).__init__()
        self.lkp = LKP(dim, lks=7, sks=1, groups=8)
        self.ska = SKA()
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        return self.bn(self.ska(x, self.lkp(x)))