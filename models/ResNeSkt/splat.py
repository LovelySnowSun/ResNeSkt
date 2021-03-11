"""Split-Attention"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Conv2d, Module, Linear, BatchNorm2d, ReLU
from torch.nn.modules.utils import _pair
from gem_funcation import gem
__all__ = ['SplAtConv2d']
class SpatialSKconv(nn.Module):
    def __init__(self, features,WH, M,G,r, stride=1, L=32):
        """ Constructor (64, 32, 3, 8, 2)
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SpatialSKconv, self).__init__()
        d = max(int(features / r), L)
        #d = 32
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            # 使用不同kernel size的卷积
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(3,
                              1,
                              kernel_size=3 + i * 2,
                              stride=stride,
                              padding=1 + i),
                    nn.BatchNorm2d(1),
                    nn.ReLU(inplace=False)))

        #self.conv = nn.Conv2d(3,1,kernel_size = 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(1, 1)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(1, 1))
        self.softmax = nn.Softmax(dim=1)
 
    def forward(self, x):
        #print("x.size:{}".format(x.shape))
        avgout = torch.mean(x, dim=1, keepdim=True)
        #print("avgout.size:{}".format(avgout.shape))
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        #print("maxout.size:{}".format(maxout.shape))
        gemout = gem(x,keepdims=True)
        #print("gemout.size:{}".format(gemout.shape))
        x = torch.cat([avgout, maxout,gemout], dim=1)
        #print("x.size:{}".format(x.shape))
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
                #print("fea.size:{}".format(fea.shape))
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        #print('fea_U:{}'.format(fea_U.shape))
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        #print("d:{},features:{}".format(d,features))
        for i, fc in enumerate(self.fcs):
            #print(i, fea_z.shape)
            vector = fc(fea_z).unsqueeze_(dim=1)
            #print(i, vector.shape)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector],
                                              dim=1)
        attention_vectors = self.softmax(attention_vectors)
        #attention_vectors = self.softmax(fea_U)#.unsqueeze_(dim=1)
        #print("attention_vectors.size:{}".format(attention_vectors.shape))
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        #print("attention_vectors.size:{}".format(attention_vectors.shape))
        #print("feas.size:{}attention_vectors.size:{}".format(feas.shape,attention_vectors.shape))
        fea_v = (feas * attention_vectors).sum(dim=1)
        #print("fea_v.size:{}".format(fea_v.shape))
        #fea_v = self.conv(fea_v)
        return self.sigmoid(fea_v)
class SplAtConv2d(Module):
    """Split-Attention Conv2d
    """
    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, bias=True,
                 radix=2, reduction_factor=4,
                 rectify=False, rectify_avg=False, norm_layer=None,
                 dropblock_prob=0.0, **kwargs):
        super(SplAtConv2d, self).__init__()
        #print("in_channels:{}.channels:{}".format(in_channels,channels))
        padding = _pair(padding)
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg = rectify_avg
        inter_channels = max(in_channels*radix//reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob
        if self.rectify:
            from rfconv import RFConv2d
            self.conv = RFConv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                                 groups=groups*radix, bias=bias, average_mode=rectify_avg, **kwargs)
        else:
            self.conv = Conv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                               groups=groups*radix, bias=bias, **kwargs)
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = norm_layer(channels*radix)
        self.relu = ReLU(inplace=True)
        self.fc1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = norm_layer(inter_channels)
        self.fc2 = Conv2d(inter_channels, channels*radix, 1, groups=self.cardinality)
        #print("inter_channels:{}.channels*radix:{}".format(inter_channels,channels*radix))
        if dropblock_prob > 0.0:
            self.dropblock = DropBlock2D(dropblock_prob, 3)
        self.rsoftmax = rSoftMax(radix, groups)
        #mid_features = int(3584/(rchannel//self.radix))
        #print(mid_features)
        self.sa_lxy = SpatialSKconv(2,32,3,8,2)

    def forward(self, x):
        #print("x.size：{}".format(x.shape))
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)
        x = self.relu(x)

        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            if torch.__version__ < '1.5':
                splited = torch.split(x, int(rchannel//self.radix), dim=1)
            else:
                splited = torch.split(x, rchannel//self.radix, dim=1)
            gap = sum(splited) 
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)

        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)
        #print("atten.size:{},rchannel//self.radix:{},x:{}".format(atten.shape,rchannel//self.radix,x.shape))
        if self.radix > 1:
            if torch.__version__ < '1.5':
                attens = torch.split(atten, int(rchannel//self.radix), dim=1)
            else:
                attens = torch.split(atten, rchannel//self.radix, dim=1)
            out = sum([att*split for (att, split) in zip(attens, splited)])
            #print("out.size:{}".format(out.shape))
            out = self.sa_lxy(out) * out
        else:
            out = atten * x
            out = self.sa_lxy(out) * out
        return out.contiguous()

class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x

