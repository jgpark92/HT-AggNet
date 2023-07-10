import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def channel_shuffle(x, groups):
    batchsize, num_channels, T = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, 
        channels_per_group, T)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, T)
    return x

class TAggBlock(nn.Module):
    def __init__(self, i_nc, o_nc, L, T):
        super(TAggBlock, self).__init__()
        self.L = L

        # local temperol convolution
        self.gconv = nn.Conv1d(i_nc, o_nc, kernel_size=5, padding='same', bias=False, groups=L)
        self.bn = nn.BatchNorm1d(o_nc)

        # temporal glance convolution
        self.tgconv = nn.Conv1d(o_nc, o_nc//L, kernel_size=T)
    
    def forward(self, x):
        x = self.gconv(x)
        x = F.relu_(self.bn(x))
        l_feat = channel_shuffle(x, self.L)
        g_feat = self.tgconv(l_feat)
        return l_feat, g_feat

class HTAggNet(nn.Module):
    def __init__(self, nc_input, n_classes, segment_size):

        super(HTAggNet, self).__init__()
        T = segment_size
        self.nc_o = 128
        self.L = 8

        self.stem = nn.Conv1d(nc_input, self.nc_o, 5, padding='same')
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        T = math.ceil(T / 2)

        self.layers = nn.ModuleList()
        for _ in range(self.L):
            self.layers.append(TAggBlock(self.nc_o, self.nc_o, self.L, T))
        
        self.fc = nn.Linear(self.nc_o, n_classes)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.maxpool(x)

        out = []
        for block in self.layers:
            x, g_feat = block(x)
            out.append(g_feat)
        
        out = torch.cat(out, dim=1)

        out = out.view(out.size(0), -1)

        logits = self.fc(out)

        return F.log_softmax(logits, dim=1)