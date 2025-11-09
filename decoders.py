import torch
import torch.nn as nn
from functools import partial
from timm.models import checkpoint
from geoseg.models.FQGU import FQGU
import math
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply
from geoseg.models.SEFA import SEFA
import torch.nn.functional as F

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        trunc_normal_tf_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

class SAB(nn.Module):
    def __init__(self, in_channels, out_channels=None, expand_ratio=2, kernel_size=3, stride=1,
                 reduction=4, group_width=16, group_count=1,
                 block_count=1, se=True, sa=False, res_scale=1.0,
                 act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d,
                 drop_block=None, drop_path_rate=0.,
                 use_aa=True, aa_layer=nn.AvgPool2d,
                 use_checkpoint=False,
                 init_scheme='kaiming_normal', init_gain=1.0):
        super(SAB, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.res_scale = res_scale
        if in_channels != out_channels:
            self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.res_conv = nn.Identity()
        self.sab = nn.Sequential(*[
            SABlock(
                in_channels, out_channels, expand_ratio, kernel_size, stride,
                reduction, group_width, group_count,
                se, sa,
                act_layer, norm_layer, drop_block, use_aa, aa_layer)
            for _ in range(block_count)
        ])
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.init_weights(init_scheme=init_scheme, init_gain=init_gain)

    def init_weights(self, init_scheme='kaiming_normal', init_gain=1.0):
        named_apply(partial(_init_weights, scheme=init_scheme), self)

    def forward(self, x):
        res_x = self.res_conv(x)
        if self.use_checkpoint:
            x = checkpoint(self.sab, x)
        else:
            x = self.sab(x)
        x = self.drop_path(x)
        x = x * self.res_scale + res_x
        return x

class SABlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, expand_ratio=2, kernel_size=3, stride=1,
                 reduction=4, group_width=16, group_count=1,
                 se=True, sa=False,
                 act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d,
                 drop_block=None,
                 use_aa=True, aa_layer=nn.AvgPool2d):
        super(SABlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        if stride != 1:
            self.avd = aa_layer(stride, stride) if use_aa else nn.AvgPool2d(stride, stride)
        else:
            self.avd = nn.Identity()
        if group_count > 1:
            conv_layer = partial(GroupConv2d, group_width=group_width)
            expand_channels = group_width * group_count * expand_ratio
            mid_channels = group_width * group_count
        else:
            conv_layer = nn.Conv2d
            expand_channels = in_channels * expand_ratio
            mid_channels = out_channels
        self.conv1 = conv_layer(in_channels, expand_channels, kernel_size=1, bias=False)
        self.bn1 = norm_layer(expand_channels)
        self.act1 = act_layer(inplace=True)
        self.conv2 = conv_layer(expand_channels, expand_channels, kernel_size=kernel_size, stride=1,
                              groups=expand_channels, bias=False)
        self.bn2 = norm_layer(expand_channels)
        self.act2 = act_layer(inplace=True)
        if sa:
            self.sa = SA(expand_channels, reduction=reduction, use_efficient_sa=True)
        else:
            self.sa = nn.Identity()
        if se:
            self.se = SE(expand_channels, reduction=reduction)
        else:
            self.se = nn.Identity()
        self.conv3 = conv_layer(expand_channels, mid_channels, kernel_size=1, bias=False)
        self.bn3 = norm_layer(mid_channels)
        self.act3 = act_layer(inplace=True)
        self.conv4 = conv_layer(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn4 = norm_layer(out_channels)
        self.drop_block = drop_block

    def forward(self, x):
        x = self.avd(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.sa(x)
        x = self.se(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        return x

class CAB(nn.Module):
    def __init__(self, in_channels, out_channels=None, expand_ratio=2, kernel_size=3, stride=1,
                 reduction=4,
                 act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d,
                 drop_block=None,
                 use_aa=True, aa_layer=nn.AvgPool2d):
        super(CAB, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        if stride != 1:
            self.avd = aa_layer(stride, stride) if use_aa else nn.AvgPool2d(stride, stride)
        else:
            self.avd = nn.Identity()
        expand_channels = in_channels * expand_ratio
        mid_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, expand_channels, kernel_size=1, bias=False)
        self.bn1 = norm_layer(expand_channels)
        self.act1 = act_layer(inplace=True)
        self.conv2 = nn.Conv2d(expand_channels, expand_channels, kernel_size=kernel_size, stride=1,
                               groups=expand_channels, bias=False)
        self.bn2 = norm_layer(expand_channels)
        self.act2 = act_layer(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv3 = nn.Conv2d(expand_channels, mid_channels, kernel_size=1, bias=False)
        self.bn3 = norm_layer(mid_channels)
        self.act3 = nn.Sigmoid()
        self.conv4 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn4 = norm_layer(out_channels)
        self.drop_block = drop_block

    def forward(self, x):
        x = self.avd(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        avg_x = self.avg_pool(x)
        max_x = self.max_pool(x)
        avg_x = self.conv3(avg_x)
        max_x = self.conv3(max_x)
        avg_x = self.bn3(avg_x)
        max_x = self.bn3(max_x)
        x = self.act3(avg_x + max_x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.conv4(x)
        x = self.bn4(x)
        return x

class SE(nn.Module):
    def __init__(self, in_channels, reduction=4, act_layer=nn.ReLU6,
                 sigmoid_act=nn.Sigmoid):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=True)
        self.gate = sigmoid_act()

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate(x_se)
        return x

class SA(nn.Module):
    def __init__(self, in_channels, reduction=4, use_efficient_sa=True):
        super(SA, self).__init__()
        self.use_efficient_sa = use_efficient_sa
        if self.use_efficient_sa:
            self.conv_reduce = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=True)
            self.conv_expand = nn.Conv2d(in_channels // reduction, 1, 1, bias=True)
        else:
            self.conv = nn.Conv2d(in_channels, 1, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.use_efficient_sa:
            x_se = self.conv_reduce(x)
            x_se = self.conv_expand(x_se)
        else:
            x_se = self.conv(x)
        x = x * self.sigmoid(x_se)
        return x

class GroupConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False,
                 group_width=16):
        super(GroupConv2d, self).__init__()
        assert in_channels % group_width == 0
        assert out_channels % group_width == 0
        self.group_width = group_width
        self.groups = in_channels // group_width
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=self.groups, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        return x

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class FQPD(nn.Module):
    def __init__(self,
                 channels=[64, 128, 320, 512],
                 decode_channels=64,
                 ):
        super().__init__()
        self.sefa1 = SEFA(input_dim=channels[2], embed_dim=channels[2], v_dim=channels[2], window_size=7)
        self.sefa2 = SEFA(input_dim=channels[1], embed_dim=channels[1], v_dim=channels[1], window_size=7)
        self.sefa3 = SEFA(input_dim=channels[0], embed_dim=channels[0], v_dim=channels[0], window_size=7)

        self.freq3 = FQGU(hr_channels=channels[1], lr_channels=channels[0], compressed_channels=320)
        self.freq2 = FQGU(hr_channels=channels[2], lr_channels=channels[1], compressed_channels=128)
        self.freq1 = FQGU(hr_channels=channels[3], lr_channels=channels[2], compressed_channels=64)

        self.cab4 = CAB(channels[0])
        self.cab3 = CAB(channels[1])
        self.cab2 = CAB(channels[2])
        self.cab1 = CAB(channels[3])

        self.sab = SAB()

    def forward(self, x, skips):
        d4 = self.cab4(x) * x
        d4 = self.sab(d4) * d4

        _, _, d3 = self.freq3(hr_feat=skips[0], lr_feat=d4)
        x3 = self.sefa3(e=skips[0], q=d3)
        d3 = d3 + x3
        d3 = self.cab3(d3) * d3
        d3 = self.sab(d3) * d3

        _, _, d2 = self.freq2(hr_feat=skips[1], lr_feat=d3)
        x2 = self.sefa2(e=skips[1], q=d2)
        d2 = d2 + x2
        d2 = self.cab2(d2) * d2
        d2 = self.sab(d2) * d2

        _, _, d1 = self.freq1(hr_feat=skips[2], lr_feat=d2)
        x1 = self.sefa1(e=skips[2], q=d1)
        d1 = d1 + x1
        d1 = self.cab1(d1) * d1
        d1 = self.sab(d1) * d1

        return [d4, d3, d2, d1]