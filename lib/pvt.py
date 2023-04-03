import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pvtv2 import pvt_v2_b2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from itertools import repeat
import collections.abc


import math


class conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, padding='same',
                 bias=False, bn=True, relu=False):
        super(conv, self).__init__()
        if '__iter__' not in dir(kernel_size):
            kernel_size = (kernel_size, kernel_size)
        if '__iter__' not in dir(stride):
            stride = (stride, stride)
        if '__iter__' not in dir(dilation):
            dilation = (dilation, dilation)

        if padding == 'same':
            width_pad_size = kernel_size[0] + (kernel_size[0] - 1) * (dilation[0] - 1)
            height_pad_size = kernel_size[1] + (kernel_size[1] - 1) * (dilation[1] - 1)
        elif padding == 'valid':
            width_pad_size = 0
            height_pad_size = 0
        else:
            if '__iter__' in dir(padding):
                width_pad_size = padding[0] * 2
                height_pad_size = padding[1] * 2
            else:
                width_pad_size = padding * 2
                height_pad_size = padding * 2

        width_pad_size = width_pad_size // 2 + (width_pad_size % 2 - 1)
        height_pad_size = height_pad_size // 2 + (height_pad_size % 2 - 1)
        pad_size = (width_pad_size, height_pad_size)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_size, dilation, groups, bias=bias)
        self.reset_parameters()

        if bn is True:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None

        if relu is True:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv.weight)


class axis_attn(nn.Module):
    def __init__(self, in_channels, mode='hw'):
        super(axis_attn, self).__init__()

        self.mode = mode

        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=(1, 1), padding=(0, 0))
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=(1, 1), padding=(0, 0))
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), padding=(0, 0))

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channel, height, width = x.size()

        axis = 1
        if 'h' in self.mode:
            axis *= height
        if 'w' in self.mode:
            axis *= width

        view = (batch_size, -1, axis)

        projected_query = self.query_conv(x).view(*view).permute(0, 2, 1)
        projected_key = self.key_conv(x).view(*view)

        attention_map = torch.bmm(projected_query, projected_key)
        attention = self.softmax(attention_map)
        projected_value = self.value_conv(x).view(*view)

        out = torch.bmm(projected_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channel, height, width)

        out = self.gamma * out + x
        return out

class VHAA(nn.Module):
    def __init__(self, in_channel, out_channel, receptive_size=3):
        super(VHAA, self).__init__()
        self.conv0 = nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), padding=(0, 0))
        self.conv1 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, receptive_size), padding=(0, ((receptive_size-1) // 2)))
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(receptive_size, 1), padding=(((receptive_size-1) // 2),0))
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(True)
        self.conv3 = nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), dilation=receptive_size, padding=receptive_size)
        self.Hattn = axis_attn(out_channel, mode='h')
        self.Wattn = axis_attn(out_channel, mode='w')

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)

        Hx = self.Hattn(x)
        Wx = self.Wattn(x)

        x = self.conv3(Hx + Wx)
        return x

class conv2d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=(1,1), padding=(1,1), dilation=1, act=True):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.BatchNorm2d):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    #     elif isinstance(m, nn.Conv2d):
    #         fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         fan_out //= m.groups
    #         m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
    #         if m.bias is not None:
    #             m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class TransferLayer(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=[1,3,5], stride=1, padding=[0,1,2], dilation=1):
        super(TransferLayer, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, out_planes,
                               kernel_size=kernel_size[0], stride=stride,
                               padding=padding[0], dilation=dilation, bias=False)

        self.conv3 = nn.Conv2d(in_planes, out_planes,
                               kernel_size=kernel_size[1], stride=stride,
                               padding=padding[1], dilation=dilation, bias=False)

        self.conv5 = nn.Conv2d(in_planes, out_planes,
                               kernel_size=kernel_size[2], stride=stride,
                               padding=padding[2], dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.bn5 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x5 = self.conv5(x)
        x5 = self.bn5(x5)
        x = x1 + x3 + x5
        return x








class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)

        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class DWConv_Mulit(nn.Module):
    def __init__(self, dim=768):
        super(DWConv_Mulit, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv_Mulit(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)


    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)


    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, q, k


class Bottleneck(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Bottleneck, self).__init__()
        self.map = nn.Conv2d(in_planes, out_planes, kernel_size=1, padding=0, bias=False)
        self.conv0 = nn.Conv2d(in_planes, out_planes // 4, kernel_size=1, padding=0, bias=False)
        self.conv1 = nn.Conv2d(out_planes // 4, out_planes // 4, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_planes // 4, out_planes, kernel_size=1, padding=0, bias=False)
        self.bn0 = nn.BatchNorm2d(out_planes // 4)
        self.bn1 = nn.BatchNorm2d(out_planes // 4)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.bn_map = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x_ = self.bn_map(self.map(x))
        x = self.relu(self.bn0(self.conv0(x)))
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(x_ + self.bn2(self.conv2(x)))
        return x

class Linear_Eca_block(nn.Module):
    """docstring for Eca_block"""
    def __init__(self):
        super(Linear_Eca_block, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=5, padding=int(5/2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, gamma=2, b=1):
        #N, C, H, W = x.size()
        y = self.avgpool(x)
        y = self.conv1d(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return y.expand_as(x)

class MBSA(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(MBSA, self).__init__()

        self.eca = Linear_Eca_block()
        self.conv = BasicConv2d(in_planes // 2, out_planes // 2, 3, 1, 1)
        self.down_c = BasicConv2d(out_planes//2, 1, 3, 1, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 1),
            nn.BatchNorm2d(out_planes))
        self.branch1 = VHAA(in_planes, out_planes, 3)
        self.branch2 = VHAA(in_planes, out_planes, 7)
        self.branch3 = VHAA(in_planes, out_planes, 9)
        self.conv_cat = conv2d(4 * out_planes, out_planes, kernel_size=3, padding=1, act=False)
        self.conv_res = conv2d(in_planes, out_planes, kernel_size=1, padding=0, act=False)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        c = x.shape[1]
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        x_t, x_c = torch.split(x, c // 2, dim=1)
        sa = self.sigmoid(self.down_c(x_c))
        gc = self.eca(x_t)
        x_c = self.conv(x_c)
        x_c = x_c * gc
        x_t = x_t * sa
        x = torch.cat((x_t, x_c), 1)
        return x

class Trans(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Trans, self).__init__()

        self.patch_embed = OverlapPatchEmbed(img_size=224 // 4, patch_size=3, stride=1, in_chans=in_planes,
                                             embed_dim=out_planes)
        self.block = Block(dim=out_planes)
        self.norm = nn.LayerNorm(out_planes)
        self.gc = Linear_Eca_block()
        self.conv = Bottleneck(in_planes, out_planes)
        self.down_c = BasicConv2d(out_planes, 1, 3, 1, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        B = x.shape[0]
        #c = x.shape[1]
        #x_t, x_c = torch.split(x, c // 2, dim=1)
        x_t, H, W = self.patch_embed(x)
        x_t, q, k = self.block(x_t, H, W)
        x_t = self.norm(x_t)
        x_t = x_t.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        q = q.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        k = q.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        atten = q * k
        atten_c = self.gc(atten)
        atten_s = self.sigmoid(self.down_c(atten))
        x_t = x_t * atten_c * atten_s
        #x_t = self.upsample(x_t)
        x_c = self.conv(x)
        #x_c = self.upsample(x_c)
        x = x_t * x_c
        x = self.upsample(x)
        return x



class Block(nn.Module):

    def __init__(self, dim, num_heads=1, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        msa, q, k = self.attn(self.norm1(x), H, W)
        x = x + msa
        x = x + self.mlp(self.norm2(x), H, W)

        return x, q, k


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

def _get_act_fn(act_name, inplace=True):
    if act_name == "relu":
        return nn.ReLU(inplace=inplace)
    elif act_name == "leaklyrelu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
    elif act_name == "gelu":
        return nn.GELU()
    else:
        raise NotImplementedError

class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        act_name="relu",
        is_transposed=False,
    ):
        """
        Convolution-BatchNormalization-ActivationLayer

        :param in_planes:
        :param out_planes:
        :param kernel_size:
        :param stride:
        :param padding:
        :param dilation:
        :param groups:
        :param bias:
        :param act_name: None denote it doesn't use the activation layer.
        :param is_transposed: True -> nn.ConvTranspose2d, False -> nn.Conv2d
        """
        super().__init__()
        if is_transposed:
            conv_module = nn.ConvTranspose2d
        else:
            conv_module = nn.Conv2d
        self.add_module(
            name="conv",
            module=conv_module(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=to_2tuple(stride),
                padding=to_2tuple(padding),
                dilation=to_2tuple(dilation),
                groups=groups,
                bias=bias,
            ),
        )
        self.add_module(name="bn", module=nn.BatchNorm2d(out_planes))
        if act_name is not None:
            self.add_module(name=act_name, module=_get_act_fn(act_name=act_name))


class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn1 = self.conv_spatial(attn)
        attn2 = self.conv1(attn1)
        a= u * attn2
        return a

class CIMM(nn.Module):
    def __init__(self, in_c, num_groups=6, hidden_dim=None):
        super().__init__()
        self.num_groups = num_groups

        hidden_dim = hidden_dim or in_c // 2
        expand_dim = hidden_dim * num_groups
        self.expand_conv = ConvBNReLU(in_c, expand_dim, 1)
        self.conv1 = LKA(num_groups * hidden_dim)
        self.gate_genator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(num_groups * hidden_dim, hidden_dim, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, num_groups * hidden_dim, 1),
            nn.Softmax(dim=1),
        )

        self.interact = nn.ModuleDict()
        self.interact["0"] = ConvBNReLU(hidden_dim, 3 * hidden_dim, 3, 1, 1)
        for group_id in range(1, num_groups - 1):
            self.interact[str(group_id)] = ConvBNReLU(2 * hidden_dim, 3 * hidden_dim, 3, 1, 1)
        self.interact[str(num_groups - 1)] = ConvBNReLU(2 * hidden_dim, 2 * hidden_dim, 3, 1, 1)

        self.fuse = nn.Sequential(nn.Conv2d(num_groups * hidden_dim, in_c, 3, 1, 1), nn.BatchNorm2d(in_c))
        self.final_relu = nn.ReLU(True)
        self.conv3=nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 3, dilation=3)
        self.conv3_3 = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 5, dilation=5)

        # self.conv3_2 = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 2, dilation=2)

    def forward(self, x):
        xs = self.expand_conv(x).chunk(self.num_groups, dim=1)
        # xs = self.expand_conv(x)
        # xs = self.conv1(xs).chunk(self.num_groups, dim=1)
        # xs[0] = self.conv3(xs[0])
        # xs[1] = self.conv3(xs[1])
        # xs[2] = self.conv3_2(xs[2])
        # xs[3] = self.conv3_2(xs[3])
        # xs[4] = self.conv3_3(xs[4])
        # xs[5] = self.conv3_3(xs[5])


        outs = []

        branch_out = self.interact["0"](xs[0])
        outs.append(branch_out.chunk(3, dim=1))

        for group_id in range(1, self.num_groups - 1):
            branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
            outs.append(branch_out.chunk(3, dim=1))

        group_id = self.num_groups - 1
        branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
        outs.append(branch_out.chunk(2, dim=1))

        out = torch.cat([o[0] for o in outs], dim=1)
        out = self.conv1(out)
        gate = self.gate_genator(torch.cat([o[-1] for o in outs], dim=1))
        out = self.fuse(out * gate)
        return self.final_relu(out + x)

class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            use_batchnorm=True,
    ):

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=not use_batchnorm,
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm:
            bn = nn.BatchNorm2d(out_channels)

        else:
            bn = nn.Identity()

        super(Conv2dReLU, self).__init__(conv, bn, relu)

class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class EAM(nn.Module):
    def __init__(self, channel):
        super(EAM, self).__init__()
        self.reduce1 = Conv1x1(320, 64)
        self.reduce2 = Conv1x1(128, 64)
        self.reduce3 = Conv1x1(64, 64)
        self.conv1 = conv2d(channel * 3, channel, kernel_size=3, padding=1, act=False)
        self.conv2 = conv2d(channel, channel, kernel_size=3, padding=1, act=False)
        self.conv3 = conv2d(channel, channel, kernel_size=3, padding=1, act=False)
        self.conv4 = conv2d(channel, channel, kernel_size=3, padding=1, act=False)
        self.conv5 = nn.Conv2d(channel, 1, kernel_size=(3, 3), padding=(1, 1))

        self.Hattn = axis_attn(channel, mode='h')
        self.Wattn = axis_attn(channel, mode='w')

        self.upsample = lambda img, size: F.interpolate(img, size=size, mode='bilinear', align_corners=True)
        self.conv_out = Conv2dReLU(channel, 1, 3, padding=1)


    def forward(self, f1, f2, f3):  # 22 22 44
        f1 = self.reduce1(f1)
        f2 = self.reduce2(f2)
        f3 = self.reduce3(f3)
        f1 = self.upsample(f1, f3.shape[-2:])
        f2 = self.upsample(f2, f3.shape[-2:])
        f3 = torch.cat([f1, f2, f3], dim=1)
        f3 = self.conv1(f3)

        Hf3 = self.Hattn(f3)
        Wf3 = self.Wattn(f3)

        f3 = self.conv2(Hf3 + Wf3)
        f3 = self.conv3(f3)
        f3 = self.conv4(f3)
        # out = self.conv5(f3)
        # x1 = self.agg1(f2, f1)
        # x2 = self.agg2(f3, x1)
        # f3 = self.conv1(x2)
        # Hf3 = self.Hattn(f3)
        # Wf3 = self.Wattn(f3)
        # f3 = self.conv2(Hf3 + Wf3)
        # f3 = self.conv3(f3)
        # f3 = self.conv4(f3)
        out = self.conv_out(f3)

        return out

class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class EEU(nn.Module):
    def __init__(self, channel):
        super(EEU, self).__init__()
        self.conv2d = ConvBNR(channel, channel, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, c, att):  # 22 44 x edge
        if c.size() != att.size():
            att = F.interpolate(att, c.size()[2:], mode='bilinear', align_corners=False)

        x = c * att + c
        x = self.conv2d(x)  # 22 22

        return x

class Decoder(nn.Module):
    def __init__(self, img_size=224,  in_chans=3,  embed_dims=[512, 320, 128, 64],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[0, 0, 0, 0]):
        super(Decoder, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.transferlayer = BasicConv2d(512, 512, 1, padding=0)
        # self.conv1 = Bottleneck(512, 320)
        # self.conv2 = Bottleneck(320, 128)
        # self.conv3 = Bottleneck(128, 64)
        # self.pooling = nn.AdaptiveAvgPool2d(1)
        # self.down_c1 = BasicConv2d(320, 1, 1, padding=0)
        # self.down_c2 = BasicConv2d(128, 1, 1, padding=0)
        # self.down_c3 = BasicConv2d(64, 1, 1, padding=0)
        self.hs0 = Trans(512, 320)
        self.hs1 = Trans(320, 128)
        self.hs2 = Trans(128, 64)
        self.hb0 = MBSA(512, 512)
        self.hb1 = MBSA(320, 320)
        self.hb2 = MBSA(128, 128)
        self.hb3 = MBSA(64, 64)
        self.d1 = nn.Sequential(CIMM(320, num_groups=6, hidden_dim=160))
        self.d2 = nn.Sequential(CIMM(128, num_groups=6, hidden_dim=64))
        self.d3 = nn.Sequential(CIMM(64, num_groups=6, hidden_dim=32))
        self.bdm = EAM(channel=64)
        self.efm1 = EEU(channel=320)
        self.efm2 = EEU(channel=128)
        self.efm3 = EEU(channel=64)

    def forward(self, pvt):
        x1 = pvt[0] #64*64*64 img_size // 4
        x2 = pvt[1] #32*32*128 img_size // 8
        x3 = pvt[2] #16*16*320 img_size // 16
        x4 = pvt[3] #8*8*512   img_size // 32
        #x4 = self.hb0(x4)
        x_4 = self.transferlayer(x4)  # 8 512 8 8
        #x = self.upsample(x_4)

        x = self.hs0(x_4)  # 8 320 16 16
        a1 = self.hb1(x3)   # MBSA
        b1 = self.hb2(x2)
        c1 = self.hb3(x1)
        bdm = self.bdm(a1, b1, c1)
        a2 = self.efm1(a1, bdm)
        # x = x * self.hb1(x3)
        # CIM
        x = x * a2
        x = self.d1(x)
        #x = self.upsample(x)

        x_3 = x


        x = self.hs1(x)  # 8 128 32 32
        b2 = self.efm2(b1, bdm)
        x = x * b2
        x = self.d2(x)
        #x = self.upsample(x)
        x_2 = x


        x = self.hs2(x)
        c2 = self.efm3(c1, bdm)
        x_1 = x * c2
        x_1 = self.d3(x_1)
        #x_1 = self.upsample(x)



        return x_1, x_2, x_3, x_4, bdm




class MCENet(nn.Module):
    def __init__(self, channel=32):
        super(MCENet, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrained_pth/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        self.decoder = Decoder()

        self.out1 = nn.Conv2d(64, 1, 1)
        self.out2 = nn.Conv2d(128, 1, 1)
        self.out3 = nn.Conv2d(320, 1, 1)
        self.out4 = nn.Conv2d(512, 1, 1)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.dc1 = nn.Conv2d(64, 64, 1)
        self.dc2 = nn.Conv2d(128, 64, 1)
        self.dc3 = nn.Conv2d(320, 64, 1)
        self.dc4 = nn.Conv2d(512, 64, 1)
        self.bn_dc1 = nn.BatchNorm2d(64)
        self.bn_dc2 = nn.BatchNorm2d(64)
        self.bn_dc3 = nn.BatchNorm2d(64)
        self.bn_dc4 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 4)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x):
        # backbone
        pvt = self.backbone(x)
        #pvt1 = self.decoder1(pvt)
        x1, x2, x3, x4, xbdm = self.decoder(pvt)
        B = x1.shape[0]
        y1 = self.pooling(self.bn_dc1(self.dc1(x1)))
        y2 = self.pooling(self.bn_dc2(self.dc2(x2)))
        y3 = self.pooling(self.bn_dc3(self.dc3(x3)))
        y4 = self.pooling(self.bn_dc4(self.dc4(x4)))
        y = y1 + y2 + y3 + y4
        coeff = self.sigmoid(self.fc2(self.relu(self.fc1(y.reshape(B, -1)))))

        prediction1 = self.out1(x1) * coeff[:,0].reshape(B, 1, 1, 1)
        prediction2 = self.out2(x2) * coeff[:,1].reshape(B, 1, 1, 1)
        prediction3 = self.out3(x3) * coeff[:,2].reshape(B, 1, 1, 1)
        prediction4 = self.out4(x4) * coeff[:,3].reshape(B, 1, 1, 1)

        prediction1_4 = F.interpolate(prediction1, scale_factor=4, mode='bilinear')
        bdm = F.interpolate(xbdm, scale_factor=4, mode='bilinear')
        prediction2_8 = F.interpolate(prediction2, scale_factor=8, mode='bilinear')
        prediction3_16 = F.interpolate(prediction3, scale_factor=16, mode='bilinear')
        prediction4_32 = F.interpolate(prediction4, scale_factor=32, mode='bilinear')
        return prediction1_4, prediction2_8, prediction3_16, prediction4_32, bdm


if __name__ == '__main__':
    model = MCE().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    prediction1, prediction2 = model(input_tensor)
    print(prediction1.size(), prediction2.size())
