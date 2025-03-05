import sys

sys.path.append('../')

import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import copy
from einops import rearrange
import numpy as np
from timm.models.layers import DropPath, to_2tuple




def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class ResBlock(nn.Module):
    def __init__(
                self, conv, n_feats, kernel_size,
                bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class S_ResBlock(nn.Module):
    def __init__(
                self, conv, n_feats, kernel_size,
                bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(S_ResBlock, self).__init__()

        assert len(conv) == 2

        m = []

        for i in range(2):
            m.append(conv[i](n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class SMCSWT(nn.Module):

    def __init__(self, window_size1, depth1, window_size2, depth2, window_size3, depth3, conv=default_conv):
        super(SMCSWT, self).__init__()

        self.scale_idx = 0

        n_colors = 1

        n_feats = 64
        kernel_size = 3
        act = nn.ReLU(True)

        self.head1 = conv(n_colors, n_feats, kernel_size)

        self.head1_1 = ResBlock(conv, n_feats, kernel_size, act=act)


        self.head1_3 = SMSBlock(dim=64,
                                  window_size=window_size1[0],
                                  depth=depth1[0],
                                  num_head=8,
                                  weight_factor=0.1, down_rank=16, memory_blocks=128,
                                  mlp_ratio=2,
                                  qkv_bias=True, qk_scale=None,
                                  split_size=1,
                                  drop_path=0
                                  )
                             

        self.body1_1 = nn.Sequential(
            ResBlock(conv, n_feats, kernel_size, act=act),
            conv(n_feats, n_feats, kernel_size),   # conv3
            act
        )

        self.body1_2 = nn.Sequential(
            *[SMSBlock(dim=64,
                     window_size=window_size1[1],
                     depth=depth1[1],
                     num_head=8,
                     weight_factor=0.1, down_rank=16, memory_blocks=128,
                     mlp_ratio=2,
                     qkv_bias=True, qk_scale=None,
                     split_size=1,
                     drop_path=0
                     )
                for _ in range(1)],
            ResBlock(conv, n_feats, kernel_size, act=act)
        )

        self.body2_1 = nn.Sequential(
            ResBlock(conv, n_feats, kernel_size, act=act),

            *[SMSBlock(dim=64,
                     window_size=window_size2[0],
                     depth=depth2[0],
                     num_head=8,
                     weight_factor=0.1, down_rank=16, memory_blocks=128,
                     mlp_ratio=2,
                     qkv_bias=True, qk_scale=None,
                     split_size=1,
                     drop_path=0
                     )
                for _ in range(1)]
        )

        self.fusion2_1 = nn.Sequential(
            # DConv
            nn.Conv2d(n_feats*2, n_feats*2, kernel_size=5, padding=2, groups=n_feats*2),
            conv(n_feats * 2, n_feats, 1)  # conv1
        )

        self.body2_2 = nn.Sequential(
            ResBlock(conv, n_feats, kernel_size, act=act),

            *[SMSBlock(dim=64,
                     window_size=window_size2[1],
                     depth=depth2[1],
                     num_head=8,
                     weight_factor=0.1, down_rank=16, memory_blocks=128,
                     mlp_ratio=2,
                     qkv_bias=True, qk_scale=None,
                     split_size=1,
                     drop_path=0
                     )
                for _ in range(1)]
        )

        self.fusion2_2 = nn.Sequential(
            nn.Conv2d(n_feats*2, n_feats*2, kernel_size=5, padding=2, groups=n_feats*2),
            conv(n_feats * 2, n_feats, 1),  # conv1
        )


        self.body2_3 = SMSBlock(dim=64,
                                  window_size=window_size2[2],
                                  depth=depth2[2],
                                  num_head=8,
                                  weight_factor=0.1, down_rank=16, memory_blocks=128,
                                  mlp_ratio=2,
                                  qkv_bias=True, qk_scale=None,
                                  split_size=1,
                                  drop_path=0
                                  )
                             

        self.body3_1 = nn.Sequential(
            conv(n_feats, n_feats, kernel_size),   # conv1
            act,

            *[SMSBlock(dim=64,
                     window_size=window_size3[0],
                     depth=depth3[0],
                     num_head=8,
                     weight_factor=0.1, down_rank=16, memory_blocks=128,
                     mlp_ratio=2,
                     qkv_bias=True, qk_scale=None,
                     split_size=1,
                     drop_path=0
                     )
                for _ in range(1)]
        )

        self.fusion3_1 = nn.Sequential(
            nn.Conv2d(n_feats*2, n_feats*2, kernel_size=5, padding=2, groups=n_feats*2),
            conv(n_feats * 2, n_feats, 1),  # conv1
        )

        self.body3_2 = nn.Sequential(
            conv(n_feats, n_feats, kernel_size),   # conv1
            act,

            *[SMSBlock(dim=64,
                     window_size=window_size3[1],
                     depth=depth3[1],
                     num_head=8,
                     weight_factor=0.1, down_rank=16, memory_blocks=128,
                     mlp_ratio=2,
                     qkv_bias=True, qk_scale=None,
                     split_size=1,
                     drop_path=0
                     )
                for _ in range(1)]
        )

        self.fusion3_2 = nn.Sequential(
            nn.Conv2d(n_feats*2, n_feats*2, kernel_size=5, padding=2, groups=n_feats*2),
            conv(n_feats * 2, n_feats, 1),  # conv1
        )


        self.body3_3 = SMSBlock(dim=64,
                                  window_size=window_size3[2],
                                  depth=depth3[2],
                                  num_head=32,
                                  weight_factor=0.1, down_rank=16, memory_blocks=128,
                                  mlp_ratio=2,
                                  qkv_bias=True, qk_scale=None,
                                  split_size=1,
                                  drop_path=0
                                  )
                          

        self.fusion3_3 = nn.Sequential(
            nn.Conv2d(n_feats*3, n_feats*3, kernel_size=5, padding=2, groups=n_feats*3),
            conv(n_feats * 3, n_feats, 1),  # conv1
        )


        self.body3_4 = SMSBlock(dim=64,
                                  window_size=window_size3[3],
                                  depth=depth3[3],
                                  num_head=8,
                                  weight_factor=0.1, down_rank=16, memory_blocks=128,
                                  mlp_ratio=2,
                                  qkv_bias=True, qk_scale=None,
                                  split_size=1,
                                  drop_path=0
                                  )
                            
        
        self.tail = conv(n_feats, n_colors, kernel_size)

    def forward(self, x):
        y = x

        x = self.head1(x)
        x = self.head1_1(x)
        x = self.head1_3(x)


        group1 = self.body1_1(x)
        group2 = self.body2_1(x)
        group3 = self.body3_1(x)   ##


        group2 = self.body2_2(self.fusion2_1(torch.cat((group1, group2), 1)))
        group3 = self.body3_2(self.fusion3_1(torch.cat((group2, group3), 1)))  ##
        group1 = self.body1_2(group1)

        group2 = self.body2_3(self.fusion2_2(torch.cat((group1, group2), 1)))
        group3 = self.body3_3(self.fusion3_2(torch.cat((group2, group3), 1)))  ##

        group3 = self.body3_4(self.fusion3_3(torch.cat((group1, group2, group3), 1)))   ##

        x = group3

        out = self.tail(x)

        return out + y




class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=True):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


class LePEAttention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0., qk_scale=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        if idx == 0:
            H_sp, W_sp = self.resolution, self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution, self.split_size
        else:
            print("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_lepe(self, x, func):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)
        lepe = func(x)
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()
        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp * self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, qkv, mask=None):
        """
        x: B L C
        """
        q, k, v = qkv[0], qkv[1], qkv[2]
        H = W = self.resolution
        B, L, C = q.shape
        q = self.im2cswin(q)
        k = self.im2cswin(k)
        v, lepe = self.get_lepe(v, self.get_v)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)
        x = (attn @ v) + lepe
        x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)  # B head N N @ B head N C
        x = windows2img(x, self.H_sp, self.W_sp, H, W).view(B, -1, C)  # B H' W' C
        return x

    def flops(self, shape):
        flops = 0
        H, W = shape
        flops += ((H // self.H_sp) * (W // self.W_sp)) * self.num_heads * (self.H_sp * self.W_sp) * (
                    self.dim // self.num_heads) * (self.H_sp * self.W_sp)
        flops += ((H // self.H_sp) * (W // self.W_sp)) * self.num_heads * (self.H_sp * self.W_sp) * (
                    self.dim // self.num_heads) * (self.H_sp * self.W_sp)
        return flops


class QKV_lowrank(nn.Module):
    def __init__(self, dim, qkv_bias, low_rank, memory_blocks, weight_factor, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.low_rank = low_rank
        self.qk = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.weight_factor = weight_factor

    def forward(self, x):
        qk = self.qk(x)
        v = self.v(x)
        qkv = torch.concat([qk, v], dim=-1)
        return qkv


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self,
                 dim,
                 window_size,
                 num_heads,
                 qkv_bias=0,
                 qk_scale=None,
                 memory_blocks=128,
                 down_rank=16,
                 weight_factor=0.1,
                 attn_drop=0.,
                 proj_drop=0.,
                 split_size=1):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.weight_factor = weight_factor

        self.attns = nn.ModuleList([
            LePEAttention(
                dim // 2, resolution=self.window_size[0], idx=i,
                split_size=split_size, num_heads=num_heads // 2, dim_out=dim // 2,
                qk_scale=qk_scale, attn_drop=attn_drop)
            for i in range(2)])

        self.qkv_lowrank = QKV_lowrank(dim, qkv_bias=qkv_bias, low_rank=down_rank, memory_blocks=memory_blocks,
                                       weight_factor=weight_factor)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # print(f'attention input shape: {x.shape}')
        #  H/window_size*W/window_size*B x window_size*window_size x C
        B, N, C = x.shape
        qkv = self.qkv_lowrank(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3)

        x1 = self.attns[0](qkv[:, :, :, :C // 2], mask)
        x2 = self.attns[1](qkv[:, :, :, C // 2:], mask)
        # x1 = 0 * x1

        attened_x = torch.cat([x1, x2], dim=2)
        attened_x = rearrange(attened_x, 'b n (g d) -> b n ( d g)', g=4)

        x = self.proj(attened_x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, shape):
        # calculate flops for 1 window with token length of N
        flops = 0
        H, W = shape
        # qkv = self.qkv(x)
        flops += 2 * self.attns[0].flops([H, W])
        flops += self.c_attns.flops([H, W])
        return flops


class SSMTDA(nn.Module):
    r"""  Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, split_size=1, drop_path=0.0,
                 weight_factor=0.1, memory_blocks=128, down_rank=16,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., act_layer=nn.GELU):
        super(SSMTDA, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.weight_factor = weight_factor

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = FeedForward(dim=dim, bias=True)

        self.attns = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, memory_blocks=memory_blocks,
            down_rank=down_rank, weight_factor=weight_factor, split_size=split_size,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.num_heads = num_heads

    def forward(self, x):
        B, C, H, W = x.shape

        x = x.flatten(2).transpose(1, 2)
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        # B x H x W x C

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attns(x_windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        norm_x = self.norm2(x)
        norm_x = torch.transpose(norm_x, 1, 2).reshape(B, C, H, W)
        x = x + self.drop_path(self.mlp(norm_x).reshape(B, C, H * W).transpose(1, 2))
        x = x.transpose(1, 2).view(B, C, H, W)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self, shape):
        flops = 0
        H, W = shape
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attns.flops([self.window_size, self.window_size])
        return flops


class SMSBlock(nn.Module):
    def __init__(self,
                 dim=64,
                 window_size=8,
                 depth=6,
                 num_head=6,
                 mlp_ratio=2,
                 qkv_bias=True, qk_scale=None,
                 weight_factor=0.1, memory_blocks=128, down_rank=16,
                 drop_path=0.0,
                 split_size=1,
                 ):
        super(SMSBlock, self).__init__()
        self.smsblock = nn.Sequential(*[
            SSMTDA(dim=dim, input_resolution=window_size, num_heads=num_head, memory_blocks=memory_blocks,
                   window_size=window_size, shift_size=0 if i % 2 == 0 else window_size // 2,
                   weight_factor=weight_factor, down_rank=down_rank,
                   split_size=split_size,
                   mlp_ratio=mlp_ratio,
                   drop_path=drop_path,
                   qkv_bias=qkv_bias, qk_scale=qk_scale, )
            for i in range(depth)])
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x):
        out = self.smsblock(x)
        out = self.conv(out) + x
        return out

    def flops(self, shape):
        flops = 0
        for blk in self.smsblock:
            flops += blk.flops(shape)
        return flops

