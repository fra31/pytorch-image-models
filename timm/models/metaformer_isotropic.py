# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
#from timm.models.registry import register_model
#from .convnext import Block, LayerNorm





class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6,
        mixer_type=None, norm_type=None, **kwargs):
        super().__init__()
        #self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size,
        #    padding=padding, groups=dim) # depthwise conv
        kwargs['norm_type'] = norm_type
        self.mixer = MIXERS[mixer_type](dim=dim, **kwargs)
        self.norm = NORM_LAYERS[norm_type](dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.mixer(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

        
class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self, pool_size=3, **kwargs):
        super().__init__()
        padding = get_padding(pool_size, stride=1) # get automatically padding value
        print(f'using pool size={pool_size}, padding={padding}')
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=padding, count_include_pad=False)
        self.normalize_input = kwargs.get('normalize', False)
        if self.normalize_input:
            print('normalizing before pooling')
            self.norm = NORM_LAYERS[kwargs['norm_type']](kwargs['dim'],
                data_format='channels_first')

    def forward(self, x):
        if self.normalize_input:
            x = self.norm(x)
        return self.pool(x) - x
        

def dw_conv(dim_in, dim_out, kernel_size, padding):
    return nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size,
        padding=padding, groups=dim_in)


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
            

class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)

        
class GroupNormLocal(nn.Module):
    def __init__(self, num_groups, normalized_shape, eps=1e-5, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.num_groups = num_groups
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_first" or len(x.shape) == 2: # it should be equivalent to GroupNorm
            return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_last":
            '''u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            '''
            # reimplementation for channel last format
            '''u = x.view(x.shape[0], -1).mean(-1, keepdim=True)
            s = (x.view(x.shape[0], -1) - u).pow(2).mean(-1)
            x = (x - u.view(-1, *[1] * (len(x.shape) - 1)) / torch.sqrt(
                s.view(-1, *[1] * (len(x.shape) - 1)) + self.eps)'''
            # suboptimal way to handle this
            x = F.group_norm(x.permute(0, 3, 1, 2), self.num_groups, self.weight, self.bias, self.eps)
            return x.permute(0, 2, 3, 1)


def get_padding(kernel_size, stride=1, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding
    

MIXERS = {'avg_pooling': Pooling,
    'avg_pooling_norm': lambda dim, **kwargs: Pooling(dim=dim,
        norm_type=kwargs['norm_type'], normalize=True),
    'dw_conv': lambda dim, **kwargs: dw_conv(dim, dim,
        kwargs['kernel_size'], kwargs['padding']),
    }
    

NORM_LAYERS = {'layer_norm': lambda dim, data_format='channels_last': LayerNorm(dim,
        eps=1e-6, data_format=data_format),
    'group_norm': lambda dim: GroupNorm(dim),
    'group_norm_loc': lambda dim: GroupNormLocal(num_groups=1, normalized_shape=dim),
    }


class MetaFormerIsotropic(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
        https://arxiv.org/pdf/2201.03545.pdf
        Isotropic ConvNeXts (Section 3.3 in paper)

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depth (tuple(int)): Number of blocks. Default: 18.
        dims (int): Feature dimension. Default: 384
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
        patch_size (int): patch size for stem
        kernel_size (int): kernel size of dw convolutions
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depth=18, dim=384, drop_path_rate=0., 
                 layer_scale_init_value=0, head_init_scale=1.,
                 patch_size=16, kernel_size=7, mixer_type=None,
                 norm_type='layer_norm', depths=None, mixers=None,
                 ):
        super().__init__()
        
        print(f'using mixer={mixer_type}, normalization={norm_type}')

        self.stem = nn.Conv2d(in_chans, dim, kernel_size=patch_size,
            stride=patch_size)
        
        padding = get_padding(kernel_size)
        #print(f'using kernel size={kernel_size}, padding={padding}')
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        if depths is None:
            # for now all blocks have same components
            self.blocks = nn.Sequential(*[Block(dim=dim, drop_path=dp_rates[i], 
                                        layer_scale_init_value=layer_scale_init_value,
                                        kernel_size=kernel_size, padding=padding,
                                        mixer_type=mixer_type, norm_type=norm_type)
                                        for i in range(depth)])
        else:
            # if different mixers are used
            assert len(mixers) == len(depths)
            assert torch.tensor(depths).sum() == depth
            blocks = []
            for mixer, depth in zip(mixers, depths):
                blocks.extend([Block(dim=dim, drop_path=0., 
                                        layer_scale_init_value=layer_scale_init_value,
                                        kernel_size=kernel_size, padding=padding,
                                        mixer_type=mixer, norm_type=norm_type)
                                        for i in range(depth)])
            self.blocks = nn.Sequential(*blocks)
                                        

        self.norm = NORM_LAYERS[norm_type](dim) # final norm layer
        self.head = nn.Linear(dim, num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
        

    def forward_features(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x





