import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from functools import partial
from timm.layers import DropPath
# from flash_cosine_sim_attention import flash_cosine_sim_attention
import math
try:
    from .shared_modules import RelativePositionBias, ContinuousPositionBias1D, MLP
except:
    from shared_modules import RelativePositionBias, ContinuousPositionBias1D, MLP

def build_time_block(params):
    """
    Builds a time block from the parameter file. 
    """
    if params.time_type == 'attention':
        return partial(AttentionBlock, params.embed_dim, params.num_heads, bias_type=params.bias_type)
    else:
        raise NotImplementedError
    

class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=12, drop_path=0, layer_scale_init_value=1e-6, bias_type='rel'):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = nn.InstanceNorm2d(hidden_dim, affine=True)
        self.norm2 = nn.InstanceNorm2d(hidden_dim, affine=True)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((hidden_dim)), 
                            requires_grad=True) if layer_scale_init_value > 0 else None
        self.input_head = nn.Conv2d(hidden_dim, 3*hidden_dim, 1)
        self.output_head = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.qnorm = nn.LayerNorm(hidden_dim//num_heads)
        self.knorm = nn.LayerNorm(hidden_dim//num_heads)
        if bias_type == 'none':
            self.rel_pos_bias = lambda x, y: None
        elif bias_type == 'continuous':
            self.rel_pos_bias = ContinuousPositionBias1D(n_heads=num_heads)
        else:
            self.rel_pos_bias = RelativePositionBias(n_heads=num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # input is t x b x c x h x w 
        T, B, C, H, W = x.shape
        input = x.clone()
        # Rearrange and prenorm
        x = rearrange(x, 't b c h w -> (t b) c h w')
        x = self.norm1(x)
        x = self.input_head(x) # Q, K, V projections
        # Rearrange for attention
        x = rearrange(x, '(t b) (he c) h w ->  (b h w) he t c', t=T, he=self.num_heads)
        q, k, v = x.tensor_split(3, dim=-1)
        q, k = self.qnorm(q), self.knorm(k)
        rel_pos_bias = self.rel_pos_bias(T, T)
        if rel_pos_bias is not None:
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=rel_pos_bias) 
        else:
            x = F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous())
        # Rearrange after attention
        x = rearrange(x, '(b h w) he t c -> (t b) (he c) h w', h=H, w=W)
        x = self.norm2(x) 
        x = self.output_head(x)
        x = rearrange(x, '(t b) c h w -> t b c h w', t=T)
        output = self.drop_path(x*self.gamma[None, None, :, None, None]) + input
        return output
