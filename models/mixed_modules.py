import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from functools import partial
from torch.utils.checkpoint import checkpoint
import math
try:
    from spatial_modules import build_space_block
    from time_modules import build_time_block, AttentionBlock
except:
    from .spatial_modules import build_space_block
    from .time_modules import build_time_block, AttentionBlock

def build_spacetime_block(params):
    """
    Builds a spacetime block from the parameter file. 
    """
    if params.block_type == 'axial':
        space_block = build_space_block(params)
        time_block = build_time_block(params)
        return partial(SpaceTimeBlock, params.embed_dim, params.num_heads, space_override=space_block,
                        time_override=time_block, gradient_checkpointing=params.gradient_checkpointing)
    else:
        raise NotImplementedError
    

class SpaceTimeBlock(nn.Module):
    """
    Alternates spatial and temporal processing. Current code base uses
    1D attention over each axis. Spatial axes share weights.

    Note: MLP is in spatial block. 
    """
    def __init__(self, hidden_dim=768, num_heads=12, drop_path=0., space_override=None, time_override=None,
                    gradient_checkpointing=False):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        if space_override is not None:
            self.spatial = space_override(drop_path=drop_path)

        if time_override is not None:
            self.temporal = time_override(drop_path=drop_path)
        else:
            self.temporal = AttentionBlock(hidden_dim, num_heads, drop_path=drop_path)

    def forward(self, x, bcs):
        # input is t x b x c x h x w 
        T, B, C, H, W = x.shape

        # Time attention
        if self.gradient_checkpointing:
            # kwargs seem to need to be passed explicitly
            wrapped_temporal = partial(self.temporal)
            x = checkpoint(wrapped_temporal, x, use_reentrant=False)
        else:
            x = self.temporal(x) # Residual in block
        # Temporal handles the rearrange so still is t x b x c x h x w 

        # Now do spatial attention
        x = rearrange(x, 't b c h w -> (t b) c h w')
        if self.gradient_checkpointing:
            # kwargs seem to need to be passed explicitly 
            wrapped_spatial = partial(self.spatial)
            x = checkpoint(wrapped_spatial, x, bcs, use_reentrant=False)
        else:
            x = self.spatial(x, bcs) # Convnext has the residual in the block
        x = rearrange(x, '(t b) c h w -> t b c h w', t=T) 

        return x
