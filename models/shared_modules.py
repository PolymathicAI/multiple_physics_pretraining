import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from einops import rearrange
from functools import partial


class ContinuousPositionBias1D(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.num_heads = n_heads
        self.cpb_mlp = nn.Sequential(nn.Linear(1, 512, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Linear(512, n_heads, bias=False))
        
    def forward(self, h, h2, bc=0):
        dtype, device = self.cpb_mlp[0].weight.dtype, self.cpb_mlp[0].weight.device
        if bc == 0: # Edges are actual endpoints
            relative_coords = torch.arange(-(h-1), h, dtype=dtype, device=device) / (h-1)
        elif bc == 1: # Periodic boundary conditions - aka opposite edges touch
            relative_coords = torch.cat([torch.arange(1, h//2+1, dtype=dtype, device=device),
                    torch.arange(-(h//2-1), h//2+1, dtype=dtype, device=device),
                    torch.arange(-(h//2-1), 0, dtype=dtype, device=device)
            ])  / (h-1)

        coords = torch.arange(h, dtype=torch.float32, device=device)
        coords = coords[None, :] - coords[:, None]
        coords = coords + (h-1)

        rel_pos_model = 16 * torch.sigmoid(self.cpb_mlp(relative_coords[:, None]).squeeze())
        biases = rel_pos_model[coords.long()]
        return biases.permute(2, 0, 1).unsqueeze(0).contiguous()


class RelativePositionBias(nn.Module):

    """
    From https://gist.github.com/huchenxucs/c65524185e8e35c4bcfae4059f896c16 

    Implementation of T5 relative position bias - can probably do better, but starting with something known.
    """
    def __init__(self, bidirectional=True, num_buckets=32, max_distance=128, n_heads=2):
        super(RelativePositionBias, self).__init__()
        self.bidirectional = bidirectional
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.n_heads = n_heads
        self.relative_attention_bias = nn.Embedding(self.num_buckets, self.n_heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=32):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
        Translate relative position to a bucket number for relative attention.
        The relative position is defined as memory_position - query_position, i.e.
        the distance in tokens from the attending position to the attended-to
        position.  If bidirectional=False, then positive relative positions are
        invalid.
        We use smaller buckets for small absolute relative_position and larger buckets
        for larger absolute relative_positions.  All relative positions >=max_distance
        map to the same bucket.  All relative positions <=-max_distance map to the
        same bucket.  This should allow for more graceful generalization to longer
        sequences than the model has been trained on.
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position, containing int32
            values in the range [0, num_buckets)
        """
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).to(torch.long) * num_buckets  # mtf.to_int32(mtf.less(n, 0)) * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))
        # now n is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def compute_bias(self, qlen, klen, bc=0):
        """ Compute binned relative position bias """
        context_position = torch.arange(qlen, dtype=torch.long,
                                        device=self.relative_attention_bias.weight.device)[:, None]
        memory_position = torch.arange(klen, dtype=torch.long,
                                       device=self.relative_attention_bias.weight.device)[None, :]
        relative_position = memory_position - context_position  # shape (qlen, klen)
        """
                   k
             0   1   2   3
        q   -1   0   1   2
            -2  -1   0   1
            -3  -2  -1   0
        """
        if bc == 1:
            thresh = klen // 2
            relative_position[relative_position < -thresh] = relative_position[relative_position < -thresh] % thresh
            relative_position[relative_position > thresh] = relative_position[relative_position > thresh] % -thresh
        rp_bucket = self._relative_position_bucket(
            relative_position,  # shape (qlen, klen)
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
        )
        rp_bucket = rp_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(rp_bucket)  # shape (qlen, klen, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, qlen, klen)
        return values

    def forward(self, qlen, klen, bc=0):
        return self.compute_bias(qlen, klen, bc)  # shape (1, num_heads, qlen, klen)
    
    

class MLP(nn.Module):
    def __init__(self, hidden_dim, exp_factor=4.):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, int(hidden_dim * exp_factor))
        self.fc2 = nn.Linear(int(hidden_dim * exp_factor), hidden_dim)
        self.act = nn.GELU()
        
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
    
class AbsolutePositionBias(nn.Module):

    """
    From https://gist.github.com/huchenxucs/c65524185e8e35c4bcfae4059f896c16 

    Implementation of T5 relative position bias - can probably do better, but starting with something known.
    """
    def __init__(self, hidden_dim, n_tokens):
        super(AbsolutePositionBias, self).__init__()
        self.bias = nn.Parameter(torch.randn(1, n_tokens, hidden_dim)*.02)

    def forward(self):
        return self.bias  # shape (1, num_heads, qlen, klen)
    

class MLP(nn.Module):
    def __init__(self, hidden_dim, exp_factor=4.):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, int(hidden_dim * exp_factor))
        self.fc2 = nn.Linear(int(hidden_dim * exp_factor), hidden_dim)
        self.act = nn.GELU()
        
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))