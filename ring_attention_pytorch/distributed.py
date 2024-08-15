import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
from torch.autograd import Function

import torch.distributed as dist

from ring_attention_pytorch.parallel_state import (
    get_cache_model_parallel_group, 
    get_cache_model_parallel_rank, 
    get_cache_model_parallel_world_size,

)

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(num, den):
    return (num % den) == 0

def pad_dim_to(t, length, dim = 0):
    pad_length = length - t.shape[dim]
    zero_pairs = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    return F.pad(t, (*((0, 0) * zero_pairs), 0, pad_length))

def all_gather_same_dim(t):
    t = t.contiguous()
    world_size = get_cache_model_parallel_world_size()
    gathered_tensors = [torch.empty_like(t, device = t.device, dtype = t.dtype) for i in range(world_size)]
    group = get_cache_model_parallel_group()
    dist.all_gather(gathered_tensors, t, group=group)
    return gathered_tensors

def gather_sizes(t, *, dim):
    size = torch.tensor(t.shape[dim], device = t.device, dtype = torch.long)
    sizes = all_gather_same_dim(size)
    return torch.stack(sizes)

def has_only_one_value(t):
    return (t == t[0]).all()

def all_gather_variable_dim(t, dim = 0, sizes = None):
    device, rank, world_size = t.device, get_cache_model_parallel_rank(), get_cache_model_parallel_world_size()

    if not exists(sizes):
        sizes = gather_sizes(t, dim = dim)

    if has_only_one_value(sizes):
        gathered_tensors = all_gather_same_dim(t)
        gathered_tensors = torch.cat(gathered_tensors, dim = dim)
        return gathered_tensors, sizes

    max_size = sizes.amax().item()

    padded_t = pad_dim_to(t, max_size, dim = dim)
    gathered_tensors = all_gather_same_dim(padded_t)

    gathered_tensors = torch.cat(gathered_tensors, dim = dim)
    seq = torch.arange(max_size, device = device)

    mask = seq[None, :] < sizes[:, None]
    mask = rearrange(mask, 'i j -> (i j)')
    seq = torch.arange(mask.shape[-1], device = device)
    indices = seq[mask]

    gathered_tensors = gathered_tensors.index_select(dim, indices)

    return gathered_tensors, sizes

class AllGatherFunction(Function):
    @staticmethod
    def forward(ctx, x, dim, sizes):
        is_bool = x.dtype == torch.bool

        if is_bool:
            x = x.int()

        x, batch_sizes = all_gather_variable_dim(x, dim = dim, sizes = sizes)
        ctx.batch_sizes = batch_sizes.tolist()
        ctx.dim = dim

        if is_bool:
            x = x.bool()

        return x, batch_sizes

    @staticmethod
    def backward(ctx, grads, _):
        batch_sizes, rank = ctx.batch_sizes, get_cache_model_parallel_rank()
        grads_by_rank = grads.split(batch_sizes, dim = ctx.dim)
        return grads_by_rank[rank], None, None

class AllGather(Module):
    def __init__(self, *, dim = 0):
        super().__init__()
        self.dim = dim

    def forward(self, x, sizes = None):
        return AllGatherFunction.apply(x, self.dim, sizes)

def split_by_rank(x):
    rank = get_cache_model_parallel_rank()
    out = x[rank]

    if isinstance(x, tuple):
        sizes = tuple(map(lambda t: t.shape[0], x))
    else:
        sizes = (x.shape[1],) * x.shape[0]

    sizes = torch.tensor(sizes, device = out.device, dtype = torch.long)
    return out, sizes
