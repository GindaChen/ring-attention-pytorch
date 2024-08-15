import os
import click
from math import ceil

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ring_attention_pytorch.ring_attention import RingAttention
from ring_attention_pytorch.distributed import all_gather_variable_dim

def setup(
    rank,
    world_size,
    use_cuda
):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    backend = "gloo" if not use_cuda else "nccl"
    dist.init_process_group(backend, rank = rank, world_size = world_size)

    if use_cuda:
        torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def start(
    rank,
    world_size,
    batch_size,
    batch_size_var_len,
    seq_len,
    num_buckets,
    num_sharded_batches,
    causal,
    striped_ring_attn,
    dim,
    heads,
    num_grouped_query_heads,
    dim_head,
    use_cuda,
    compare_regular_attn
):
    setup(rank, world_size, use_cuda)

    ring_seq_size = ceil(seq_len / world_size) * num_sharded_batches
    bucket_size = ring_seq_size // num_buckets

    print(f'rank: {rank}, ring_seq_size: {ring_seq_size}, bucket_size: {bucket_size}')

    ring_attention = RingAttention(
        dim = dim,
        causal = causal,
        dim_head = dim_head,
        heads = heads,
        num_grouped_query_heads = num_grouped_query_heads,
        ring_attn = True,
        striped_ring_attn = striped_ring_attn,
        ring_seq_size = ring_seq_size,
        bucket_size = bucket_size,
        use_cuda_kernel = use_cuda,
        auto_shard_seq = True
    )

    if batch_size_var_len:
        batch_size = batch_size + rank

    seq = torch.randn(batch_size, seq_len, dim)

    # move to cuda if needed
    print(f'rank: {rank}, use_cuda: {use_cuda}')

    if use_cuda:
        seq = seq.cuda(rank)
        ring_attention.cuda(rank)

    ring_input = seq.clone().requires_grad_()
    print(f'Start forward pass for rank {rank}')
    # ddp_ring_attention = DDP(ring_attention)
    ddp_ring_attention = (ring_attention)
    print(f'Finish forward pass for rank {rank}')
    ring_out = ddp_ring_attention(ring_input)
    if rank == 0:
        ring_attention = ring_attention.cpu()
    
    cleanup()


def test(
    world_size: int,
    batch_size: int,
    num_sharded_batches: int,
    batch_size_var_len: bool,
    use_cuda: bool,
    causal: bool,
    striped_ring_attn: bool,
    num_buckets: int,
    seq_len: int,
    model_dim: int,
    heads: int,
    num_grouped_query_heads: int,
    dim_head: int,
    compare_regular_attn: bool
):
    assert not use_cuda or world_size <= torch.cuda.device_count(), f'world size {world_size} must be less than the number of cuda devices {torch.cuda.device_count()}'

    mp.spawn(
        start,
        args = (
            world_size,
            batch_size,
            batch_size_var_len,
            seq_len,
            num_buckets,
            num_sharded_batches,
            causal,
            striped_ring_attn,
            model_dim,
            heads,
            num_grouped_query_heads,
            dim_head,
            use_cuda,
            compare_regular_attn
        ),
        nprocs = world_size,
        join = True
    )
    print(f'Run completed.')

if __name__ == '__main__':    
    test(
        # world_size = 8,
        world_size = 2,
        batch_size = 1,
        num_sharded_batches = 1,
        batch_size_var_len = False,
        use_cuda = True,
        causal = True,
        # striped_ring_attn = False,
        striped_ring_attn = True,
        num_buckets = 2,
        seq_len = 1024 * 128,
        num_grouped_query_heads = 1,
        model_dim = 1024,
        heads = 16,
        dim_head = 128,
        compare_regular_attn = False,
    )