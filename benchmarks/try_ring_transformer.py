import os
import click
from math import ceil

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


from ring_attention_pytorch.ring_attention import RingTransformer
from transformers import AutoConfig

def setup(rank, world_size, use_cuda):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    backend = "gloo" if not use_cuda else "nccl"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    if use_cuda:
        torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def start(*args):
    
    _start(*args)

def _start(
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
    compare_regular_attn,
    num_tokens,
    depth,
    model
):
    setup(rank, world_size, use_cuda)
    if rank == 0:
        print(
            f"--world-size {world_size} "
            f"--batch-size {batch_size} "
            f"--num-sharded-batches {num_sharded_batches} "
            f"--batch-size-var-len {batch_size_var_len} "
            f"--use-cuda {use_cuda} "
            f"--causal {causal} "
            f"--striped-ring-attn {striped_ring_attn} "
            f"--num-buckets {num_buckets} "
            f"--seq-len {seq_len} "
            f"--dim {dim} "
            f"--heads {heads} "
            f"--num-grouped-query-heads {num_grouped_query_heads} "
            f"--dim-head {dim_head} "
            f"--num-tokens {num_tokens} "
            f"--depth {depth} "
            f"--model {model} "
        )
        

    if model:
        config = AutoConfig.from_pretrained(model)
        num_tokens = config.vocab_size
        hidden_size = config.hidden_size
        dim = hidden_size
        heads = config.num_attention_heads
        depth = config.num_hidden_layers
        dim_head = hidden_size // heads
        # num_tokens = config.vocab_size // world_size
        # hidden_size = config.hidden_size // world_size
        # # hidden_size = config.hidden_size
        # dim = hidden_size
        # heads = config.num_attention_heads // world_size
        # depth = config.num_hidden_layers
        # dim_head = hidden_size // heads

        if rank == 0:
            print(f'Loaded model config: {config}')
            print(f'num_tokens: {num_tokens}, dim: {dim}, heads: {heads}, depth: {depth}, dim_head: {dim_head}')

    if use_cuda:
        torch.cuda.set_device(rank)

    ring_seq_size = ceil(seq_len / world_size) * num_sharded_batches
    bucket_size = ring_seq_size // num_buckets

    print(f"Preparing the model for rank {rank}.")
    with torch.inference_mode():
        ring_attention_net = RingTransformer(
            num_tokens=num_tokens,
            dim=dim,
            causal=causal,
            depth=depth,
            heads=heads,
            num_grouped_query_heads=num_grouped_query_heads,
            dim_head=dim_head,
            ring_attn=True,
            striped_ring_attn=striped_ring_attn,
            ring_seq_size=ring_seq_size,
            bucket_size=bucket_size,
            # TODO: What is auto_shard_seq?
            auto_shard_seq=False,
        )
        # ring_attention_net = FSDP(ring_attention_net, device_id=rank)
        ring_attention_net.eval()
        ring_attention_net.cuda(rank)

    print("Finished preparing the model.")
    torch.distributed.barrier()

    if batch_size_var_len:
        batch_size = batch_size + rank

    with torch.inference_mode():
        seq = torch.randint(0, num_tokens, (batch_size, seq_len))

        # move to cuda if needed
        if use_cuda:
            seq = seq.cuda(rank)
        print(f'Prepared data for rank {rank}')
        torch.distributed.barrier()

        # ring
        # Wram up
        # for _ in range(2):
        #     _ = ring_attention_net(seq)
        
        if use_cuda:
            st = torch.cuda.Event(enable_timing=True)
            ed = torch.cuda.Event(enable_timing=True)

        print(f'Start forward pass for rank {rank}')
        if use_cuda:
            st.record()
        ring_out = ring_attention_net(seq)
        if use_cuda:
            ed.record()
        if use_cuda:
            torch.cuda.synchronize()
        print(f'Finish forward pass for rank {rank}.')
    
        duration = 0
        if use_cuda:
            duration = st.elapsed_time(ed)
            print(f'Finish forward pass for rank {rank}: {duration} ms')
        else:
            print(f'Finish forward pass for rank {rank}')

    # validate output is the same for sequence split across machines vs without
    if rank == 0:
        # ring_attention_net = ring_attention_net.cpu()
        # ring_out = ring_out.cpu()
        print('✅ Finished testing.')
        print(f'🎉 Finish testing {model = }, {world_size = }, {seq_len = }, {duration = }ms')

    if rank == 0:
        import wandb
        wandb.init(project="ring-transformer", name=f"w{world_size}-b{batch_size}-s{seq_len}-m{model}-nsb-{num_sharded_batches}-nb-{num_buckets}")
        wandb.log({
            "duration": duration,
            "world_size": world_size,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "model": model,
            "num_sharded_batches": num_sharded_batches,
            "num_buckets": num_buckets
        })

    cleanup()

@click.command()
@click.option('--world-size', default=8, help='number of machines / processes')
@click.option('--batch-size', default=1, help='test batch size')
@click.option('--num-sharded-batches', default=1, help='number of sharded batches')
@click.option('--batch-size-var-len', is_flag=True, help='test variable lengthed batch sizes')
@click.option('--use-cuda', is_flag=True, help='whether to test with CUDA and NCCL')
@click.option('--causal', is_flag=True, help='test autoregressive')
@click.option('--striped-ring-attn', is_flag=True, help='test striped ring attention from MIT follow up paper')
@click.option('--num-buckets', default=2, help='number of buckets per machine')
@click.option('--seq-len', default=32, help='sequence length to test')
@click.option('--model-dim', default=8, help='model dimensions for testing')
@click.option('--heads', default=8, help='number of query attention heads')
@click.option('--num-grouped-query-heads', default=2, help='number of query attention head groups')
@click.option('--dim-head', default=16, help='attention head dimension')
@click.option('--compare-regular-attn', is_flag=True, help='compare ring to regular attention')
@click.option('--num-tokens', default=256, help='number of tokens in the vocabulary')
@click.option('--depth', default=2, help='number of layers in the transformer')
@click.option('--model', default='', help='Pretrained transformer model name (e.g., "bert-base-uncased"). If provided, this overrides other options.')
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
    compare_regular_attn: bool,
    num_tokens: int,
    depth: int,
    model: str
):
    assert not use_cuda or world_size <= torch.cuda.device_count(), f'world size {world_size} must be less than the number of cuda devices {torch.cuda.device_count()}'

    mp.spawn(
        start,
        args=(
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
            compare_regular_attn,
            num_tokens,
            depth,
            model
        ),
        nprocs=world_size,
        join=True
    )

if __name__ == '__main__':
    test()