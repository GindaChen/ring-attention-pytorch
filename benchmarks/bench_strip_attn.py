import torch
from ring_attention_pytorch import RingAttention
import argparse

parser =  argparse.ArgumentParser(description='Ring Attention')
parser.add_argument('--use_strip_attn', action='store_true', help='use strip attention')
args = parser.parse_args()


ring_attn_kwargs = dict(
    ring_attn = True,
    ring_seq_size = 512,
)
strip_attn_kwargs = dict(
    striped_ring_attn = True,
)

attn_kwargs = ring_attn_kwargs if not args.use_strip_attn else strip_attn_kwargs

attn = RingAttention(
    dim = 512,
    dim_head = 64,
    heads = 8,
    causal = True,
    auto_shard_seq = True,
    use_cuda_kernel = True,
    **attn_kwargs
)

tokens = torch.randn(1, 1024, 512)
attended = attn(tokens)

assert attended.shape == tokens.shape

if args.use_strip_attn:
    print('✅ Run strip attention success.')
else:
    print('✅ Run ring attention success.')