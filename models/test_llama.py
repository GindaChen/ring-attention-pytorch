
from ring_attention_pytorch.parallel_state import get_cache_model_parallel_group, get_cache_model_parallel_rank, get_cache_model_parallel_world_size, initialize_model_parallel

# from functools import lru_cache, partial, wraps

# cache = partial(lru_cache, maxsize = None)

# @cache()
# def get_rank():
#     return get_cache_model_parallel_rank()

# @cache()
# def get_world_size():
#     return get_cache_model_parallel_world_size()

# import ring_attention_pytorch.ring

# ring_attention_pytorch.ring.get_rank = get_rank
# ring_attention_pytorch.ring.get_world_size = get_world_size

import torch.distributed
from transformers import LlamaConfig

from llama import LlamaModel, LlamaForCausalLM, set_ring_seq_and_bucket_size
import torch
import torch.distributed as dist


from transformers import LlamaConfig

# Set inference mode to True
torch.set_grad_enabled(False)


DEBUG_MODE = True

def debug_print(message):
    if DEBUG_MODE:
        print(message)


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Test Llama model")
    # tp_size
    parser.add_argument("--tp_size", type=int, default=1, help="The tensor parallel size")
    # sp_size
    parser.add_argument("--sp_size", type=int, default=1, help="The sequence parallel size")
    # seq_len
    parser.add_argument("--seq_len", type=int, default=1024, help="The sequence length")
    # model_name
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B", help="The model name")
    # strip attentio
    parser.add_argument("--enable_strip_attention", action="store_true", help="Enable strip attention")
    
    return parser.parse_args()

args = parse_args()
print(args)

# Set default device to cuda

# Initialize the distributed backend
debug_print("Initializing the distributed backend...")
dist.init_process_group(backend='nccl')  # Assuming you're using NCCL as the backend
world_size = torch.distributed.get_world_size()

rank = torch.distributed.get_rank()
torch.cuda.set_device(rank)


# Initialize model parallelism with tp=8, pp=1, cp=1
debug_print(f"Initializing model parallelism with tp={args.tp_size}, pp=1, cp={args.sp_size}...")
initialize_model_parallel(
    tensor_model_parallel_size=args.tp_size,
    pipeline_model_parallel_size=1,
    cache_model_parallel_size=args.sp_size,
)

torch.distributed.barrier()

if rank == 0:
    print("All group has setup the cache parallel group.")

# Set the sequence length
seq_len = args.seq_len
sequence_parallel_world_size = args.sp_size

if rank == 0:
    # print world size
    print(f"World size: {torch.distributed.get_world_size()}")

set_ring_seq_and_bucket_size(
    ring_seq_len=(seq_len // sequence_parallel_world_size),
    bucket_size=(seq_len // sequence_parallel_world_size), 
)


# Initialize the configuration automatically from the pretrained model
model_name = "meta-llama/Meta-Llama-3-8B"
# model_name = "TinyLlama/TinyLlama_v1.1"
debug_print(f"Loading configuration from the pretrained model {model_name}...")


config = LlamaConfig.from_pretrained(model_name)
config.enable_strip_attention = args.enable_strip_attention

# Initialize the model using the automatically loaded config
debug_print("Initializing the Llama model using the loaded configuration...")

with torch.inference_mode():
    model = LlamaForCausalLM(config)

    # Load the pretrained weights into the model
    debug_print("Loading pretrained weights into the model...")
    # model.load_weights(model_name, load_format="auto")

    # Set the model to evaluation mode
    model.eval()
    model.to('cuda')

num_blocks = 1024
block_size = 16
num_kv_heads = config.num_key_value_heads
head_dim = config.hidden_size // config.num_attention_heads

# Example inputs to test the forward pass
# hidden_states = torch.randint(low=0, high=10, size=(1, 128, config.hidden_size))

seq = torch.randint(1, 128, (seq_len // sequence_parallel_world_size, ))
positions = torch.arange(seq_len // sequence_parallel_world_size).unsqueeze(0).long()  # Example positions
kv_caches = [None] * config.num_hidden_layers  # Placeholder for kv_caches

debug_print("Finish creating kv cache blocks.")

# Move inputs to the appropriate device (GPU)
seq = seq.to('cuda')
positions = positions.to('cuda')

# Warm up
debug_print("Warm up...")
with torch.inference_mode():
    model(seq, positions, kv_caches)

import gc
gc.collect()
debug_print("Warm up and gc finish. Start actual run...")
# Perform the forward pass
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
with torch.inference_mode():
    output = model(seq, positions, kv_caches)
end.record()

torch.cuda.synchronize()

duration = start.elapsed_time(end)

if rank == 0:
    print(f"Running {model_name = } with {seq_len = }: {duration} ms")