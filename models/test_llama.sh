set -ex

MASTER_ADDR=localhost MASTER_PORT=51234 torchrun --nproc_per_node=8 test_llama.py --tp_size 1 --sp_size 8 --seq_len $((2 ** 10 * 64)) --model_name "meta-llama/Meta-Llama-3-8B"


MASTER_ADDR=localhost MASTER_PORT=51234 torchrun --nproc_per_node=8 test_llama.py --tp_size 1 --sp_size 8 --seq_len $((2 ** 10 * 512)) --model_name "meta-llama/Meta-Llama-3-8B" --enable_strip_attention


MASTER_ADDR=localhost MASTER_PORT=51234 torchrun --nproc_per_node=8 test_llama.py --tp_size 1 --sp_size 8 --seq_len $((2 ** 20)) --model_name "meta-llama/Meta-Llama-3-8B"

MASTER_ADDR=localhost MASTER_PORT=51234 torchrun --nproc_per_node=8 test_llama.py --tp_size 2 --sp_size 4 --seq_len 1024 --model_name "meta-llama/Meta-Llama-3-8B"

MASTER_ADDR=localhost MASTER_PORT=51234 torchrun --nproc_per_node=8 test_llama.py --tp_size 4 --sp_size 2 --seq_len 1024 --model_name "meta-llama/Meta-Llama-3-8B"

# With strip
# Running model_name = 'meta-llama/Meta-Llama-3-8B' with seq_len = 65536: 3302.7470703125 ms
# Running model_name = 'meta-llama/Meta-Llama-3-8B' with seq_len = 262144: 17198.33203125 ms
# Running model_name = 'meta-llama/Meta-Llama-3-8B' with seq_len = 524288: 55582.921875 ms

# Without strip
# Running model_name = 'meta-llama/Meta-Llama-3-8B' with seq_len = 524288: 83453.84375 ms