# Setup Environment


Setup environment 
```bash
mamba create -p ./env python=3.10
mamba activate ./env
mamba install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -e .
pip install vllm 'torch==2.4.0'
```

Test 
```bash
MASTER_ADDR=localhost MASTER_PORT=51234 torchrun --nproc_per_node=8 test_llama.py --tp_size 1 --sp_size 8 --seq_len $((2 ** 10 * 64)) --model_name "meta-llama/Meta-Llama-3-8B"
```