#!/bin/bash

set -e

# 手动添加 conda 到 PATH（替换为你的实际 conda 安装路径）
export PATH=~/space/.miniconda/condabin:$PATH

cd ~/lxl/DEIM-DEIM

# 初始化 conda hook，优先 zsh，然后 bash
if eval "$(conda shell.zsh hook)" 2>/dev/null; then
    echo "Using zsh conda hook"
elif eval "$(conda shell.bash hook)" 2>/dev/null; then
    echo "Using bash conda hook"
else
    echo "Error: Failed to initialize conda. Please check conda installation."
    exit 1
fi

# 激活环境
conda activate deim || { echo "Error: Failed to activate deim environment."; exit 1; }

ulimit -n 65536

# 优化多线程性能（避免系统过载）
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# GPU配置
export CUDA_VISIBLE_DEVICES=0,2
NUM_GPUS=2

# torchrun --master_port=9928 --nproc_per_node=$NUM_GPUS train.py -c configs/yaml/deim.yml

# torchrun --master_port=9928 --nproc_per_node=$NUM_GPUS train.py -c configs/dfine/dfine_hgnetv2_n_mal_custom.yml

# torchrun --master_port=9928 --nproc_per_node=$NUM_GPUS train.py -c configs/deim/deim_hgnetv2_n_custom.yml

# 都是160轮ok。别忘了教龙哥

# CUDA_VISIBLE_DEVICES=4 python train.py -c configs/yaml/deim_dfine_hgnetv2_n_mg_test.yml

CUDA_VISIBLE_DEVICES=3 python train.py -c configs/yaml/dfine.yml

echo "✅ Training completed!"
