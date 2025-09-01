#!/bin/bash
echo "Setting up Tokenize-Anything environment..."

# 创建并激活 conda 环境
conda create -p /opt/data/private/envs/tokenize_anything python=3.9 -y
source activate /opt/data/private/envs/tokenize_anything

# 设置 CUDA 环境变量
export CUDA_HOME=/usr/local/cuda-11.3
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 安装指定版本的 PyTorch
pip3 install torch==2.1.0+cu118 torchvision==0.16.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# 安装指定版本的 numpy
pip install numpy==1.24.3

# 安装 flash-attention
pip install flash-attn==2.3.3 --no-build-isolation

# 安装项目基础依赖
pip install opencv-python
pip install Pillow
pip install gradio-image-prompter
pip install sentencepiece

# 安装项目本身
CUDA_HOME=/usr/local/cuda-11.3 pip install -e .

echo "Environment setup completed!"