#!/bin/bash

# ============ Simulator 启动脚本 ============
# 用于生成样本数据

echo "======================== Simulator 启动 ========================"
echo

# 检查 Python 环境
echo "🔍 检查环境..."
python3 --version
python3 -c "import torch; print(f'✓ PyTorch 版本: {torch.__version__}')"
python3 -c "import torch; print(f'✓ CUDA 可用: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'✓ GPU 数量: {torch.cuda.device_count()}')"
if [ $? -ne 0 ]; then
    echo "❌ 环境检查失败，请确保已安装 PyTorch"
    exit 1
fi

echo

# 安装依赖（可选，如果还未安装）
echo "📦 检查依赖..."
pip list | grep -q transformers
if [ $? -ne 0 ]; then
    echo "📥 安装依赖..."
    pip install -r requirements.txt -q
else
    echo "✓ 依赖已安装"
fi

echo

# 运行生成脚本
echo "🚀 启动数据生成器..."
echo "=================================================="
python3 data_generator.py
RESULT=$?
echo "=================================================="

if [ $RESULT -eq 0 ]; then
    echo
    echo "✅ 生成完成!"
    echo "📁 样本已保存到: generated_samples/samples.jsonl"
else
    echo
    echo "❌ 生成失败 (码: $RESULT)"
    exit 1
fi
