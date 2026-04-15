#!/bin/bash

# 单GPU数据生成启动脚本
# 确保GPU正确使用并生成25,280个样本

set -e

cd "$(dirname "$0")"

echo "========================================="
echo "🚀 单GPU多维度数据生成启动脚本"
echo "========================================="
echo ""

# 1. 检查Ollama服务
echo "📋 步骤1: 检查Ollama服务..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "❌ Ollama服务未响应 (http://localhost:11434)"
    echo "请在新终端中执行: ollama serve"
    exit 1
fi
echo "✓ Ollama服务正常"
echo ""

# 2. 检查模型
echo "📋 步骤2: 检查LLM模型..."
if ! curl -s http://localhost:11434/api/tags | grep -q "qwen2.5:14b"; then
    echo "⚠ qwen2.5:14b 未加载，开始下载..."
    echo "提示: 这可能需要10-15分钟"
    echo ""
    ollama pull qwen2.5:14b
fi
echo "✓ qwen2.5:14b 已准备"
echo ""

# 3. 检查GPU
echo "📋 步骤3: 检查GPU配置..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || {
    echo "⚠ 未检测到NVIDIA GPU"
    exit 1
}
echo ""

# 4. 启动生成
echo "📋 步骤4: 启动数据生成..."
echo "生成规模: 79 indicators × 4³ dimension combos × 5 reps = 25,280 samples"
echo "预计耗时: 4-6小时 (单GPU)"
echo "输出路径: /home/user/workspace/SHLi/AI for radicalisation/data/generated_samples"
echo ""
echo "========================================="
echo ""

# 记录开始时间
START_TIME=$(date +%s)

# 使用CUDA_VISIBLE_DEVICES=0指定GPU0运行
export CUDA_VISIBLE_DEVICES=0

python3 data_generator_ollama_single_gpu.py

# 计算耗时
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "========================================="
echo "✅ 生成完成!"
echo "耗时: ${MINUTES}分${SECONDS}秒"
echo "========================================="
