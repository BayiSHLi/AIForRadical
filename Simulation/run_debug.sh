#!/bin/bash

# ============ 调试模式运行脚本 ============
# 用于诊断 CUDA 错误

echo "======================== 调试模式启动 ========================"
echo

# 设置调试环境变量
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

echo "🔍 调试环境变量已设置:"
echo "   CUDA_LAUNCH_BLOCKING=1"
echo "   TORCH_USE_CUDA_DSA=1"
echo

# 检查 CUDA 状态
echo "🔍 检查 CUDA 状态..."
python3 -c "
import torch
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA 版本: {torch.version.cuda}')
    print(f'GPU 数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'  显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f}GB')
"

echo

# 运行生成脚本（调试模式）
echo "🚀 启动数据生成器（调试模式）..."
echo "=================================================="
python3 data_generator.py
RESULT=$?
echo "=================================================="

if [ $RESULT -eq 0 ]; then
    echo
    echo "✅ 生成完成!"
else
    echo
    echo "❌ 生成失败 (码: $RESULT)"
    echo
    echo "💡 故障排查建议:"
    echo "   1. 检查显存是否充足"
    echo "   2. 尝试使用更小的模型"
    echo "   3. 减少 max_length 参数"
    echo "   4. 查看上方的详细错误信息"
    exit 1
fi
