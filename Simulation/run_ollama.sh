#!/bin/bash

# ============ Ollama 数据生成器启动脚本 ============
# 使用本地 Ollama qwen2.5:7b 快速生成样本

echo "======================== Simulator (Ollama) ========================"
echo

# 检查 Ollama 服务
echo "🔍 检查 Ollama 服务..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✓ Ollama 服务运行正常"
else
    echo "❌ Ollama 服务未运行"
    echo "   请启动 Ollama: ollama serve"
    exit 1
fi

echo

# 检查模型
echo "🔍 检查 qwen2.5:7b 模型..."
if curl -s http://localhost:11434/api/tags | grep -q "qwen2.5:7b"; then
    echo "✓ 模型已下载"
else
    echo "⚠️  模型未找到，尝试下载..."
    ollama pull qwen2.5:7b
    if [ $? -ne 0 ]; then
        echo "❌ 模型下载失败"
        exit 1
    fi
fi

echo

# 运行生成脚本
echo "🚀 启动数据生成器..."
echo "=================================================="
python3 data_generator_ollama.py
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
