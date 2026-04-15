#!/bin/bash
# ============ LLM模型下载与启动脚本 ============
# 用于qwen2.5:14b的快速部署和测试

set -e

echo "=================================="
echo "🚀 Ollama LLM 部署脚本"
echo "=================================="

# 检查Ollama是否运行
check_ollama() {
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "❌ Ollama未运行，请先执行: ollama serve"
        exit 1
    fi
    echo "✓ Ollama服务运行中"
}

# 下载更强的LLM
download_model() {
    MODEL=$1
    echo ""
    echo "📥 下载模型: $MODEL"
    echo "这可能需要几分钟（模型约为 9GB）"
    ollama pull "$MODEL"
    echo "✓ 模型下载完成"
}

# 列出已有模型
list_models() {
    echo ""
    echo "📋 已有模型："
    curl -s http://localhost:11434/api/tags | python3 -m json.tool | grep -A 2 '"name"' || echo "无"
}

# 测试模型
test_model() {
    MODEL=$1
    echo ""
    echo "🧪 测试模型: $MODEL"
    python3 << 'EOF'
import sys
from llama_index.llms.ollama import Ollama

try:
    llm = Ollama(model=sys.argv[1], temperature=0.7)
    response = llm.complete("Write a brief 1-sentence message about radicalization awareness.")
    print(f"✓ 模型测试成功")
    print(f"响应: {response.text[:100]}...")
except Exception as e:
    print(f"❌ 模型测试失败: {e}")
    sys.exit(1)
EOF
}

# 主流程
main() {
    check_ollama
    
    echo ""
    echo "选择操作："
    echo "1) 下载并切换到 qwen2.5:14b (推荐，相比7b更强)"
    echo "2) 下载 qwen2:72b (最强但显存需求更高)"
    echo "3) 列出已有模型"
    echo "4) 测试当前模型"
    
    read -p "请选择 (1-4): " choice
    
    case $choice in
        1)
            download_model "qwen2.5:14b"
            test_model "qwen2.5:14b"
            echo ""
            echo "✓ 现在可以使用 qwen2.5:14b 进行生成"
            echo "  推荐命令: python3 data_generator_ollama_multiGPU.py"
            ;;
        2)
            download_model "qwen2:72b"
            test_model "qwen2:72b"
            ;;
        3)
            list_models
            ;;
        4)
            read -p "输入模型名称 (默认: qwen2.5:14b): " MODEL
            MODEL=${MODEL:-qwen2.5:14b}
            test_model "$MODEL"
            ;;
        *)
            echo "❌ 无效选择"
            exit 1
            ;;
    esac
}

main "$@"
