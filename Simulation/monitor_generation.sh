#!/bin/bash

# 监控脚本：实时展示生成进度和GPU使用

echo "🔍 多维度数据生成监控面板"
echo "========================================="
echo ""

while true; do
    clear
    echo "🔍 多维度数据生成监控面板"
    echo "========================================"
    echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # 1. 模型下载进度
    echo "📥 模型下载进度:"
    if pgrep -f "ollama pull" > /dev/null; then
        tail -1 /tmp/ollama_pull.log 2>/dev/null | grep -o '[0-9]*%' || echo "  处理中..."
    else
        if ollama list | grep -q "qwen2.5:14b"; then
            echo "  ✅ qwen2.5:14b 已完成"
        else
            echo "  ⏳ 等待下载..."
        fi
    fi
    echo ""
    
    # 2. 数据生成进度
    echo "📊 数据生成进度:"
    SAMPLES=$(wc -l < /home/user/workspace/SHLi/AI\ for\ radicalisation/data/generated_samples/samples_multidim_79x64x5.jsonl 2>/dev/null || echo 0)
    if [ "$SAMPLES" -gt 0 ]; then
        PCT=$((SAMPLES * 100 / 25280))
        BAR=$(printf '█%.0s' $(seq 1 $((PCT / 5))))
        EMPTY=$(printf '░%.0s' $(seq 1 $((20 - PCT / 5))))
        echo "  $BAR$EMPTY $SAMPLES/25280 ($PCT%)"
        
        # 计算速率
        if [ -f /tmp/last_count.txt ]; then
            LAST_COUNT=$(cat /tmp/last_count.txt)
            RATE=$((SAMPLES - LAST_COUNT))
            if [ $RATE -gt 0 ]; then
                REMAINING=$((25280 - SAMPLES))
                ETA_SECS=$((REMAINING / (RATE / 10)))  # 假设每10秒计算一次速率
                ETA_HOURS=$((ETA_SECS / 3600))
                ETA_MINS=$(((ETA_SECS % 3600) / 60))
                echo "  ⏱️  预计ETA: ${ETA_HOURS}小时${ETA_MINS}分钟 (速率: $((RATE/10)) samples/s)"
            fi
        fi
        echo "$SAMPLES" > /tmp/last_count.txt
    else
        if pgrep -f "data_generator_ollama_single_gpu" > /dev/null; then
            echo "  ⏳ 初始化中..."
        else
            echo "  ⏸️  等待开始..."
        fi
    fi
    echo ""
    
    # 3. GPU使用情况
    echo "🖥️  GPU使用情况:"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | awk '{print "  GPU"$1": "$2"% 显存 "$3"MB/"$4"MB"}'
    echo ""
    
    # 4. 进程状态
    echo "⚙️  进程状态:"
    if pgrep -f "ollama pull" > /dev/null; then
        echo "  ✅ 模型下载: 运行中"
    else
        echo "  ⏸️  模型下载: 未运行"
    fi
    
    if pgrep -f "data_generator_ollama_single_gpu" > /dev/null; then
        echo "  ✅ 数据生成: 运行中"
    else
        echo "  ⏸️  数据生成: 未运行"
    fi
    
    if pgrep -f "Ollama" > /dev/null; then
        echo "  ✅ Ollama服务: 运行中"
    else
        echo "  ⚠️  Ollama服务: 未运行"
    fi
    echo ""
    
    # 5. 输出文件状态
    echo "📁 输出文件:"
    OUTPUT_FILE="/home/user/workspace/SHLi/AI for radicalisation/data/generated_samples/samples_multidim_79x64x5.jsonl"
    if [ -f "$OUTPUT_FILE" ]; then
        SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
        echo "  📄 $OUTPUT_FILE"
        echo "     大小: $SIZE"
    else
        echo "  ⏳ 文件未创建"
    fi
    echo ""
    
    echo "========================================"
    echo "按 Ctrl+C 退出 | 自动刷新间隔: 10秒"
    sleep 10
done
