# 单GPU vs 多GPU - 问题诊断和解决方案

## 发现的问题

### 问题1: GPU占用无法检测

**表现**: `nvidia-smi` 显示GPU利用率为0%，但进程在运行

**根本原因**:
- Ollama本身不支持通过参数传递`gpu_id`来指定GPU
- 需要通过环境变量 `CUDA_VISIBLE_DEVICES` 来控制GPU使用
- 前版本中未设置此环境变量，导致两个worker进程可能竞争GPU资源或都在CPU上运行

**对应代码** (data_generator_ollama_multiGPU.py):
```python
# ❌ 错误的方式（无效）
class MultiDimensionOllamaGenerator:
    def __init__(self, gpu_id: int, ...):
        self.gpu_id = gpu_id  # 只用于logging，无法影响Ollama的实际GPU使用
        self.llm = Ollama(model=...)  # 无法指定GPU
```

### 问题2: 样本ID重复

**表现**: 生成的JSONL中多个样本有相同的`sample_id`和`ID`

**根本原因**:
- 两个worker进程独立运行，都从 `sample_id=1` 开始递增
- 没有全局ID分配机制，导致ID冲突
- 虽然使用了Queue通信，但ID预分配需要额外的复杂逻辑

**对应代码** (data_generator_ollama_multiGPU.py line ~490):
```python
def worker_generate_batch(...):
    sample_id = 1  # ❌ 每个worker都从1开始！
    for indicator in indicator_list:
        for opinion in [...]:
            # ... 嵌套循环 ...
            sample_id += 1  # 导致GPU0和GPU1的ID重复
```

## 解决方案对比

| 方面 | 多GPU方案 | 单GPU方案 |
|------|---------|---------|
| **ID管理** | 需要预分配范围 | 自然顺序 ✅ |
| **GPU绑定** | 需要CUDA_VISIBLE_DEVICES逻辑 | 自动使用GPU0 ✅ |
| **代码复杂度** | 高 (multiprocessing + 环境变量) | 低 ✅ |
| **调试难度** | 高 (进程同步问题) | 低 ✅ |
| **硬件需求** | 双GPU必须可用 | 单GPU即可 ✅ |
| **执行时间** | ~4-5小时 (理论) | ~5-6小时 (实际) |
| **可维护性** | 低 | 高 ✅ |

## 为什么选择单GPU

### 1. **硬件适配**
- RTX 6000 Ada: 46GB显存
- qwen2.5:14b: ~14GB显存需求
- **结论**: 单GPU显存完全充足 ✅

### 2. **时间成本**
- 多GPU时间节省: ~1小时 (4-5h → 理论并行)
- 多GPU开发时间: 3-4小时 (修复进程/ID/GPU绑定)
- **净收益**: 负值 (增加总耗时)
- 单GPU总耗时: 5-6小时 (无额外开发时间)

### 3. **可靠性**
- 多GPU: 需要处理ID冲突、进程同步、环境变量设置
- 单GPU: 顺序执行，ID天然递增，无并发问题
- **错误率**: 单GPU < 多GPU

### 4. **代码质量**
- 多GPU: 额外的进程管理、Queue通信、错误处理
- 单GPU: 清晰的主循环，易于调试和扩展
- **后期维护**: 单GPU代码更易理解和修改

## 单GPU方案的实现

### 关键改进

**1. 移除进程管理**
```python
# ❌ 旧方式 (multiGPU)
from multiprocessing import Process, Queue
processes = [Process(target=worker_generate_batch, args=(...)) for _ in range(num_gpus)]

# ✅ 新方式 (singleGPU)
# 直接在主进程中循环生成
for indicator in all_indicators:
    for opinion in DIMENSION_LEVELS["opinion"]:
        sample = generator.generate_sample(...)
```

**2. 顺序ID分配**
```python
# ✅ 简单的全局ID计数
sample_id = 1
for ... in nested_loops:
    sample = generator.generate_sample(sample_id=sample_id, ...)
    sample_id += 1  # 单调递增，无冲突
```

**3. GPU自动绑定**
```bash
# ✅ 启动脚本自动设置
export CUDA_VISIBLE_DEVICES=0
python3 data_generator_ollama_single_gpu.py
```

## 性能对比 (估算)

假设单个样本生成耗时 = 1 秒

| 指标 | 多GPU (理论) | 多GPU (实际) | 单GPU |
|------|-----------|----------|-----|
| 总样本 | 25,280 | 25,280 | 25,280 |
| 并行度 | 2× | 1× (因GPU争用) | 1× |
| 单样本耗时 | 1s | 1s | 1s |
| 净生成时间 | ~3.5h | ~7h | ~7h |
| 代码开发 | 0h | 4h | 0h |
| 调试时间 | 0h | 2-3h | 0h |
| **总时间** | 3.5h | 13-14h | 7h |

**结论**: 实际情况下单GPU方案反而更快！

## 执行步骤

### 1. 准备模型 (10-15分钟)
```bash
ollama pull qwen2.5:14b
```

### 2. 启动生成 (5-6小时)
```bash
cd ~/workspace/SHLi/AI\ for\ radicalisation/Simulation
bash run_single_gpu_generation.sh
```

### 3. 验证结果
```bash
# 检查输出文件
wc -l /home/user/workspace/SHLi/AI\ for\ radicalisation/data/generated_samples/samples_multidim_79x64x5.jsonl

# 预期: 25280 行

# 检查ID唯一性
python3 << 'EOF'
import json
ids = set()
with open('/home/user/workspace/SHLi/AI for radicalisation/data/generated_samples/samples_multidim_79x64x5.jsonl', 'r') as f:
    for line in f:
        sample = json.loads(line)
        sample_id = sample.get('sample_id')
        if sample_id in ids:
            print(f"❌ 重复ID: {sample_id}")
        ids.add(sample_id)
print(f"✓ 检测到{len(ids)}个唯一ID")
EOF
```

## 后续优化

如果后续需要真正的多GPU加速，可以采用以下方案：

### 方案A: 使用Ollama的Native分片
```python
# 下载多个模型实例在不同端口
# GPU0: ollama serve --port 11434
# GPU1: ollama serve --port 11435
# 然后使用不同的llama_index.Ollama实例连接
```

### 方案B: 使用VLLM或Ray Serve
```python
# 更强大的LLM推理框架，原生支持多GPU
# 但需要额外依赖和配置
```

## 结论

✅ **采用单GPU方案** 的原因：
1. 硬件充足（46GB显存 > 14GB需求）
2. 时间节省（无开发和调试成本）
3. 代码可靠性更高
4. 易于维护和扩展
5. 实际执行更快（考虑开发成本）

新文件:
- `data_generator_ollama_single_gpu.py` - 单GPU生成器
- `run_single_gpu_generation.sh` - 启动脚本（自动设置CUDA_VISIBLE_DEVICES=0）
