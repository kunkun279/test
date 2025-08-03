# ckpt_ema_500.pt 使用指南

本指南说明如何正确使用 `ckpt_ema_500.pt` 检查点进行MPJPE评价。

## 🔍 问题分析

`ckpt_ema_500.pt` 检查点的条件编码器架构与当前模型代码不匹配：

### 检查点中的条件编码器架构
```
Layer 0: Linear(4, 192) + bias     # 输入4维 -> 192维隐藏层
Layer 2: Linear(192, 384) + bias   # 192维 -> 384维输出
Layer 3: LayerNorm(384) + bias     # 384维归一化
```

### 当前模型期望的架构
```
Layer 0: Linear(4, 384) + bias     # 输入4维 -> 384维隐藏层
Layer 2: Linear(384, 384) + bias   # 384维 -> 384维输出
Layer 3: LayerNorm(384) + bias     # 384维归一化
```

## ✅ 解决方案

我们创建了兼容的模型架构和评价脚本来解决这个问题。

## 🚀 使用方法

### 方法1: 使用兼容评价脚本（推荐）

```bash
# 使用专门的兼容评价脚本
python evaluate_ckpt_500_compatible.py --cfg h36m_fast --ckpt ./checkpoints/ckpt_ema_500.pt
```

这个脚本会：
- 自动创建兼容的模型架构
- 正确加载 `ckpt_ema_500.pt` 检查点
- 运行完整的MPJPE评价
- 保存结果到 `compatible_mpjpe_results.txt`

### 方法2: 检查条件数据

如果遇到条件数据问题，先运行：

```bash
# 检查并修复条件数据
python simple_condition_check.py
```

### 方法3: 分析检查点架构

了解检查点详细信息：

```bash
# 分析条件编码器架构
python fix_condition_encoder.py --ckpt ./checkpoints/ckpt_ema_500.pt

# 分析完整检查点架构
python match_checkpoint_config.py --ckpt ./checkpoints/ckpt_ema_500.pt
```

## 📊 预期结果

使用兼容脚本后，您将获得：

### 标准指标
- APD, ADE, FDE, MMADE, MMFDE

### MPJPE指标
- **单模态MPJPE**: MPJPE_80ms, MPJPE_160ms, MPJPE_320ms, MPJPE_400ms, MPJPE_1000ms
- **多模态MPJPE**: MMMPJPE_80ms, MMMPJPE_160ms, MMMPJPE_320ms, MMMPJPE_400ms, MMMPJPE_1000ms

## 🔧 技术细节

### 兼容模型架构

`models/compatible_transformer.py` 中的 `CompatibleMotionTransformer` 类：

```python
# 精确匹配检查点的条件编码器
self.condition_encoder = nn.Sequential(
    nn.Linear(condition_dim, 192),  # Layer 0: 匹配检查点
    nn.ReLU(),                      # Layer 1: 激活函数
    nn.Linear(192, 384),           # Layer 2: 匹配检查点
    nn.LayerNorm(384)              # Layer 3: 匹配检查点
)
```

### 配置文件

使用 `h36m_fast` 配置，因为它与检查点的主要架构匹配：
- `latent_dims: 384` ✅
- `num_layers: 6` ✅
- `num_heads: 8` ✅

## 🆚 与其他检查点的对比

| 检查点 | 配置 | 条件编码器 | 使用方法 |
|--------|------|------------|----------|
| ckpt_ema_150.pt | h36m | 标准架构 | `python main.py --cfg h36m --mode eval --ckpt ./checkpoints/ckpt_ema_150.pt` |
| ckpt_ema_500.pt | h36m_fast | 兼容架构 | `python evaluate_ckpt_500_compatible.py --cfg h36m_fast --ckpt ./checkpoints/ckpt_ema_500.pt` |

## 📁 输出文件

评价完成后，结果保存在：

```
results/h36m_fast_*/
├── compatible_mpjpe_results.txt    # MPJPE专用结果
├── log/
│   └── compatible_mpjpe_eval_log.txt  # 详细日志
└── ...
```

## 🔍 故障排除

### 问题1: 条件数据加载失败
```bash
python simple_condition_check.py
```

### 问题2: 检查点加载失败
```bash
python simple_checkpoint_test.py --ckpt ./checkpoints/ckpt_ema_500.pt
```

### 问题3: 架构不匹配
确保使用兼容评价脚本：
```bash
python evaluate_ckpt_500_compatible.py
```

## 💡 重要提示

1. **不要使用标准的 `main.py`** 来评价 `ckpt_ema_500.pt`，因为架构不匹配
2. **使用专门的兼容脚本** `evaluate_ckpt_500_compatible.py`
3. **配置文件使用 `h36m_fast`**，不是 `h36m`
4. **条件数据必须正确加载**，如有问题先运行条件数据检查

## 🎯 快速开始

```bash
# 1. 检查条件数据
python simple_condition_check.py

# 2. 运行兼容评价
python evaluate_ckpt_500_compatible.py --cfg h36m_fast --ckpt ./checkpoints/ckpt_ema_500.pt

# 3. 查看结果
cat results/h36m_fast_*/compatible_mpjpe_results.txt
```

## 📞 获取帮助

如果仍有问题：

1. 检查日志文件：`results/h36m_fast_*/log/compatible_mpjpe_eval_log.txt`
2. 运行诊断工具：`python fix_condition_encoder.py --ckpt ./checkpoints/ckpt_ema_500.pt`
3. 验证检查点：`python simple_checkpoint_test.py --ckpt ./checkpoints/ckpt_ema_500.pt`

现在您应该能够成功使用 `ckpt_ema_500.pt` 进行MPJPE评价了！🎉
