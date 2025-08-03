# MPJPE Evaluation for HumanMAC-main

This document describes the MPJPE (Mean Per Joint Position Error) evaluation system added to the HumanMAC-main project for human motion prediction evaluation.

## Overview

MPJPE is a standard metric in human motion prediction research that measures the average Euclidean distance between predicted and ground truth joint positions. This implementation provides evaluation at standard time intervals commonly used in the field.

## Features

- **Standard Evaluation Intervals**: 80ms, 160ms, 320ms, 400ms, 1000ms
- **Multi-modal Support**: Both single-modal and multi-modal MPJPE evaluation
- **Dataset Compatibility**: Supports H36M (50 FPS) and HumanEva (60 FPS) datasets
- **Enhanced Model Support**: Compatible with condition-enhanced models
- **Configurable**: Easy to modify intervals and FPS settings
- **Comprehensive Logging**: Detailed results logging and CSV export

## Files Added/Modified

### New Files
- `utils/mpjpe_evaluation.py`: Core MPJPE evaluation utilities
- `evaluate_mpjpe.py`: Standalone evaluation script
- `test_mpjpe.py`: Test script for validation
- `MPJPE_EVALUATION_README.md`: This documentation

### Modified Files
- `utils/metrics.py`: Added MPJPE computation functions
- `utils/evaluation.py`: Integrated MPJPE into main evaluation pipeline
- `cfg/h36m.yml`: Added FPS and interval configuration
- `cfg/h36m_fast.yml`: Added FPS and interval configuration
- `cfg/humaneva.yml`: Added FPS and interval configuration

## Usage

### 1. Integrated Evaluation (推荐)

MPJPE指标会在正常评价过程中自动计算：

```bash
# H36M数据集评价
python main.py --cfg h36m --mode eval --ckpt ./checkpoints/ckpt_ema_500.pt

# H36M快速配置评价
python main.py --cfg h36m_fast --mode eval --ckpt ./checkpoints/ckpt_ema_150.pt

# HumanEva数据集评价  
python main.py --cfg humaneva --mode eval --ckpt ./checkpoints/humaneva_ckpt.pt
```

结果将包含原有指标（APD, ADE, FDE, MMADE, MMFDE）和新的MPJPE指标。

### 2. 独立MPJPE评价

仅进行MPJPE评价：

```bash
# H36M数据集
python evaluate_mpjpe.py --cfg h36m --ckpt ./checkpoints/ckpt_ema_500.pt

# H36M快速配置
python evaluate_mpjpe.py --cfg h36m_fast --ckpt ./checkpoints/ckpt_ema_150.pt

# HumanEva数据集
python evaluate_mpjpe.py --cfg humaneva --ckpt ./checkpoints/humaneva_ckpt.pt
```

### 3. 测试MPJPE实现

验证MPJPE实现的正确性：

```bash
python test_mpjpe.py
```

### 4. 编程使用

```python
from utils.mpjpe_evaluation import MPJPEEvaluator, evaluate_motion_prediction_mpjpe

# 初始化评价器
evaluator = MPJPEEvaluator(fps=50, intervals_ms=[80, 160, 320, 400, 1000])

# 评价预测结果
# pred: [K, t_pred, 3*joints] - K个预测样本
# gt: [t_pred, 3*joints] - 真实值
# gt_multi: [M, t_pred, 3*joints] - 多模态真实值（可选）
results = evaluator.compute_all_mpjpe_metrics(pred, gt, gt_multi)

# 或使用便捷函数
results = evaluate_motion_prediction_mpjpe(pred, gt, gt_multi, fps=50)
```

## 配置

### 数据集特定设置

**H36M数据集 (cfg/h36m.yml, cfg/h36m_fast.yml):**
```yaml
fps: 50  # 50 FPS
mpjpe_intervals: [80, 160, 320, 400, 1000]  # 标准间隔（毫秒）
```

**HumanEva数据集 (cfg/humaneva.yml):**
```yaml
fps: 60  # 60 FPS  
mpjpe_intervals: [80, 160, 320, 400, 1000]  # 标准间隔（毫秒）
```

### 自定义间隔

要在不同时间间隔进行评价，修改配置文件中的`mpjpe_intervals`：

```yaml
mpjpe_intervals: [100, 200, 500, 1000]  # 自定义间隔
```

## 输出指标

### 单模态MPJPE
- `MPJPE_80ms`: 80毫秒时的MPJPE
- `MPJPE_160ms`: 160毫秒时的MPJPE  
- `MPJPE_320ms`: 320毫秒时的MPJPE
- `MPJPE_400ms`: 400毫秒时的MPJPE
- `MPJPE_1000ms`: 1000毫秒时的MPJPE

### 多模态MPJPE
- `MMMPJPE_80ms`: 80毫秒时的多模态MPJPE
- `MMMPJPE_160ms`: 160毫秒时的多模态MPJPE
- `MMMPJPE_320ms`: 320毫秒时的多模态MPJPE
- `MMMPJPE_400ms`: 400毫秒时的多模态MPJPE
- `MMMPJPE_1000ms`: 1000毫秒时的多模态MPJPE

## 输出文件

### CSV文件
- `results/[dataset]/stats_latest.csv`: 最新评价结果
- `results/[dataset]/stats.csv`: 历史评价结果

### 日志文件
- `results/[dataset]/log.txt`: 详细评价日志
- `results/[dataset]/mpjpe_results.txt`: MPJPE专用结果文件（独立评价）

## 技术细节

### MPJPE计算

1. **单模态MPJPE**: 
   - 在K个样本中选择整体误差最小的预测
   - 在每个时间间隔计算每个关节的欧几里得距离
   - 对所有关节求平均

2. **多模态MPJPE**:
   - 对每个预测，在多个真实值中找到最接近的
   - 计算到最接近真实值的每关节距离
   - 对预测和关节求平均

### 帧索引计算

时间间隔转换为帧索引：
```python
frame_index = int(milliseconds * fps / 1000)
```

### 数据格式

- **输入**: `[batch, time, 3*joints]` 其中关节坐标被展平（x,y,z坐标）
- **H36M**: 过滤后17个关节（原始32个关节）
- **HumanEva**: 关节数量根据数据集配置变化

## 兼容性

### 增强模型支持

该MPJPE评价系统兼容带有condition参数的增强模型：

```python
# 自动检测并支持condition参数
sampled_motion = diffusion.sample_ddim(model_select,
                                       traj_dct,
                                       traj_dct_cond,
                                       mode_dict,
                                       condition=condition)  # 可选参数
```

### 向后兼容

对于不支持condition参数的原始模型，系统会自动回退到标准调用方式。

## 验证

MPJPE实现已通过以下测试验证：
- ✅ 基础MPJPE计算正确性
- ✅ 已知误差的MPJPE验证
- ✅ 多模态MPJPE计算
- ✅ 不同FPS下的帧索引计算
- ✅ 短预测序列处理
- ✅ 与原有评价指标的集成
- ✅ 便捷函数的功能验证

## 性能考虑

- GPU加速用于可用时
- 批处理以提高计算效率
- 大规模预测集的内存高效处理
- 长时间评价的进度条显示

## 故障排除

### 常见问题

1. **CUDA内存不足**: 减少批量大小或使用CPU评价
2. **帧索引错误**: 验证FPS设置与数据集匹配
3. **形状不匹配**: 确保预测格式符合预期输入

### 调试模式

在评价脚本中启用详细日志记录以进行调试。

## 参考文献

- Human3.6M数据集评价协议
- 标准运动预测评价实践
- 多模态运动预测评价方法

## 未来增强

- 支持额外的评价间隔
- 按动作的MPJPE分解
- MPJPE随时间的可视化
- 与其他运动预测指标的集成
