# HumanMAC Human3.6M MPJPE 评估指南

## 🎯 概述

本指南提供了在HumanMAC-main项目中评估Human3.6M数据集MPJPE指标的完整方案，支持在80、160、320、400、1000ms时间点评估不同动作的预测精度。

## 📁 新增文件

1. **`eval_h36m_mpjpe_fixed.py`** - 主要评估脚本（修复版）
2. **`eval_h36m_debug.py`** - 调试测试脚本
3. **`eval_h36m_simple.py`** - 简化测试脚本
4. **`run_h36m_evaluation.sh`** - 批处理运行脚本
5. **`H36M_EVALUATION_GUIDE.md`** - 本使用指南

## 🚀 快速开始

### 1. 环境检查和数据测试

```bash
cd HumanMAC-main
python eval_h36m_simple.py --all_tests
```

这会测试：
- 数据加载功能
- MPJPE计算功能
- 基线评估（零速度预测）

### 2. 调试测试

```bash
python eval_h36m_debug.py
```

这会测试：
- 所有导入
- 配置加载
- 模型创建
- 数据预处理
- 模型推理

### 3. 完整模型评估

```bash
# 评估所有动作
python eval_h36m_mpjpe_fixed.py \
    --model_path checkpoints/ckpt_ema_500.pt \
    --num_samples 50 \
    --output_dir ./eval_results

# 评估特定动作
python eval_h36m_mpjpe_fixed.py \
    --model_path checkpoints/ckpt_ema_500.pt \
    --actions Walking Eating Sitting \
    --num_samples 50
```

### 4. 使用批处理脚本

```bash
# 使脚本可执行
chmod +x run_h36m_evaluation.sh

# 运行测试
./run_h36m_evaluation.sh test

# 运行快速评估
./run_h36m_evaluation.sh quick

# 评估Walking动作
./run_h36m_evaluation.sh walking --num_samples 20
```

## 📊 评估指标说明

### MPJPE (Mean Per Joint Position Error)
- **定义**: 预测关节位置与真实关节位置的平均欧氏距离
- **单位**: 毫米 (mm)
- **计算**: `MPJPE = mean(||pred_joints - gt_joints||_2)`

### 时间点设置
- **80ms**: 4帧 (50 FPS × 0.08s)
- **160ms**: 8帧 (50 FPS × 0.16s)
- **320ms**: 16帧 (50 FPS × 0.32s)
- **400ms**: 20帧 (50 FPS × 0.40s)
- **1000ms**: 50帧 (50 FPS × 1.0s)

### Human3.6M动作类别
```
'Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning',
'Photo', 'Posing', 'Purchases', 'Sitting', 'SittingDown',
'Smoking', 'Waiting', 'WalkDog', 'Walking', 'WalkTogether'
```

## 🔧 详细使用方法

### 主要评估脚本参数

```bash
python eval_h36m_mpjpe_fixed.py [OPTIONS]

选项:
  --config PATH          配置文件路径 (默认: cfg/h36m.yml)
  --model_path PATH      训练好的模型路径 (默认: checkpoints/ckpt_ema_500.pt)
  --output_dir PATH      结果输出目录 (默认: ./eval_results)
  --num_samples INT      生成预测样本数 (默认: 50)
  --actions [ACTIONS]    指定评估的动作 (默认: 全部)
  --device DEVICE        计算设备 (默认: cuda)
```

### 批处理脚本命令

```bash
./run_h36m_evaluation.sh [COMMAND] [OPTIONS]

命令:
  test                    运行基础测试
  quick                   运行快速基线评估
  eval                    运行完整模型评估
  walking                 仅评估Walking动作
  locomotion              评估运动类动作 (Walking, WalkDog, WalkTogether)
  sitting                 评估坐姿类动作 (Sitting, SittingDown)
  all                     运行所有评估 (test + quick + full)

选项:
  --model_path PATH       模型检查点路径
  --output_dir PATH       输出目录
  --num_samples N         样本数量
  --device DEVICE         使用的设备
```

### 示例命令

```bash
# 1. 基础测试
./run_h36m_evaluation.sh test

# 2. 快速评估
./run_h36m_evaluation.sh quick

# 3. 完整评估
./run_h36m_evaluation.sh eval

# 4. 高精度评估（更多样本）
./run_h36m_evaluation.sh eval --num_samples 100

# 5. 特定动作评估
./run_h36m_evaluation.sh walking --num_samples 20

# 6. 运动类动作评估
./run_h36m_evaluation.sh locomotion --output_dir locomotion_results

# 7. CPU评估（如果GPU不可用）
./run_h36m_evaluation.sh eval --device cpu
```

## 📈 结果输出

### 控制台输出示例
```
=== HumanMAC Human3.6M MPJPE Evaluation ===
Model: checkpoints/ckpt_ema_500.pt
Device: cuda
Number of samples: 50
Time horizons: [80, 160, 320, 400, 1000] ms

Evaluating action: Walking
Processing Walking: 100%|██████████| 100/100 [02:15<00:00,  1.35s/it]

Results for Walking:
  80ms: 45.23 mm
  160ms: 78.45 mm
  320ms: 125.67 mm
  400ms: 145.89 mm
  1000ms: 234.56 mm

=== SUMMARY ===
Average MPJPE across all actions:
  80ms: 52.34 mm
  160ms: 89.67 mm
  320ms: 142.89 mm
  400ms: 167.23 mm
  1000ms: 278.45 mm
```

### CSV结果文件
结果会保存到 `eval_results/h36m_mpjpe_results.csv`:

```csv
Action,80ms,160ms,320ms,400ms,1000ms
Directions,48.23,82.45,135.67,158.89,245.67
Discussion,51.34,87.56,140.23,165.78,267.89
Walking,45.23,78.45,125.67,145.89,234.56
...
Average,52.34,89.67,142.89,167.23,278.45
```

## 🔍 故障排除

### 常见问题

#### 1. 模型文件不存在
```
FileNotFoundError: Model checkpoint not found
```
**解决方案**: 检查模型路径
```bash
# 检查可用的模型文件
ls checkpoints/
# 使用正确的模型路径
python eval_h36m_mpjpe_fixed.py --model_path checkpoints/ckpt_ema_150.pt
```

#### 2. GPU内存不足
```
RuntimeError: CUDA out of memory
```
**解决方案**: 减少样本数或使用CPU
```bash
# 减少样本数
./run_h36m_evaluation.sh eval --num_samples 10
# 或使用CPU
./run_h36m_evaluation.sh eval --device cpu
```

#### 3. 导入错误
```
ImportError: No module named 'xxx'
```
**解决方案**: 确保在HumanMAC-main目录下运行
```bash
cd HumanMAC-main
python eval_h36m_mpjpe_fixed.py ...
```

### 调试步骤

1. **运行调试脚本**:
```bash
python eval_h36m_debug.py
```

2. **运行简单测试**:
```bash
python eval_h36m_simple.py --test_data
```

3. **检查数据加载**:
```bash
python eval_h36m_simple.py --quick_eval
```

## 📊 性能基准

### 典型MPJPE值范围（参考）
- **80ms**: 40-60 mm
- **160ms**: 70-100 mm
- **320ms**: 120-160 mm
- **400ms**: 140-180 mm
- **1000ms**: 200-300 mm

### 评估时间估算
- **单个动作**: 2-5分钟
- **所有动作**: 30-60分钟
- **高样本数(100)**: 时间翻倍

## 🎯 最佳实践

1. **首次使用**: 先运行调试脚本确保环境正常
2. **快速验证**: 使用少量样本(10-20)快速测试
3. **正式评估**: 使用50-100样本获得稳定结果
4. **对比实验**: 保持相同的num_samples进行公平对比
5. **结果保存**: 使用有意义的output_dir名称保存不同实验结果

## 🔗 相关资源

- [Human3.6M数据集](http://vision.imar.ro/human3.6m/)
- [HumanMAC论文](https://arxiv.org/abs/2302.03665)
- [MPJPE评估标准](https://github.com/una-dinosauria/human-motion-prediction)

---

**注意**: 确保在运行评估前已经训练好HumanMAC模型，并且Human3.6M数据集已正确下载和预处理。
