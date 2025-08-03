# 基于PISL样条学习和HumanMAC扩散模型的人体动作预测

本项目结合了PISL（Physics-Informed Spline Learning）中的样条学习方法和HumanMAC扩散模型，用于预测人体未来动作。该项目实现了一个完整的人体动作预测系统，包含数据处理、模型训练、评估和可视化等功能。

## 🎯 核心思路

1. **运动学建模**: 利用PISL中的样条学习方法，从人体运动历史数据中推导出每个关节点的运动轨迹运动学方程
2. **扩散模型预测**: 将运动学方程嵌入到HumanMAC扩散模型中，进行未来人体动作预测
3. **物理约束**: 结合物理约束确保预测动作的合理性和连续性
4. **多模态融合**: 融合PISL和扩散模型的预测结果，提高预测精度

## 📁 项目结构

```
human_motion_prediction/
├── models/                     # 模型实现
│   ├── pisl_spline.py         # PISL样条学习模块
│   ├── humanmac_diffusion.py  # HumanMAC扩散模型
│   └── fusion_model.py        # 融合模型
├── data/                       # 数据处理
│   └── preprocessing.py        # 数据预处理和加载
├── training/                   # 训练模块
│   └── train_fusion.py        # 融合模型训练脚本
├── evaluation/                 # 评估工具
│   ├── metrics.py             # 评估指标
│   └── visualize.py           # 可视化工具
├── configs/                    # 配置文件
│   └── config.yaml            # 主配置文件
├── inference.py               # 推理脚本
├── requirements.txt           # 依赖包
└── README.md                  # 项目说明
```

## 🚀 快速开始

### 1. 环境安装

```bash
# 克隆项目
git clone <repository_url>
cd human_motion_prediction

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

项目支持Human3.6M数据集格式。将数据放在`./data`目录下：

```
data/
├── S1.h5
├── S5.h5
├── S6.h5
└── ...
```

### 3. 模型训练

使用默认配置训练融合模型：

```bash
python human_motion_prediction/training/train_fusion.py --config human_motion_prediction/configs/config.yaml
```

自定义训练参数：

```bash
python human_motion_prediction/training/train_fusion.py \
    --config human_motion_prediction/configs/config.yaml \
    --batch_size 32 \
    --learning_rate 2e-4 \
    --num_epochs 200 \
    --use_wandb
```

### 4. 模型推理

使用训练好的模型进行预测：

```bash
# 单个预测
python human_motion_prediction/inference.py \
    --config human_motion_prediction/configs/config.yaml \
    --model checkpoints/best_model.pth \
    --input motion_data.npy \
    --output_dir results

# 批量预测
python human_motion_prediction/inference.py \
    --config human_motion_prediction/configs/config.yaml \
    --model checkpoints/best_model.pth \
    --batch_mode \
    --output_dir batch_results

# 仅使用PISL预测
python human_motion_prediction/inference.py \
    --config human_motion_prediction/configs/config.yaml \
    --model checkpoints/best_model.pth \
    --use_pisl_only

# 仅使用扩散模型预测
python human_motion_prediction/inference.py \
    --config human_motion_prediction/configs/config.yaml \
    --model checkpoints/best_model.pth \
    --use_diffusion_only
```

## 🔧 配置说明

主要配置文件为`configs/config.yaml`，包含以下配置项：

- **model_config**: 模型架构参数
- **training_config**: 训练参数
- **data_config**: 数据处理参数
- **experiment_config**: 实验设置
- **evaluation_config**: 评估配置

详细配置说明请参考配置文件中的注释。

## 📊 评估指标

项目支持多种评估指标：

- **MPJPE**: 平均关节位置误差
- **P-MPJPE**: Procrustes对齐的MPJPE
- **速度误差**: 关节速度预测误差
- **加速度误差**: 关节加速度预测误差
- **物理合理性**: 速度和加速度约束违反率
- **骨骼一致性**: 骨骼长度变化的一致性
- **运动平滑度**: 基于Jerk的平滑度评估

## 🎨 可视化功能

项目提供丰富的可视化功能：

1. **3D人体骨架可视化**: 显示预测的3D人体姿态
2. **运动轨迹对比**: 比较不同方法的预测轨迹
3. **误差热力图**: 显示各关节的预测误差
4. **指标对比图**: 可视化不同方法的评估指标
5. **交互式3D动画**: 基于Plotly的交互式可视化
6. **预测对比视频**: 生成动态对比视频

## 🏗️ 模型架构

### PISL样条学习模块
- 使用B样条基函数建模关节轨迹
- 结合物理约束（速度、加速度限制）
- 优化样条控制点以拟合历史数据

### HumanMAC扩散模型
- 基于Transformer的编码器-解码器架构
- 支持DDPM和DDIM采样
- 条件扩散生成，以历史运动为条件

### 融合模型
- 自适应权重学习
- 多层次特征融合
- 物理约束验证

## 📈 性能特点

- **高精度**: 融合多种预测方法，提高预测精度
- **物理合理性**: 内置物理约束，确保预测动作合理
- **实时性**: 支持DDIM快速采样，提高推理速度
- **可扩展性**: 模块化设计，易于扩展和修改
- **可视化**: 丰富的可视化工具，便于分析和展示

## 🛠️ 开发指南

### 添加新的评估指标

在`evaluation/metrics.py`中添加新的指标计算函数：

```python
def compute_new_metric(self, predictions, targets):
    # 实现新指标的计算
    return metric_value
```

### 自定义可视化

在`evaluation/visualize.py`中添加新的可视化函数：

```python
def plot_custom_visualization(self, data, **kwargs):
    # 实现自定义可视化
    return figure
```

### 修改模型架构

可以通过修改配置文件或直接修改模型代码来调整架构：

- 修改`models/pisl_spline.py`调整样条学习参数
- 修改`models/humanmac_diffusion.py`调整扩散模型架构
- 修改`models/fusion_model.py`调整融合策略

## 📚 参考文献

1. PISL: Physics-Informed Spline Learning
2. HumanMAC: Masked Motion Completion for Human Motion Prediction
3. Human3.6M Dataset
4. Denoising Diffusion Probabilistic Models

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进项目。请确保：

1. 代码风格符合项目规范
2. 添加适当的测试和文档
3. 更新相关的配置文件

## 📄 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 提交GitHub Issue
- 发送邮件至：[your-email@example.com]

---

**注意**: 本项目为研究和学习目的，请遵循相关数据集的使用协议。