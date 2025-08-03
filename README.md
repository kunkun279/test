# Human Motion Prediction with PISL and HumanMAC

一个基于物理约束样条学习(PISL)和HumanMAC扩散模型的人体动作预测项目。

## 🎯 项目概述

本项目创新性地结合了：
- **PISL (Physics-Informed Spline Learning)**: 基于B样条的物理约束运动学方程学习
- **HumanMAC扩散模型**: 基于Transformer的条件扩散生成模型
- **自适应融合**: 智能融合两种预测方法，确保预测的准确性和物理合理性

## 📁 项目结构

```
human_motion_prediction/
├── models/                     # 核心模型实现
│   ├── pisl_spline.py         # PISL样条学习模块
│   ├── humanmac_diffusion.py  # HumanMAC扩散模型
│   └── fusion_model.py        # 融合预测模型
├── data/                      # 数据处理模块
│   └── preprocessing.py       # 数据预处理和加载
├── training/                  # 训练相关
│   └── train_fusion.py        # 融合模型训练脚本
├── evaluation/                # 评估模块
│   ├── metrics.py             # 评估指标计算
│   └── visualize.py           # 可视化工具
├── configs/                   # 配置文件
│   └── config.yaml            # 主配置文件
└── inference.py               # 推理脚本
examples/                      # 示例和演示
├── demo.py                    # 完整功能演示
└── simple_demo.py             # 简化演示
setup.py                       # 项目安装脚本
requirements.txt               # 依赖列表
```

## 🚀 快速开始

### 1. 环境准备

#### 选项A: 使用pip安装（推荐）

```bash
# 克隆项目（如果从git仓库）
git clone <repository_url>
cd human_motion_prediction

# 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或者 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 安装项目
pip install -e .
```

#### 选项B: 使用conda安装

```bash
# 创建conda环境
conda create -n motion_pred python=3.8
conda activate motion_pred

# 安装PyTorch（根据您的CUDA版本）
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 安装其他依赖
pip install -r requirements.txt
pip install -e .
```

### 2. 数据准备

#### Human3.6M数据集（推荐）

```bash
# 下载Human3.6M数据集
# 将数据放置在 data/human36m/ 目录下
mkdir -p data/human36m
# 复制您的.h5文件到此目录
```

#### 使用示例数据

```bash
# 运行简化演示（无需外部数据）
python3 examples/simple_demo.py

# 运行完整演示（使用生成的示例数据）
python examples/demo.py
```

### 3. 配置设置

编辑 `human_motion_prediction/configs/config.yaml` 文件：

```yaml
# 主要配置项
data_config:
  data_path: "data/human36m"      # 数据路径
  sequence_length: 50             # 历史序列长度
  prediction_length: 25           # 预测长度
  
model_config:
  pisl:
    degree: 3                     # B样条阶数
    n_control_points: 10          # 控制点数量
  diffusion:
    num_timesteps: 1000           # 扩散步数
    beta_schedule: "cosine"       # 噪声调度
    
training_config:
  batch_size: 32                  # 批次大小
  learning_rate: 0.001            # 学习率
  num_epochs: 100                 # 训练轮数
```

## 🔧 使用方法

### 训练模型

```bash
# 使用默认配置训练
python human_motion_prediction/training/train_fusion.py

# 指定配置文件训练
python human_motion_prediction/training/train_fusion.py --config configs/config.yaml

# 从检查点恢复训练
python human_motion_prediction/training/train_fusion.py --resume checkpoints/latest.pth

# 使用多GPU训练
python human_motion_prediction/training/train_fusion.py --multi_gpu
```

### 模型推理

```bash
# 单个序列预测
python human_motion_prediction/inference.py \
    --config configs/config.yaml \
    --model checkpoints/best_model.pth \
    --input data/test_sequence.npy \
    --output results/prediction.npy

# 批量预测
python human_motion_prediction/inference.py \
    --config configs/config.yaml \
    --model checkpoints/best_model.pth \
    --batch_mode \
    --input_dir data/test_sequences/ \
    --output_dir results/predictions/

# 不同预测模式
python human_motion_prediction/inference.py \
    --model checkpoints/best_model.pth \
    --mode pisl_only          # 仅使用PISL预测
    # --mode diffusion_only    # 仅使用扩散模型预测
    # --mode fusion           # 使用融合预测（默认）
```

### 评估模型

```bash
# 评估模型性能
python human_motion_prediction/inference.py \
    --config configs/config.yaml \
    --model checkpoints/best_model.pth \
    --evaluate \
    --test_data data/test_set.h5 \
    --save_metrics results/metrics.json

# 生成可视化结果
python human_motion_prediction/inference.py \
    --config configs/config.yaml \
    --model checkpoints/best_model.pth \
    --visualize \
    --output_dir results/visualizations/
```

## 📊 监控和日志

### TensorBoard监控

```bash
# 启动TensorBoard
tensorboard --logdir logs/tensorboard

# 在浏览器中访问 http://localhost:6006
```

### Weights & Biases监控

```bash
# 登录W&B（首次使用）
wandb login

# 训练时会自动记录到W&B
# 访问 https://wandb.ai 查看实验结果
```

## 🎨 可视化功能

### 生成3D动画

```python
from human_motion_prediction.evaluation.visualize import MotionVisualizer

visualizer = MotionVisualizer()

# 创建3D动画
visualizer.create_3d_animation(
    motion_data=predicted_motion,
    save_path="results/animation.gif",
    fps=30
)

# 比较预测结果
visualizer.compare_predictions(
    history=history_motion,
    ground_truth=gt_motion,
    predictions={
        'PISL': pisl_prediction,
        'Diffusion': diffusion_prediction,
        'Fusion': fusion_prediction
    },
    joint_idx=0,  # 根关节
    save_path="results/comparison.png"
)
```

### 生成误差热力图

```python
from human_motion_prediction.evaluation.metrics import MotionMetrics

metrics = MotionMetrics()
visualizer = MotionVisualizer()

# 计算误差
errors = metrics.compute_metrics(predictions, ground_truth)

# 生成热力图
visualizer.plot_error_heatmap(
    errors['joint_errors'],
    save_path="results/error_heatmap.png"
)
```

## 🔍 高级用法

### 自定义数据集

```python
from human_motion_prediction.data.preprocessing import MotionDataProcessor

# 创建数据处理器
processor = MotionDataProcessor(
    normalization='standard',  # 或 'minmax'
    augmentation=True
)

# 处理自定义数据
processed_data = processor.process_data(
    raw_motion_data,
    sequence_length=50,
    prediction_length=25
)
```

### 模型微调

```python
from human_motion_prediction.models.fusion_model import PISLHumanMACFusion

# 加载预训练模型
model = PISLHumanMACFusion.load_from_checkpoint("checkpoints/pretrained.pth")

# 冻结部分参数
for param in model.pisl_model.parameters():
    param.requires_grad = False

# 微调扩散模型部分
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)
```

### 自定义物理约束

```python
from human_motion_prediction.models.pisl_spline import PhysicsInformedSpline

# 自定义关节限制
joint_limits = {
    'shoulder': {'min': -180, 'max': 180},
    'elbow': {'min': 0, 'max': 150},
    # ... 其他关节
}

# 自定义速度限制
velocity_limits = {
    'max_linear_velocity': 2.0,  # m/s
    'max_angular_velocity': 5.0  # rad/s
}

pisl_model = PhysicsInformedSpline(
    joint_limits=joint_limits,
    velocity_limits=velocity_limits
)
```

## 📈 性能优化

### 内存优化

```bash
# 使用梯度累积减少内存使用
python human_motion_prediction/training/train_fusion.py \
    --gradient_accumulation_steps 4 \
    --batch_size 8

# 使用混合精度训练
python human_motion_prediction/training/train_fusion.py \
    --mixed_precision
```

### 推理加速

```bash
# 使用DDIM快速采样
python human_motion_prediction/inference.py \
    --sampling_method ddim \
    --ddim_steps 50  # 而不是1000步

# 批量推理
python human_motion_prediction/inference.py \
    --batch_size 64 \
    --batch_mode
```

## 🐛 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减少批次大小
   export CUDA_VISIBLE_DEVICES=0
   python train_fusion.py --batch_size 16
   ```

2. **依赖版本冲突**
   ```bash
   # 使用conda解决依赖
   conda env create -f environment.yml
   ```

3. **数据加载错误**
   ```bash
   # 检查数据路径和格式
   python -c "from human_motion_prediction.data.preprocessing import MotionDataLoader; loader = MotionDataLoader('data/human36m'); print(loader.check_data())"
   ```

### 调试模式

```bash
# 启用调试模式
python human_motion_prediction/training/train_fusion.py --debug

# 详细日志
python human_motion_prediction/training/train_fusion.py --log_level DEBUG
```

## 📚 API文档

### 核心类说明

- **`PhysicsInformedSpline`**: PISL样条学习模型
- **`HumanMACDiffusion`**: HumanMAC扩散模型
- **`PISLHumanMACFusion`**: 融合预测模型
- **`MotionDataProcessor`**: 数据预处理工具
- **`MotionMetrics`**: 评估指标计算
- **`MotionVisualizer`**: 可视化工具

详细API文档请参考各模块的docstring。

## 🤝 贡献指南

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- Human3.6M数据集提供者
- PyTorch团队
- 相关研究论文的作者们

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 创建Issue
- 发送邮件到 [your-email@example.com]

---

**Happy Motion Predicting! 🚀**
