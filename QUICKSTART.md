# 🚀 快速入门指南

这是一个5分钟快速上手指南，帮助您立即开始使用人体动作预测项目。

## ⚡ 超快速开始（无需依赖）

如果您只想快速了解项目结构和功能：

```bash
# 直接运行简化演示
python3 examples/simple_demo.py
```

这将展示项目的完整架构、核心概念和功能介绍，无需安装任何外部依赖。

## 🔧 完整功能体验

### 1. 一键安装依赖

```bash
# 安装所有必需的包
pip install torch torchvision numpy scipy matplotlib scikit-learn tqdm
```

### 2. 运行完整演示

```bash
# 运行功能完整的演示
python examples/demo.py
```

这将：
- 生成示例运动数据
- 演示PISL样条学习
- 展示HumanMAC扩散模型
- 运行融合预测
- 生成可视化结果

## 📊 主要输出

运行演示后，您将看到：

1. **PISL学习结果**：
   ```
   PISL Spline Learning Demo
   ========================
   Learning spline for joint 0 (root)...
   Physics loss: 0.0234
   Smoothness loss: 0.0156
   ```

2. **扩散模型训练**：
   ```
   HumanMAC Diffusion Demo
   =======================
   Training diffusion model...
   Epoch 1/5, Loss: 0.8765
   ```

3. **融合预测结果**：
   ```
   Fusion Model Demo
   =================
   PISL Prediction MPJPE: 45.23mm
   Diffusion Prediction MPJPE: 52.18mm
   Fusion Prediction MPJPE: 38.91mm
   ```

4. **可视化文件**：
   - `results/pisl_trajectory.png` - PISL轨迹预测
   - `results/diffusion_samples.png` - 扩散模型生成结果
   - `results/fusion_comparison.png` - 融合预测对比
   - `results/error_heatmap.png` - 误差热力图

## 🎯 核心概念速览

### PISL (Physics-Informed Spline Learning)
```python
# 物理约束的B样条学习
pisl_model = PhysicsInformedSpline(
    degree=3,                    # B样条阶数
    n_control_points=10,         # 控制点数量
    max_velocity=2.0,           # 最大速度约束
    max_acceleration=5.0        # 最大加速度约束
)
```

### HumanMAC扩散模型
```python
# 基于Transformer的扩散模型
diffusion_model = HumanMACDiffusion(
    input_dim=66,               # 输入维度（22关节×3坐标）
    hidden_dim=512,             # 隐藏层维度
    num_layers=8,               # Transformer层数
    num_timesteps=1000          # 扩散步数
)
```

### 融合预测
```python
# 自适应融合两种预测方法
fusion_model = PISLHumanMACFusion(
    pisl_config=pisl_config,
    diffusion_config=diffusion_config
)

# 进行预测
prediction = fusion_model.predict(
    condition=history_motion,    # 历史动作
    target_length=25            # 预测长度
)
```

## 📈 评估指标

项目提供全面的评估指标：

- **MPJPE**: 平均关节位置误差
- **P-MPJPE**: Procrustes对齐后的MPJPE
- **速度误差**: 关节速度预测准确性
- **加速度误差**: 关节加速度预测准确性
- **物理合理性**: 违反物理约束的比例
- **运动平滑性**: 基于jerk的平滑度评估

## 🎨 可视化功能

### 3D动画生成
```python
visualizer.create_3d_animation(
    motion_data=prediction,
    save_path="animation.gif"
)
```

### 预测对比
```python
visualizer.compare_predictions(
    ground_truth=gt_motion,
    predictions={
        'PISL': pisl_pred,
        'Diffusion': diff_pred,
        'Fusion': fusion_pred
    }
)
```

## 🔄 下一步

1. **使用真实数据**：
   ```bash
   # 下载Human3.6M数据集
   # 放置在 data/human36m/ 目录
   ```

2. **训练自己的模型**：
   ```bash
   python human_motion_prediction/training/train_fusion.py
   ```

3. **自定义配置**：
   编辑 `human_motion_prediction/configs/config.yaml`

4. **进行推理**：
   ```bash
   python human_motion_prediction/inference.py --model checkpoints/best_model.pth
   ```

## 🆘 遇到问题？

### 常见解决方案

1. **Python命令不存在**：
   ```bash
   # 使用python3替代python
   python3 examples/demo.py
   ```

2. **依赖缺失**：
   ```bash
   # 安装特定包
   pip install numpy torch matplotlib
   ```

3. **权限问题**：
   ```bash
   # 使用用户安装
   pip install --user torch numpy matplotlib
   ```

4. **内存不足**：
   ```bash
   # 减少批次大小
   export BATCH_SIZE=8
   ```

## 📞 获取帮助

- 查看完整文档：`README.md`
- 检查配置文件：`human_motion_prediction/configs/config.yaml`
- 运行简化演示：`python3 examples/simple_demo.py`

---

🎉 **恭喜！您已经完成了快速入门。现在可以开始探索更多高级功能了！**