"""
人体动作预测演示脚本
展示PISL样条学习和HumanMAC扩散模型融合预测的完整流程
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from human_motion_prediction.models.pisl_spline import PhysicsInformedSpline
from human_motion_prediction.models.humanmac_diffusion import HumanMACDiffusion
from human_motion_prediction.models.fusion_model import PISLHumanMACFusion
from human_motion_prediction.data.preprocessing import MotionDataProcessor
from human_motion_prediction.evaluation.metrics import MotionMetrics
from human_motion_prediction.evaluation.visualize import MotionVisualizer


def generate_sample_motion_data(seq_len: int = 50, n_joints: int = 22) -> np.ndarray:
    """
    生成示例人体运动数据
    
    Args:
        seq_len: 序列长度
        n_joints: 关节数量
        
    Returns:
        motion_data: [seq_len, n_joints, 3] 运动数据
    """
    print("生成示例人体运动数据...")
    
    t = np.linspace(0, 2, seq_len)
    motion_data = np.zeros((seq_len, n_joints, 3))
    
    # 为每个关节生成不同频率的正弦运动
    for joint in range(n_joints):
        for dim in range(3):
            # 不同关节和维度使用不同的频率和幅度
            frequency = 0.5 + joint * 0.05 + dim * 0.1
            amplitude = 0.3 + joint * 0.02
            phase = joint * 0.1 + dim * 0.2
            
            motion_data[:, joint, dim] = amplitude * np.sin(2 * np.pi * frequency * t + phase) + \
                                       0.05 * np.random.randn(seq_len)
    
    print(f"生成的运动数据形状: {motion_data.shape}")
    return motion_data


def demo_pisl_spline_learning():
    """演示PISL样条学习"""
    print("\n" + "="*60)
    print("演示1: PISL样条学习")
    print("="*60)
    
    # 生成示例数据
    motion_data = generate_sample_motion_data(seq_len=30)
    
    # 创建PISL模型
    pisl_model = PhysicsInformedSpline(
        n_joints=22,
        spline_degree=3,
        n_control_points=8,
        physics_weight=1.0,
        smoothness_weight=0.1
    )
    
    # 拟合运动数据
    print("拟合运动轨迹到样条表示...")
    time_points = torch.linspace(0, 1, 30)
    results = pisl_model.fit_trajectory(
        torch.FloatTensor(motion_data), 
        time_points, 
        n_iterations=200
    )
    
    print(f"拟合完成，最终损失: {results['losses'][-1]['total']:.6f}")
    
    # 预测未来运动
    print("预测未来运动...")
    future_time = torch.linspace(1, 1.5, 15)
    future_motion = pisl_model.predict_future(
        torch.FloatTensor(motion_data[-1]), 
        future_time
    )
    
    print(f"预测的未来运动形状: {future_motion.shape}")
    
    # 获取运动学参数
    kinematic_params = pisl_model.get_kinematic_equations()
    print(f"学习到的样条控制点形状: {kinematic_params['control_points'].shape}")
    
    return motion_data, future_motion.numpy(), kinematic_params


def demo_humanmac_diffusion():
    """演示HumanMAC扩散模型"""
    print("\n" + "="*60)
    print("演示2: HumanMAC扩散模型")
    print("="*60)
    
    # 创建扩散模型
    diffusion_model = HumanMACDiffusion(
        input_dim=66,
        d_model=256,  # 为了演示使用较小的模型
        n_encoder_layers=3,
        n_decoder_layers=3,
        n_heads=4,
        n_timesteps=100  # 使用较少的时间步
    )
    
    print(f"扩散模型参数数量: {sum(p.numel() for p in diffusion_model.parameters()):,}")
    
    # 生成示例数据
    batch_size = 2
    condition_len = 25
    pred_len = 25
    
    condition = torch.randn(batch_size, condition_len, 66)
    target = torch.randn(batch_size, pred_len, 66)
    
    # 计算训练损失
    print("计算扩散损失...")
    loss = diffusion_model.compute_loss(target, condition)
    print(f"扩散损失: {loss.item():.6f}")
    
    # 生成样本（使用DDIM快速采样）
    print("使用DDIM采样生成运动...")
    with torch.no_grad():
        samples = diffusion_model.ddim_sample(
            shape=(batch_size, pred_len, 66),
            condition=condition,
            n_steps=20,  # 使用较少的采样步数
            eta=0.0
        )
    
    print(f"生成样本形状: {samples.shape}")
    
    return condition.numpy(), samples.numpy()


def demo_fusion_model():
    """演示融合模型"""
    print("\n" + "="*60)
    print("演示3: PISL-HumanMAC融合模型")
    print("="*60)
    
    # 创建融合模型
    fusion_model = PISLHumanMACFusion(
        n_joints=22,
        input_dim=66,
        d_model=256,  # 使用较小的模型进行演示
        n_encoder_layers=3,
        n_decoder_layers=3,
        n_heads=4,
        n_timesteps=100,
        fusion_weight=0.5
    )
    
    print(f"融合模型参数数量: {sum(p.numel() for p in fusion_model.parameters()):,}")
    
    # 生成示例数据
    batch_size = 2
    condition_len = 25
    target_len = 25
    
    condition = torch.randn(batch_size, condition_len, 66)
    target = torch.randn(batch_size, target_len, 66)
    
    # 计算损失
    print("计算融合模型损失...")
    losses = fusion_model.compute_loss(condition, target)
    
    print("损失组成:")
    for loss_name, loss_value in losses.items():
        print(f"  {loss_name}: {loss_value.item():.6f}")
    
    # 进行预测
    print("进行融合预测...")
    with torch.no_grad():
        # 获取所有预测结果
        predictions = fusion_model.forward(condition, target_len)
        
        print("预测结果:")
        for pred_name, pred_tensor in predictions.items():
            if isinstance(pred_tensor, torch.Tensor):
                print(f"  {pred_name}: {pred_tensor.shape}")
    
    # 分析预测质量
    print("分析预测质量...")
    analysis = fusion_model.analyze_predictions(condition, target_len)
    
    print("物理合理性分数:")
    for method, score in analysis['physics_scores'].items():
        print(f"  {method}: {score.item():.4f}")
    
    print("平滑度分数:")
    for method, score in analysis['smoothness_scores'].items():
        print(f"  {method}: {score.item():.4f}")
    
    return condition.numpy(), predictions, analysis


def demo_data_processing():
    """演示数据处理"""
    print("\n" + "="*60)
    print("演示4: 数据处理和评估")
    print("="*60)
    
    # 生成示例数据
    motion_data = generate_sample_motion_data(seq_len=100)
    
    # 创建数据处理器
    processor = MotionDataProcessor(
        n_joints=22,
        fps=30,
        normalize_method="standard",
        augmentation_config={
            'add_noise': True,
            'noise_std': 0.01,
            'rotation': True,
            'rotation_angles': [-5, 5],
            'mirror': True
        }
    )
    
    # 数据标准化
    print("数据标准化...")
    normalized_data = processor.fit_transform(motion_data)
    print(f"标准化后数据形状: {normalized_data.shape}")
    
    # 创建序列
    print("创建训练序列...")
    history_seq, future_seq = processor.create_sequences(
        normalized_data, 
        history_length=25, 
        future_length=25, 
        stride=5
    )
    print(f"历史序列形状: {history_seq.shape}")
    print(f"未来序列形状: {future_seq.shape}")
    
    # 数据增强
    print("数据增强...")
    sample_motion = motion_data[:30]
    augmented_motions = processor.augment_data(sample_motion)
    print(f"增强后数据数量: {len(augmented_motions)}")
    
    # 质量验证
    print("运动质量验证...")
    quality_metrics = processor.validate_motion(sample_motion)
    print("质量指标:")
    for metric, value in quality_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return history_seq, future_seq


def demo_evaluation_and_visualization():
    """演示评估和可视化"""
    print("\n" + "="*60)
    print("演示5: 评估和可视化")
    print("="*60)
    
    # 生成示例预测数据
    batch_size = 4
    seq_len = 25
    n_joints = 22
    
    # 真实数据
    ground_truth = np.random.randn(batch_size, seq_len, n_joints, 3) * 0.1
    
    # 模拟不同方法的预测结果
    predictions = {
        'pisl': ground_truth + 0.05 * np.random.randn(*ground_truth.shape),
        'diffusion': ground_truth + 0.03 * np.random.randn(*ground_truth.shape),
        'fusion': ground_truth + 0.02 * np.random.randn(*ground_truth.shape)
    }
    
    # 创建评估器
    metrics_calculator = MotionMetrics(n_joints=22, fps=30)
    
    # 计算评估指标
    print("计算评估指标...")
    all_metrics = {}
    
    for method_name, pred in predictions.items():
        pred_flat = pred.reshape(batch_size, seq_len, -1)
        gt_flat = ground_truth.reshape(batch_size, seq_len, -1)
        
        metrics = metrics_calculator.compute_metrics(pred_flat, gt_flat)
        all_metrics[method_name] = metrics
        
        print(f"\n{method_name.upper()} 方法指标:")
        print(f"  MPJPE: {metrics['mpjpe']:.2f} mm")
        print(f"  P-MPJPE: {metrics['p_mpjpe']:.2f} mm")
        print(f"  速度误差: {metrics['velocity_error']:.2f} mm/s")
        print(f"  物理合理性: {metrics['velocity_violation_rate']:.4f}")
    
    # 创建可视化器
    visualizer = MotionVisualizer(n_joints=22, fps=30)
    
    # 可视化示例
    print("\n生成可视化...")
    
    # 1. 3D骨架可视化
    sample_motion = ground_truth[0]
    fig_skeleton = visualizer.plot_3d_skeleton(sample_motion, "示例3D骨架")
    plt.savefig("demo_skeleton.png", dpi=150, bbox_inches='tight')
    print("3D骨架图已保存为 demo_skeleton.png")
    
    # 2. 指标比较
    simplified_metrics = {}
    for method, metrics in all_metrics.items():
        simplified_metrics[method] = {
            'mpjpe': metrics['mpjpe'],
            'p_mpjpe': metrics['p_mpjpe'],
            'velocity_error': metrics['velocity_error'],
            'acceleration_error': metrics['acceleration_error'],
            'bone_consistency': metrics['bone_consistency'],
            'smoothness': metrics['smoothness']
        }
    
    fig_metrics = visualizer.plot_metrics_comparison(simplified_metrics)
    plt.savefig("demo_metrics.png", dpi=150, bbox_inches='tight')
    print("指标比较图已保存为 demo_metrics.png")
    
    # 3. 误差热力图
    errors = np.random.rand(seq_len, n_joints) * 50  # 模拟误差数据
    fig_heatmap = visualizer.plot_error_heatmap(errors, "关节位置误差热力图")
    plt.savefig("demo_heatmap.png", dpi=150, bbox_inches='tight')
    print("误差热力图已保存为 demo_heatmap.png")
    
    plt.close('all')  # 关闭所有图形
    
    return all_metrics


def main():
    """主演示函数"""
    print("基于PISL样条学习和HumanMAC扩散模型的人体动作预测演示")
    print("="*80)
    
    try:
        # 演示1: PISL样条学习
        history_motion, pisl_prediction, kinematic_params = demo_pisl_spline_learning()
        
        # 演示2: HumanMAC扩散模型
        condition_data, diffusion_prediction = demo_humanmac_diffusion()
        
        # 演示3: 融合模型
        fusion_condition, fusion_predictions, fusion_analysis = demo_fusion_model()
        
        # 演示4: 数据处理
        history_seq, future_seq = demo_data_processing()
        
        # 演示5: 评估和可视化
        evaluation_metrics = demo_evaluation_and_visualization()
        
        print("\n" + "="*80)
        print("演示完成！")
        print("="*80)
        print("\n生成的文件:")
        print("- demo_skeleton.png: 3D人体骨架可视化")
        print("- demo_metrics.png: 评估指标比较")
        print("- demo_heatmap.png: 关节误差热力图")
        print("\n所有演示模块都运行成功！")
        
    except Exception as e:
        print(f"\n演示过程中出现错误: {str(e)}")
        print("这可能是由于缺少依赖包或其他环境问题导致的。")
        print("请检查是否已正确安装所有依赖包。")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()