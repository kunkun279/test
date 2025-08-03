"""
人体动作预测推理脚本
基于PISL样条学习和HumanMAC扩散模型的融合预测
"""

import os
import sys
import torch
import numpy as np
import argparse
import yaml
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.fusion_model import PISLHumanMACFusion
from data.preprocessing import MotionDataProcessor
from evaluation.metrics import MotionMetrics
from evaluation.visualize import MotionVisualizer


class MotionPredictor:
    """人体动作预测器"""
    
    def __init__(self, config_path: str, model_path: str):
        """
        初始化预测器
        
        Args:
            config_path: 配置文件路径
            model_path: 模型权重路径
        """
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device(self.config['experiment_config']['device'])
        
        # 初始化模型
        self.model = self._load_model(model_path)
        
        # 初始化数据处理器
        self.processor = MotionDataProcessor(
            n_joints=self.config['model_config']['n_joints'],
            normalize_method=self.config['data_config']['normalize_method']
        )
        
        # 初始化评估器和可视化器
        self.metrics = MotionMetrics(
            n_joints=self.config['model_config']['n_joints'],
            fps=self.config['data_config']['fps']
        )
        
        self.visualizer = MotionVisualizer(
            n_joints=self.config['model_config']['n_joints'],
            fps=self.config['data_config']['fps']
        )
        
        print(f"模型加载完成，设备: {self.device}")
    
    def _load_model(self, model_path: str) -> PISLHumanMACFusion:
        """加载模型"""
        # 创建模型
        model = PISLHumanMACFusion(**self.config['model_config'])
        
        # 加载权重
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 处理不同的保存格式
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"从检查点加载模型: epoch {checkpoint.get('epoch', 'unknown')}")
            else:
                model.load_state_dict(checkpoint)
                print("加载模型权重")
        else:
            print(f"警告: 模型文件不存在 {model_path}，使用随机初始化权重")
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def predict_motion(self, 
                      history_motion: np.ndarray,
                      future_length: int = None,
                      use_pisl_only: bool = False,
                      use_diffusion_only: bool = False) -> Dict[str, np.ndarray]:
        """
        预测未来运动
        
        Args:
            history_motion: [seq_len, n_joints, 3] 或 [seq_len, n_joints * 3] 历史运动
            future_length: 预测长度，默认使用配置中的值
            use_pisl_only: 仅使用PISL预测
            use_diffusion_only: 仅使用扩散模型预测
            
        Returns:
            predictions: 预测结果字典
        """
        if future_length is None:
            future_length = self.config['training_config']['future_length']
        
        # 数据预处理
        if history_motion.ndim == 3:  # [seq_len, n_joints, 3]
            history_motion = history_motion.reshape(history_motion.shape[0], -1)
        
        # 标准化
        if not self.processor.is_fitted:
            # 如果处理器未拟合，使用历史数据拟合
            self.processor.fit(history_motion[np.newaxis, ...])
        
        normalized_history = self.processor.transform(history_motion[np.newaxis, ...])
        
        # 转换为张量
        history_tensor = torch.FloatTensor(normalized_history).to(self.device)
        
        # 预测
        with torch.no_grad():
            if use_pisl_only:
                predictions = self.model.predict(
                    history_tensor, future_length, 
                    use_pisl_only=True
                )
                pred_dict = {'pisl_prediction': predictions}
            elif use_diffusion_only:
                predictions = self.model.predict(
                    history_tensor, future_length,
                    use_diffusion_only=True
                )
                pred_dict = {'diffusion_prediction': predictions}
            else:
                # 获取所有预测结果
                all_predictions = self.model.forward(history_tensor, future_length)
                
                pred_dict = {
                    'pisl_prediction': all_predictions['pisl_predictions'],
                    'diffusion_prediction': all_predictions['diffusion_predictions'],
                    'fusion_prediction': all_predictions['fused_predictions']
                }
        
        # 转换回原始尺度并转为numpy
        results = {}
        for name, pred in pred_dict.items():
            pred_np = pred.cpu().numpy()
            # 逆标准化
            pred_denorm = self.processor.inverse_transform(pred_np)
            results[name] = pred_denorm[0]  # 移除batch维度
        
        return results
    
    def evaluate_predictions(self, 
                           predictions: Dict[str, np.ndarray],
                           ground_truth: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        评估预测结果
        
        Args:
            predictions: 预测结果字典
            ground_truth: [seq_len, n_joints, 3] 真实运动
            
        Returns:
            metrics: 各种方法的评估指标
        """
        if ground_truth.ndim == 3:
            ground_truth = ground_truth.reshape(ground_truth.shape[0], -1)
        
        results = {}
        
        for method_name, pred in predictions.items():
            if pred.ndim == 3:
                pred = pred.reshape(pred.shape[0], -1)
            
            # 添加batch维度
            pred_batch = pred[np.newaxis, ...]
            gt_batch = ground_truth[np.newaxis, ...]
            
            # 计算指标
            metrics = self.metrics.compute_metrics(pred_batch, gt_batch)
            results[method_name] = metrics
        
        return results
    
    def visualize_predictions(self,
                            history: np.ndarray,
                            predictions: Dict[str, np.ndarray],
                            ground_truth: Optional[np.ndarray] = None,
                            save_dir: Optional[str] = None) -> Dict[str, plt.Figure]:
        """
        可视化预测结果
        
        Args:
            history: 历史运动
            predictions: 预测结果
            ground_truth: 真实未来运动（可选）
            save_dir: 保存目录
            
        Returns:
            figures: 生成的图形字典
        """
        figures = {}
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # 确保数据格式正确
        if history.ndim == 2:
            history = history.reshape(history.shape[0], self.config['model_config']['n_joints'], 3)
        
        pred_3d = {}
        for name, pred in predictions.items():
            if pred.ndim == 2:
                pred_3d[name] = pred.reshape(pred.shape[0], self.config['model_config']['n_joints'], 3)
            else:
                pred_3d[name] = pred
        
        if ground_truth is not None:
            if ground_truth.ndim == 2:
                ground_truth = ground_truth.reshape(ground_truth.shape[0], self.config['model_config']['n_joints'], 3)
        
        # 1. 轨迹比较图
        if ground_truth is not None:
            fig_comparison = self.visualizer.compare_predictions(
                history, ground_truth, pred_3d, joint_idx=0
            )
            figures['trajectory_comparison'] = fig_comparison
            
            if save_dir:
                fig_comparison.savefig(os.path.join(save_dir, 'trajectory_comparison.png'), 
                                     dpi=300, bbox_inches='tight')
        
        # 2. 3D骨架可视化
        for name, pred in pred_3d.items():
            # 组合历史和预测
            full_motion = np.concatenate([history, pred], axis=0)
            
            fig_skeleton = self.visualizer.plot_3d_skeleton(
                full_motion, title=f'{name.replace("_", " ").title()} - 3D Skeleton'
            )
            figures[f'{name}_skeleton'] = fig_skeleton
            
            if save_dir:
                fig_skeleton.savefig(os.path.join(save_dir, f'{name}_skeleton.png'),
                                   dpi=300, bbox_inches='tight')
        
        # 3. 关节轨迹3D图
        for joint_idx in [0, 7, 10]:  # Hip, LeftForeArm, RightForeArm
            fig_traj = self.visualizer.plot_trajectory_3d(
                np.concatenate([history, list(pred_3d.values())[0]], axis=0),
                joint_idx=joint_idx
            )
            figures[f'trajectory_joint_{joint_idx}'] = fig_traj
            
            if save_dir:
                fig_traj.savefig(os.path.join(save_dir, f'trajectory_joint_{joint_idx}.png'),
                                dpi=300, bbox_inches='tight')
        
        return figures
    
    def create_prediction_video(self,
                              history: np.ndarray,
                              predictions: Dict[str, np.ndarray],
                              ground_truth: Optional[np.ndarray] = None,
                              save_path: str = "prediction_video.mp4"):
        """
        创建预测对比视频
        
        Args:
            history: 历史运动
            predictions: 预测结果
            ground_truth: 真实未来运动
            save_path: 视频保存路径
        """
        if ground_truth is not None:
            self.visualizer.save_comparison_video(
                history, ground_truth, predictions, save_path
            )
            print(f"预测对比视频已保存到: {save_path}")
        else:
            print("需要真实数据才能创建对比视频")
    
    def batch_predict(self, 
                     data_dir: str,
                     output_dir: str,
                     max_samples: int = 10):
        """
        批量预测
        
        Args:
            data_dir: 数据目录
            output_dir: 输出目录
            max_samples: 最大样本数
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 这里应该实现批量数据加载逻辑
        # 为了示例，我们生成一些模拟数据
        print(f"开始批量预测，输出目录: {output_dir}")
        
        for i in range(min(max_samples, 5)):  # 限制示例数量
            # 生成模拟历史数据
            seq_len = self.config['training_config']['history_length']
            n_joints = self.config['model_config']['n_joints']
            
            # 模拟运动数据
            t = np.linspace(0, 1, seq_len)
            history = np.zeros((seq_len, n_joints, 3))
            
            for joint in range(n_joints):
                for dim in range(3):
                    frequency = 0.5 + joint * 0.1 + dim * 0.2
                    history[:, joint, dim] = np.sin(2 * np.pi * frequency * t) + \
                                           0.1 * np.random.randn(seq_len)
            
            # 预测
            predictions = self.predict_motion(history)
            
            # 可视化
            sample_dir = os.path.join(output_dir, f'sample_{i+1}')
            figures = self.visualize_predictions(history, predictions, save_dir=sample_dir)
            
            # 保存预测结果
            for name, pred in predictions.items():
                np.save(os.path.join(sample_dir, f'{name}.npy'), pred)
            
            print(f"样本 {i+1} 处理完成")
        
        print("批量预测完成!")


def main():
    parser = argparse.ArgumentParser(description='人体动作预测推理')
    parser.add_argument('--config', type=str, 
                       default='human_motion_prediction/configs/config.yaml',
                       help='配置文件路径')
    parser.add_argument('--model', type=str,
                       default='checkpoints/best_model.pth',
                       help='模型权重路径')
    parser.add_argument('--input', type=str, default=None,
                       help='输入运动数据文件(.npy)')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='输出目录')
    parser.add_argument('--future_length', type=int, default=25,
                       help='预测长度')
    parser.add_argument('--use_pisl_only', action='store_true',
                       help='仅使用PISL预测')
    parser.add_argument('--use_diffusion_only', action='store_true',
                       help='仅使用扩散模型预测')
    parser.add_argument('--create_video', action='store_true',
                       help='创建预测视频')
    parser.add_argument('--batch_mode', action='store_true',
                       help='批量预测模式')
    
    args = parser.parse_args()
    
    # 创建预测器
    predictor = MotionPredictor(args.config, args.model)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f'prediction_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    if args.batch_mode:
        # 批量预测模式
        predictor.batch_predict(
            data_dir='./data',
            output_dir=output_dir,
            max_samples=10
        )
    else:
        # 单个预测模式
        if args.input and os.path.exists(args.input):
            # 从文件加载数据
            history_motion = np.load(args.input)
            print(f"从文件加载历史运动数据: {history_motion.shape}")
        else:
            # 生成示例数据
            print("生成示例历史运动数据...")
            seq_len = 25
            n_joints = 22
            
            t = np.linspace(0, 1, seq_len)
            history_motion = np.zeros((seq_len, n_joints, 3))
            
            for joint in range(n_joints):
                for dim in range(3):
                    frequency = 0.5 + joint * 0.1 + dim * 0.2
                    history_motion[:, joint, dim] = np.sin(2 * np.pi * frequency * t) + \
                                                  0.1 * np.random.randn(seq_len)
        
        # 进行预测
        print("开始预测...")
        predictions = predictor.predict_motion(
            history_motion,
            future_length=args.future_length,
            use_pisl_only=args.use_pisl_only,
            use_diffusion_only=args.use_diffusion_only
        )
        
        print("预测完成!")
        for name, pred in predictions.items():
            print(f"  {name}: {pred.shape}")
        
        # 可视化结果
        print("生成可视化...")
        figures = predictor.visualize_predictions(
            history_motion, predictions, save_dir=output_dir
        )
        
        # 保存预测结果
        for name, pred in predictions.items():
            np.save(os.path.join(output_dir, f'{name}.npy'), pred)
        
        # 创建视频（如果需要）
        if args.create_video:
            video_path = os.path.join(output_dir, 'prediction_video.mp4')
            # 这里需要真实数据才能创建对比视频
            print("视频创建需要真实数据进行对比")
        
        print(f"结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()