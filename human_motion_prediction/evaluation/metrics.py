"""
运动预测评估指标模块
包含MPJPE、加速度误差、物理合理性等评估指标
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import euclidean
from sklearn.metrics import mean_squared_error, mean_absolute_error


class MotionMetrics:
    """人体运动预测评估指标"""
    
    def __init__(self, n_joints: int = 22, fps: int = 30):
        self.n_joints = n_joints
        self.fps = fps
        self.dt = 1.0 / fps
        
        # Human3.6M关节连接
        self.joint_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # 脊柱
            (0, 5), (5, 6), (6, 7),          # 左臂
            (0, 8), (8, 9), (9, 10),         # 右臂
            (0, 11), (11, 12), (12, 13),     # 左腿
            (0, 14), (14, 15), (15, 16),     # 右腿
            (1, 17), (17, 18), (18, 19),     # 头部
            (7, 20), (10, 21)                # 手部
        ]
        
        # 关节组定义
        self.joint_groups = {
            'upper_body': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 18, 19, 20, 21],
            'lower_body': [11, 12, 13, 14, 15, 16],
            'arms': [5, 6, 7, 8, 9, 10, 20, 21],
            'legs': [11, 12, 13, 14, 15, 16],
            'torso': [0, 1, 2, 3, 4]
        }
    
    def compute_metrics(self, 
                       predictions: np.ndarray, 
                       targets: np.ndarray,
                       compute_detailed: bool = True) -> Dict[str, float]:
        """
        计算所有评估指标
        
        Args:
            predictions: [batch_size, seq_len, n_joints * 3] 预测结果
            targets: [batch_size, seq_len, n_joints * 3] 真实值
            compute_detailed: 是否计算详细指标
            
        Returns:
            metrics: 评估指标字典
        """
        # 确保数据格式正确
        if predictions.ndim == 3 and predictions.shape[-1] == self.n_joints * 3:
            pred_joints = predictions.reshape(predictions.shape[0], predictions.shape[1], self.n_joints, 3)
            target_joints = targets.reshape(targets.shape[0], targets.shape[1], self.n_joints, 3)
        else:
            pred_joints = predictions
            target_joints = targets
        
        metrics = {}
        
        # 1. 基本位置误差指标
        metrics.update(self._compute_position_metrics(pred_joints, target_joints))
        
        # 2. 速度和加速度指标
        metrics.update(self._compute_velocity_metrics(pred_joints, target_joints))
        
        # 3. 物理合理性指标
        metrics.update(self._compute_physics_metrics(pred_joints))
        
        # 4. 详细指标（如果需要）
        if compute_detailed:
            metrics.update(self._compute_detailed_metrics(pred_joints, target_joints))
        
        return metrics
    
    def _compute_position_metrics(self, 
                                 pred_joints: np.ndarray, 
                                 target_joints: np.ndarray) -> Dict[str, float]:
        """计算位置相关指标"""
        metrics = {}
        
        # MPJPE (Mean Per Joint Position Error)
        joint_errors = np.linalg.norm(pred_joints - target_joints, axis=-1)  # [batch, seq, joints]
        metrics['mpjpe'] = np.mean(joint_errors) * 1000  # 转换为毫米
        
        # 每个时间步的MPJPE
        mpjpe_per_frame = np.mean(joint_errors, axis=(0, 2)) * 1000
        for i, mpjpe in enumerate(mpjpe_per_frame):
            metrics[f'mpjpe_frame_{i+1}'] = mpjpe
        
        # 最终帧MPJPE（重要指标）
        metrics['mpjpe_final'] = mpjpe_per_frame[-1]
        
        # P-MPJPE (Procrustes-aligned MPJPE)
        p_mpjpe = self._compute_p_mpjpe(pred_joints, target_joints)
        metrics['p_mpjpe'] = p_mpjpe * 1000
        
        # MSE和MAE
        mse = np.mean((pred_joints - target_joints) ** 2)
        mae = np.mean(np.abs(pred_joints - target_joints))
        metrics['mse'] = mse
        metrics['mae'] = mae * 1000  # 转换为毫米
        
        return metrics
    
    def _compute_velocity_metrics(self, 
                                 pred_joints: np.ndarray, 
                                 target_joints: np.ndarray) -> Dict[str, float]:
        """计算速度和加速度相关指标"""
        metrics = {}
        
        # 计算速度
        pred_velocities = np.diff(pred_joints, axis=1) / self.dt
        target_velocities = np.diff(target_joints, axis=1) / self.dt
        
        # 速度误差
        velocity_errors = np.linalg.norm(pred_velocities - target_velocities, axis=-1)
        metrics['velocity_error'] = np.mean(velocity_errors) * 1000
        
        # 计算加速度
        pred_accelerations = np.diff(pred_velocities, axis=1) / self.dt
        target_accelerations = np.diff(target_velocities, axis=1) / self.dt
        
        # 加速度误差
        acceleration_errors = np.linalg.norm(pred_accelerations - target_accelerations, axis=-1)
        metrics['acceleration_error'] = np.mean(acceleration_errors) * 1000
        
        # Jerk误差（加速度的变化率）
        pred_jerk = np.diff(pred_accelerations, axis=1) / self.dt
        target_jerk = np.diff(target_accelerations, axis=1) / self.dt
        jerk_errors = np.linalg.norm(pred_jerk - target_jerk, axis=-1)
        metrics['jerk_error'] = np.mean(jerk_errors) * 1000
        
        return metrics
    
    def _compute_physics_metrics(self, pred_joints: np.ndarray) -> Dict[str, float]:
        """计算物理合理性指标"""
        metrics = {}
        
        # 计算速度和加速度
        velocities = np.diff(pred_joints, axis=1) / self.dt
        accelerations = np.diff(velocities, axis=1) / self.dt
        
        # 速度合理性（速度不应超过人体极限）
        velocity_magnitudes = np.linalg.norm(velocities, axis=-1)
        max_reasonable_velocity = 5.0  # 5 m/s
        velocity_violations = np.mean(velocity_magnitudes > max_reasonable_velocity)
        metrics['velocity_violation_rate'] = velocity_violations
        metrics['max_velocity'] = np.max(velocity_magnitudes)
        metrics['mean_velocity'] = np.mean(velocity_magnitudes)
        
        # 加速度合理性
        acceleration_magnitudes = np.linalg.norm(accelerations, axis=-1)
        max_reasonable_acceleration = 20.0  # 20 m/s²
        acceleration_violations = np.mean(acceleration_magnitudes > max_reasonable_acceleration)
        metrics['acceleration_violation_rate'] = acceleration_violations
        metrics['max_acceleration'] = np.max(acceleration_magnitudes)
        metrics['mean_acceleration'] = np.mean(acceleration_magnitudes)
        
        # 骨骼长度一致性
        bone_consistency = self._compute_bone_consistency(pred_joints)
        metrics['bone_consistency'] = bone_consistency
        
        # 运动平滑度
        smoothness = self._compute_smoothness(pred_joints)
        metrics['smoothness'] = smoothness
        
        return metrics
    
    def _compute_detailed_metrics(self, 
                                 pred_joints: np.ndarray, 
                                 target_joints: np.ndarray) -> Dict[str, float]:
        """计算详细的分组指标"""
        metrics = {}
        
        # 按关节组计算MPJPE
        for group_name, joint_indices in self.joint_groups.items():
            if all(idx < self.n_joints for idx in joint_indices):
                pred_group = pred_joints[:, :, joint_indices, :]
                target_group = target_joints[:, :, joint_indices, :]
                
                group_errors = np.linalg.norm(pred_group - target_group, axis=-1)
                metrics[f'mpjpe_{group_name}'] = np.mean(group_errors) * 1000
        
        # 按时间窗口计算指标
        seq_len = pred_joints.shape[1]
        windows = [
            ('short_term', slice(0, seq_len // 3)),
            ('medium_term', slice(seq_len // 3, 2 * seq_len // 3)),
            ('long_term', slice(2 * seq_len // 3, seq_len))
        ]
        
        for window_name, window_slice in windows:
            pred_window = pred_joints[:, window_slice, :, :]
            target_window = target_joints[:, window_slice, :, :]
            
            window_errors = np.linalg.norm(pred_window - target_window, axis=-1)
            metrics[f'mpjpe_{window_name}'] = np.mean(window_errors) * 1000
        
        return metrics
    
    def _compute_p_mpjpe(self, 
                        pred_joints: np.ndarray, 
                        target_joints: np.ndarray) -> float:
        """计算Procrustes对齐的MPJPE"""
        batch_size, seq_len = pred_joints.shape[:2]
        total_error = 0.0
        total_count = 0
        
        for b in range(batch_size):
            for t in range(seq_len):
                pred_frame = pred_joints[b, t]  # [n_joints, 3]
                target_frame = target_joints[b, t]  # [n_joints, 3]
                
                # Procrustes对齐
                aligned_pred = self._procrustes_align(pred_frame, target_frame)
                
                # 计算误差
                errors = np.linalg.norm(aligned_pred - target_frame, axis=1)
                total_error += np.sum(errors)
                total_count += len(errors)
        
        return total_error / total_count
    
    def _procrustes_align(self, 
                         pred: np.ndarray, 
                         target: np.ndarray) -> np.ndarray:
        """Procrustes对齐"""
        # 中心化
        pred_centered = pred - np.mean(pred, axis=0)
        target_centered = target - np.mean(target, axis=0)
        
        # 计算最优旋转矩阵
        H = pred_centered.T @ target_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # 确保是旋转矩阵（行列式为正）
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # 计算缩放因子
        scale = np.sum(S) / np.sum(pred_centered ** 2)
        
        # 应用变换
        aligned = scale * pred_centered @ R + np.mean(target, axis=0)
        
        return aligned
    
    def _compute_bone_consistency(self, pred_joints: np.ndarray) -> float:
        """计算骨骼长度一致性"""
        batch_size, seq_len = pred_joints.shape[:2]
        
        bone_length_vars = []
        
        for joint1, joint2 in self.joint_connections:
            if joint1 < self.n_joints and joint2 < self.n_joints:
                # 计算每帧的骨骼长度
                bone_vectors = pred_joints[:, :, joint1, :] - pred_joints[:, :, joint2, :]
                bone_lengths = np.linalg.norm(bone_vectors, axis=-1)  # [batch, seq]
                
                # 计算长度变化的方差
                bone_length_var = np.var(bone_lengths, axis=1)  # [batch]
                bone_length_vars.extend(bone_length_var)
        
        # 返回平均方差的倒数（一致性越高，方差越小）
        mean_var = np.mean(bone_length_vars)
        return 1.0 / (1.0 + mean_var)
    
    def _compute_smoothness(self, pred_joints: np.ndarray) -> float:
        """计算运动平滑度"""
        # 计算二阶差分（加速度）
        first_diff = np.diff(pred_joints, axis=1)
        second_diff = np.diff(first_diff, axis=1)
        
        # 计算加速度的变化率（Jerk）
        jerk = np.diff(second_diff, axis=1)
        jerk_magnitude = np.linalg.norm(jerk, axis=-1)
        
        # 平滑度是Jerk的倒数
        mean_jerk = np.mean(jerk_magnitude)
        return 1.0 / (1.0 + mean_jerk)
    
    def compute_fid_score(self, 
                         pred_features: np.ndarray, 
                         real_features: np.ndarray) -> float:
        """
        计算Fréchet Inception Distance (FID) 用于评估生成质量
        
        Args:
            pred_features: 预测运动的特征
            real_features: 真实运动的特征
        """
        # 计算均值和协方差
        mu_pred = np.mean(pred_features, axis=0)
        mu_real = np.mean(real_features, axis=0)
        
        sigma_pred = np.cov(pred_features, rowvar=False)
        sigma_real = np.cov(real_features, rowvar=False)
        
        # 计算FID
        diff = mu_pred - mu_real
        
        # 计算矩阵平方根
        covmean = np.linalg.sqrtm(sigma_pred @ sigma_real)
        
        # 处理数值不稳定性
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = np.sum(diff ** 2) + np.trace(sigma_pred + sigma_real - 2 * covmean)
        
        return fid
    
    def compute_diversity_score(self, predictions: np.ndarray) -> float:
        """
        计算预测的多样性分数
        
        Args:
            predictions: [n_samples, seq_len, n_joints, 3] 多个预测样本
        """
        n_samples = predictions.shape[0]
        
        if n_samples < 2:
            return 0.0
        
        # 计算所有样本对之间的平均距离
        total_distance = 0.0
        pair_count = 0
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                # 计算两个序列之间的平均距离
                distance = np.mean(np.linalg.norm(predictions[i] - predictions[j], axis=-1))
                total_distance += distance
                pair_count += 1
        
        return total_distance / pair_count if pair_count > 0 else 0.0
    
    def print_metrics_summary(self, metrics: Dict[str, float]):
        """打印指标摘要"""
        print("\n" + "="*50)
        print("运动预测评估指标摘要")
        print("="*50)
        
        # 主要指标
        main_metrics = ['mpjpe', 'p_mpjpe', 'mpjpe_final', 'velocity_error', 'acceleration_error']
        print("\n主要指标:")
        for metric in main_metrics:
            if metric in metrics:
                print(f"  {metric:20s}: {metrics[metric]:8.2f}")
        
        # 物理合理性指标
        physics_metrics = ['velocity_violation_rate', 'acceleration_violation_rate', 
                          'bone_consistency', 'smoothness']
        print("\n物理合理性指标:")
        for metric in physics_metrics:
            if metric in metrics:
                print(f"  {metric:20s}: {metrics[metric]:8.4f}")
        
        # 分组指标
        group_metrics = [k for k in metrics.keys() if k.startswith('mpjpe_') and 
                        any(group in k for group in self.joint_groups.keys())]
        if group_metrics:
            print("\n分组指标 (MPJPE):")
            for metric in sorted(group_metrics):
                print(f"  {metric:20s}: {metrics[metric]:8.2f}")
        
        print("="*50)


# 示例使用
if __name__ == "__main__":
    # 创建评估器
    metrics_calculator = MotionMetrics(n_joints=22, fps=30)
    
    # 生成示例数据
    batch_size = 8
    seq_len = 25
    n_joints = 22
    
    # 模拟预测和真实数据
    predictions = np.random.randn(batch_size, seq_len, n_joints, 3) * 0.1
    targets = np.random.randn(batch_size, seq_len, n_joints, 3) * 0.1
    
    # 计算指标
    metrics = metrics_calculator.compute_metrics(predictions, targets)
    
    # 打印结果
    metrics_calculator.print_metrics_summary(metrics)
    
    # 计算多样性分数
    diversity = metrics_calculator.compute_diversity_score(predictions)
    print(f"\n多样性分数: {diversity:.4f}")
    
    # 计算FID分数（使用简化特征）
    pred_features = predictions.reshape(batch_size, -1)
    real_features = targets.reshape(batch_size, -1)
    fid = metrics_calculator.compute_fid_score(pred_features, real_features)
    print(f"FID分数: {fid:.4f}")