"""
PISL (Physics-Informed Spline Learning) 模块
用于从人体运动历史数据中学习关节点运动轨迹的运动学方程
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import BSpline, splrep, splev
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt


class PhysicsInformedSpline(nn.Module):
    """
    物理约束样条学习模块
    结合物理约束学习关节运动轨迹的样条表示
    """
    
    def __init__(self, 
                 n_joints: int = 22,
                 spline_degree: int = 3,
                 n_control_points: int = 10,
                 physics_weight: float = 1.0,
                 smoothness_weight: float = 0.1):
        super().__init__()
        
        self.n_joints = n_joints
        self.spline_degree = spline_degree
        self.n_control_points = n_control_points
        self.physics_weight = physics_weight
        self.smoothness_weight = smoothness_weight
        
        # 每个关节的样条控制点
        self.control_points = nn.Parameter(
            torch.randn(n_joints, 3, n_control_points) * 0.1
        )
        
        # 时间节点 (knot vector)
        self.register_buffer('knots', self._create_knot_vector())
        
        # 物理约束参数
        self.max_velocity = 5.0  # m/s
        self.max_acceleration = 20.0  # m/s²
        self.joint_limits = self._init_joint_limits()
        
    def _create_knot_vector(self) -> torch.Tensor:
        """创建B样条的节点向量"""
        # 使用均匀节点向量
        n_knots = self.n_control_points + self.spline_degree + 1
        knots = torch.linspace(0, 1, n_knots - 2 * self.spline_degree)
        
        # 添加边界重复节点
        start_knots = torch.zeros(self.spline_degree)
        end_knots = torch.ones(self.spline_degree)
        
        return torch.cat([start_knots, knots, end_knots])
    
    def _init_joint_limits(self) -> Dict:
        """初始化关节角度限制"""
        # 简化的人体关节限制 (以弧度为单位)
        return {
            'shoulder': (-np.pi, np.pi),
            'elbow': (0, np.pi),
            'hip': (-np.pi/2, np.pi/2),
            'knee': (0, np.pi),
            'ankle': (-np.pi/4, np.pi/4)
        }
    
    def forward(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播：计算给定时间点的位置、速度和加速度
        
        Args:
            t: 时间点 [batch_size, seq_len]
            
        Returns:
            positions: [batch_size, seq_len, n_joints, 3]
            velocities: [batch_size, seq_len, n_joints, 3] 
            accelerations: [batch_size, seq_len, n_joints, 3]
        """
        batch_size, seq_len = t.shape
        
        # 计算B样条基函数
        positions = []
        velocities = []
        accelerations = []
        
        for i in range(seq_len):
            t_i = t[:, i:i+1]  # [batch_size, 1]
            
            # 计算位置 (0阶导数)
            pos = self._evaluate_spline(t_i, derivative=0)
            positions.append(pos)
            
            # 计算速度 (1阶导数)
            vel = self._evaluate_spline(t_i, derivative=1)
            velocities.append(vel)
            
            # 计算加速度 (2阶导数)
            acc = self._evaluate_spline(t_i, derivative=2)
            accelerations.append(acc)
        
        positions = torch.stack(positions, dim=1)  # [batch_size, seq_len, n_joints, 3]
        velocities = torch.stack(velocities, dim=1)
        accelerations = torch.stack(accelerations, dim=1)
        
        return positions, velocities, accelerations
    
    def _evaluate_spline(self, t: torch.Tensor, derivative: int = 0) -> torch.Tensor:
        """
        评估B样条在给定时间点的值或导数
        
        Args:
            t: 时间点 [batch_size, 1]
            derivative: 导数阶数 (0=位置, 1=速度, 2=加速度)
            
        Returns:
            values: [batch_size, n_joints, 3]
        """
        batch_size = t.shape[0]
        
        # 计算B样条基函数
        basis_values = self._compute_basis_functions(t, derivative)
        
        # 计算样条值: sum(control_points * basis_functions)
        # basis_values: [batch_size, n_control_points]
        # control_points: [n_joints, 3, n_control_points]
        
        values = torch.einsum('bi,jki->bjk', basis_values, self.control_points)
        
        return values  # [batch_size, n_joints, 3]
    
    def _compute_basis_functions(self, t: torch.Tensor, derivative: int = 0) -> torch.Tensor:
        """
        计算B样条基函数及其导数
        使用Cox-de Boor递归算法
        """
        batch_size = t.shape[0]
        n = self.n_control_points
        p = self.spline_degree
        
        # 初始化基函数矩阵
        N = torch.zeros(batch_size, n, p + 1, device=t.device)
        
        # 0度基函数
        for i in range(n):
            mask = (t.squeeze(-1) >= self.knots[i]) & (t.squeeze(-1) < self.knots[i + 1])
            N[:, i, 0] = mask.float()
        
        # 递归计算高阶基函数
        for r in range(1, p + 1):
            for i in range(n - r):
                # 左项
                if self.knots[i + r] != self.knots[i]:
                    left = (t.squeeze(-1) - self.knots[i]) / (self.knots[i + r] - self.knots[i])
                    N[:, i, r] += left * N[:, i, r - 1]
                
                # 右项
                if self.knots[i + r + 1] != self.knots[i + 1]:
                    right = (self.knots[i + r + 1] - t.squeeze(-1)) / (self.knots[i + r + 1] - self.knots[i + 1])
                    N[:, i, r] += right * N[:, i + 1, r - 1]
        
        # 如果需要导数，递归计算
        if derivative > 0:
            return self._compute_derivative_basis(N[:, :, p], derivative, t)
        
        return N[:, :n-p, p]  # 返回p度基函数
    
    def _compute_derivative_basis(self, basis: torch.Tensor, order: int, t: torch.Tensor) -> torch.Tensor:
        """计算基函数的导数"""
        if order == 0:
            return basis
        
        # 简化实现：使用数值微分
        h = 1e-6
        t_plus = t + h
        t_minus = t - h
        
        basis_plus = self._compute_basis_functions(t_plus, 0)
        basis_minus = self._compute_basis_functions(t_minus, 0)
        
        derivative = (basis_plus - basis_minus) / (2 * h)
        
        if order > 1:
            return self._compute_derivative_basis(derivative, order - 1, t)
        
        return derivative
    
    def compute_physics_loss(self, positions: torch.Tensor, 
                           velocities: torch.Tensor, 
                           accelerations: torch.Tensor) -> torch.Tensor:
        """
        计算物理约束损失
        
        Args:
            positions: [batch_size, seq_len, n_joints, 3]
            velocities: [batch_size, seq_len, n_joints, 3]
            accelerations: [batch_size, seq_len, n_joints, 3]
        """
        loss = 0.0
        
        # 速度约束
        velocity_magnitude = torch.norm(velocities, dim=-1)  # [batch_size, seq_len, n_joints]
        velocity_violation = torch.relu(velocity_magnitude - self.max_velocity)
        loss += torch.mean(velocity_violation ** 2)
        
        # 加速度约束
        acceleration_magnitude = torch.norm(accelerations, dim=-1)
        acceleration_violation = torch.relu(acceleration_magnitude - self.max_acceleration)
        loss += torch.mean(acceleration_violation ** 2)
        
        # 关节角度约束 (简化版本)
        # 这里假设positions包含关节角度信息
        joint_violations = self._compute_joint_limit_violations(positions)
        loss += torch.mean(joint_violations)
        
        return loss
    
    def _compute_joint_limit_violations(self, positions: torch.Tensor) -> torch.Tensor:
        """计算关节限制违反"""
        # 简化实现：检查位置是否在合理范围内
        position_magnitude = torch.norm(positions, dim=-1)
        violations = torch.relu(position_magnitude - 2.0)  # 假设最大关节位移为2米
        return violations ** 2
    
    def compute_smoothness_loss(self, accelerations: torch.Tensor) -> torch.Tensor:
        """
        计算平滑性损失 (加速度的变化率)
        """
        # 计算加速度的时间导数 (jerk)
        jerk = accelerations[:, 1:] - accelerations[:, :-1]
        return torch.mean(jerk ** 2)
    
    def fit_trajectory(self, trajectory_data: torch.Tensor, 
                      time_points: torch.Tensor,
                      n_iterations: int = 1000) -> Dict:
        """
        拟合轨迹数据到样条表示
        
        Args:
            trajectory_data: [seq_len, n_joints, 3] 轨迹数据
            time_points: [seq_len] 时间点
            n_iterations: 优化迭代次数
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        
        losses = []
        
        for iteration in range(n_iterations):
            optimizer.zero_grad()
            
            # 前向传播
            t_batch = time_points.unsqueeze(0)  # [1, seq_len]
            pred_pos, pred_vel, pred_acc = self.forward(t_batch)
            
            # 数据拟合损失
            data_loss = torch.mean((pred_pos[0] - trajectory_data) ** 2)
            
            # 物理约束损失
            physics_loss = self.compute_physics_loss(pred_pos, pred_vel, pred_acc)
            
            # 平滑性损失
            smoothness_loss = self.compute_smoothness_loss(pred_acc)
            
            # 总损失
            total_loss = (data_loss + 
                         self.physics_weight * physics_loss + 
                         self.smoothness_weight * smoothness_loss)
            
            total_loss.backward()
            optimizer.step()
            
            losses.append({
                'total': total_loss.item(),
                'data': data_loss.item(),
                'physics': physics_loss.item(),
                'smoothness': smoothness_loss.item()
            })
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Loss = {total_loss.item():.6f}")
        
        return {
            'losses': losses,
            'control_points': self.control_points.detach().clone(),
            'knots': self.knots.clone()
        }
    
    def get_kinematic_equations(self) -> Dict:
        """
        获取学习到的运动学方程参数
        """
        return {
            'control_points': self.control_points.detach().clone(),
            'knots': self.knots.clone(),
            'spline_degree': self.spline_degree,
            'n_control_points': self.n_control_points
        }
    
    def predict_future(self, current_state: torch.Tensor, 
                      future_time_points: torch.Tensor) -> torch.Tensor:
        """
        基于当前状态预测未来轨迹
        
        Args:
            current_state: [n_joints, 3] 当前关节位置
            future_time_points: [future_seq_len] 未来时间点
            
        Returns:
            future_positions: [future_seq_len, n_joints, 3]
        """
        with torch.no_grad():
            t_batch = future_time_points.unsqueeze(0)  # [1, future_seq_len]
            future_pos, _, _ = self.forward(t_batch)
            return future_pos[0]  # [future_seq_len, n_joints, 3]


class TrajectoryExtractor:
    """
    从运动数据中提取关节轨迹的工具类
    """
    
    @staticmethod
    def extract_joint_trajectories(motion_data: np.ndarray, 
                                 joint_indices: List[int]) -> Dict[int, np.ndarray]:
        """
        从运动数据中提取指定关节的轨迹
        
        Args:
            motion_data: [seq_len, n_joints, 3] 运动数据
            joint_indices: 要提取的关节索引列表
            
        Returns:
            trajectories: 关节索引到轨迹数据的映射
        """
        trajectories = {}
        for joint_idx in joint_indices:
            trajectories[joint_idx] = motion_data[:, joint_idx, :]
        return trajectories
    
    @staticmethod
    def smooth_trajectory(trajectory: np.ndarray, 
                         smoothing_factor: float = 0.1) -> np.ndarray:
        """
        平滑轨迹数据
        """
        from scipy.ndimage import gaussian_filter1d
        
        smoothed = np.zeros_like(trajectory)
        for dim in range(trajectory.shape[1]):
            smoothed[:, dim] = gaussian_filter1d(
                trajectory[:, dim], 
                sigma=smoothing_factor * trajectory.shape[0]
            )
        
        return smoothed
    
    @staticmethod
    def compute_derivatives(trajectory: np.ndarray, 
                          dt: float = 1/30) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算轨迹的速度和加速度
        
        Args:
            trajectory: [seq_len, 3] 位置轨迹
            dt: 时间步长
            
        Returns:
            velocity: [seq_len, 3] 速度
            acceleration: [seq_len, 3] 加速度
        """
        # 使用中心差分计算导数
        velocity = np.gradient(trajectory, dt, axis=0)
        acceleration = np.gradient(velocity, dt, axis=0)
        
        return velocity, acceleration


# 示例使用
if __name__ == "__main__":
    # 创建PISL样条学习模型
    pisl_model = PhysicsInformedSpline(
        n_joints=22,
        spline_degree=3,
        n_control_points=10
    )
    
    # 生成示例轨迹数据
    seq_len = 50
    time_points = torch.linspace(0, 1, seq_len)
    
    # 模拟人体关节轨迹 (简单的正弦运动)
    trajectory_data = torch.zeros(seq_len, 22, 3)
    for joint in range(22):
        for dim in range(3):
            frequency = 0.5 + joint * 0.1 + dim * 0.2
            trajectory_data[:, joint, dim] = torch.sin(2 * np.pi * frequency * time_points)
    
    # 拟合轨迹
    print("开始拟合轨迹...")
    results = pisl_model.fit_trajectory(trajectory_data, time_points, n_iterations=500)
    
    # 获取运动学方程
    kinematic_params = pisl_model.get_kinematic_equations()
    print(f"学习到的控制点形状: {kinematic_params['control_points'].shape}")
    
    # 预测未来轨迹
    future_time = torch.linspace(1, 1.5, 25)  # 预测未来0.5秒
    future_trajectory = pisl_model.predict_future(
        trajectory_data[-1], future_time
    )
    print(f"预测的未来轨迹形状: {future_trajectory.shape}")