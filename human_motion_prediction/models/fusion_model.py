"""
融合模型：结合PISL样条学习和HumanMAC扩散模型
用于高质量的人体动作预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

from .pisl_spline import PhysicsInformedSpline
from .humanmac_diffusion import HumanMACDiffusion


class PISLHumanMACFusion(nn.Module):
    """
    PISL-HumanMAC融合模型
    结合物理约束的样条学习和扩散模型的生成能力
    """
    
    def __init__(self,
                 n_joints: int = 22,
                 input_dim: int = 66,  # n_joints * 3
                 d_model: int = 512,
                 spline_degree: int = 3,
                 n_control_points: int = 10,
                 n_encoder_layers: int = 6,
                 n_decoder_layers: int = 6,
                 n_heads: int = 8,
                 d_ff: int = 2048,
                 max_seq_len: int = 1000,
                 dropout: float = 0.1,
                 n_timesteps: int = 1000,
                 beta_schedule: str = "cosine",
                 physics_weight: float = 1.0,
                 smoothness_weight: float = 0.1,
                 fusion_weight: float = 0.5):
        super().__init__()
        
        self.n_joints = n_joints
        self.input_dim = input_dim
        self.fusion_weight = fusion_weight
        
        # PISL样条学习模块
        self.pisl_model = PhysicsInformedSpline(
            n_joints=n_joints,
            spline_degree=spline_degree,
            n_control_points=n_control_points,
            physics_weight=physics_weight,
            smoothness_weight=smoothness_weight
        )
        
        # HumanMAC扩散模型
        self.diffusion_model = HumanMACDiffusion(
            input_dim=input_dim,
            d_model=d_model,
            n_encoder_layers=n_encoder_layers,
            n_decoder_layers=n_decoder_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            n_timesteps=n_timesteps,
            beta_schedule=beta_schedule
        )
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(input_dim * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, input_dim),
            nn.Tanh()  # 输出归一化
        )
        
        # 权重学习网络
        self.weight_predictor = nn.Sequential(
            nn.Linear(input_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # 运动学约束验证器
        self.constraint_validator = nn.Sequential(
            nn.Linear(input_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, 
                condition: torch.Tensor,
                target_length: int,
                future_time_points: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播：融合PISL和扩散模型的预测
        
        Args:
            condition: [batch_size, condition_len, input_dim] 历史运动条件
            target_length: 预测序列长度
            future_time_points: [target_length] 未来时间点
            
        Returns:
            predictions: 包含各种预测结果的字典
        """
        batch_size, condition_len, _ = condition.shape
        device = condition.device
        
        # 1. PISL样条预测
        pisl_predictions = self._pisl_predict(condition, target_length, future_time_points)
        
        # 2. 获取运动学参数
        kinematic_params = self._extract_kinematic_params(condition)
        
        # 3. HumanMAC扩散预测
        diffusion_predictions = self._diffusion_predict(
            condition, target_length, kinematic_params
        )
        
        # 4. 融合预测结果
        fused_predictions = self._fuse_predictions(
            pisl_predictions, diffusion_predictions, condition
        )
        
        return {
            'pisl_predictions': pisl_predictions,
            'diffusion_predictions': diffusion_predictions,
            'fused_predictions': fused_predictions,
            'kinematic_params': kinematic_params
        }
    
    def _pisl_predict(self, 
                     condition: torch.Tensor, 
                     target_length: int,
                     future_time_points: Optional[torch.Tensor] = None) -> torch.Tensor:
        """使用PISL进行预测"""
        batch_size = condition.shape[0]
        device = condition.device
        
        if future_time_points is None:
            future_time_points = torch.linspace(1, 1.5, target_length, device=device)
        
        predictions = []
        
        for b in range(batch_size):
            # 将条件数据重塑为 [seq_len, n_joints, 3]
            condition_reshaped = condition[b].view(-1, self.n_joints, 3)
            
            # 拟合PISL模型
            time_points = torch.linspace(0, 1, condition.shape[1], device=device)
            self.pisl_model.fit_trajectory(condition_reshaped, time_points, n_iterations=100)
            
            # 预测未来
            current_state = condition_reshaped[-1]  # [n_joints, 3]
            future_pred = self.pisl_model.predict_future(current_state, future_time_points)
            
            # 重塑为 [target_length, input_dim]
            future_pred_flat = future_pred.view(target_length, -1)
            predictions.append(future_pred_flat)
        
        return torch.stack(predictions, dim=0)  # [batch_size, target_length, input_dim]
    
    def _extract_kinematic_params(self, condition: torch.Tensor) -> Dict:
        """从历史数据中提取运动学参数"""
        batch_size, condition_len, _ = condition.shape
        device = condition.device
        
        # 使用第一个样本的参数作为代表（简化处理）
        condition_reshaped = condition[0].view(-1, self.n_joints, 3)
        time_points = torch.linspace(0, 1, condition_len, device=device)
        
        # 拟合样条获取参数
        self.pisl_model.fit_trajectory(condition_reshaped, time_points, n_iterations=50)
        kinematic_params = self.pisl_model.get_kinematic_equations()
        
        return kinematic_params
    
    def _diffusion_predict(self, 
                          condition: torch.Tensor, 
                          target_length: int,
                          kinematic_params: Dict) -> torch.Tensor:
        """使用HumanMAC扩散模型进行预测"""
        batch_size = condition.shape[0]
        
        # 使用DDIM快速采样
        predictions = self.diffusion_model.ddim_sample(
            shape=(batch_size, target_length, self.input_dim),
            condition=condition,
            kinematic_params=kinematic_params,
            n_steps=50,
            eta=0.0  # 确定性采样
        )
        
        return predictions
    
    def _fuse_predictions(self, 
                         pisl_pred: torch.Tensor,
                         diffusion_pred: torch.Tensor,
                         condition: torch.Tensor) -> torch.Tensor:
        """融合PISL和扩散模型的预测结果"""
        batch_size, target_length, input_dim = pisl_pred.shape
        
        # 连接两个预测结果
        combined = torch.cat([pisl_pred, diffusion_pred], dim=-1)  # [batch_size, target_length, input_dim*2]
        
        # 通过融合层
        fused = self.fusion_layer(combined)  # [batch_size, target_length, input_dim]
        
        # 学习自适应权重
        weights = self.weight_predictor(condition.mean(dim=1))  # [batch_size, 1]
        weights = weights.unsqueeze(1).expand(-1, target_length, -1)  # [batch_size, target_length, 1]
        
        # 加权融合
        adaptive_fused = (weights * pisl_pred + 
                         (1 - weights) * diffusion_pred + 
                         self.fusion_weight * fused) / (2 + self.fusion_weight)
        
        return adaptive_fused
    
    def compute_loss(self, 
                    condition: torch.Tensor,
                    target: torch.Tensor,
                    future_time_points: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        计算总损失
        
        Args:
            condition: [batch_size, condition_len, input_dim] 历史运动
            target: [batch_size, target_len, input_dim] 真实未来运动
            future_time_points: 未来时间点
            
        Returns:
            losses: 各种损失的字典
        """
        batch_size, target_len, input_dim = target.shape
        
        # 前向预测
        predictions = self.forward(condition, target_len, future_time_points)
        
        # 1. PISL损失
        pisl_pred = predictions['pisl_predictions']
        pisl_loss = F.mse_loss(pisl_pred, target)
        
        # 2. 扩散损失
        kinematic_params = predictions['kinematic_params']
        diffusion_loss = self.diffusion_model.compute_loss(target, condition, kinematic_params)
        
        # 3. 融合损失
        fused_pred = predictions['fused_predictions']
        fusion_loss = F.mse_loss(fused_pred, target)
        
        # 4. 物理约束损失
        physics_loss = self._compute_physics_constraint_loss(fused_pred, condition)
        
        # 5. 连续性损失
        continuity_loss = self._compute_continuity_loss(fused_pred, condition)
        
        # 总损失
        total_loss = (pisl_loss + 
                     diffusion_loss + 
                     fusion_loss + 
                     0.1 * physics_loss + 
                     0.1 * continuity_loss)
        
        return {
            'total_loss': total_loss,
            'pisl_loss': pisl_loss,
            'diffusion_loss': diffusion_loss,
            'fusion_loss': fusion_loss,
            'physics_loss': physics_loss,
            'continuity_loss': continuity_loss
        }
    
    def _compute_physics_constraint_loss(self, 
                                       predictions: torch.Tensor,
                                       condition: torch.Tensor) -> torch.Tensor:
        """计算物理约束损失"""
        batch_size, seq_len, _ = predictions.shape
        
        # 重塑为关节坐标
        pred_joints = predictions.view(batch_size, seq_len, self.n_joints, 3)
        
        # 计算速度和加速度
        dt = 1/30  # 假设30fps
        velocities = torch.diff(pred_joints, dim=1) / dt
        accelerations = torch.diff(velocities, dim=1) / dt
        
        # 速度约束
        velocity_magnitude = torch.norm(velocities, dim=-1)
        velocity_violation = torch.relu(velocity_magnitude - 5.0)  # 最大速度5m/s
        velocity_loss = torch.mean(velocity_violation ** 2)
        
        # 加速度约束
        acceleration_magnitude = torch.norm(accelerations, dim=-1)
        acceleration_violation = torch.relu(acceleration_magnitude - 20.0)  # 最大加速度20m/s²
        acceleration_loss = torch.mean(acceleration_violation ** 2)
        
        return velocity_loss + acceleration_loss
    
    def _compute_continuity_loss(self, 
                               predictions: torch.Tensor,
                               condition: torch.Tensor) -> torch.Tensor:
        """计算连续性损失"""
        # 确保预测的第一帧与历史的最后一帧连续
        last_condition = condition[:, -1, :]  # [batch_size, input_dim]
        first_prediction = predictions[:, 0, :]  # [batch_size, input_dim]
        
        continuity_loss = F.mse_loss(first_prediction, last_condition)
        
        return continuity_loss
    
    @torch.no_grad()
    def predict(self, 
               condition: torch.Tensor,
               target_length: int,
               future_time_points: Optional[torch.Tensor] = None,
               use_pisl_only: bool = False,
               use_diffusion_only: bool = False) -> torch.Tensor:
        """
        预测未来运动
        
        Args:
            condition: 历史运动条件
            target_length: 预测长度
            future_time_points: 未来时间点
            use_pisl_only: 仅使用PISL预测
            use_diffusion_only: 仅使用扩散模型预测
            
        Returns:
            predictions: 预测的运动序列
        """
        self.eval()
        
        predictions = self.forward(condition, target_length, future_time_points)
        
        if use_pisl_only:
            return predictions['pisl_predictions']
        elif use_diffusion_only:
            return predictions['diffusion_predictions']
        else:
            return predictions['fused_predictions']
    
    def analyze_predictions(self, 
                          condition: torch.Tensor,
                          target_length: int) -> Dict[str, torch.Tensor]:
        """
        分析不同预测方法的结果
        """
        with torch.no_grad():
            predictions = self.forward(condition, target_length)
            
            # 计算预测质量指标
            pisl_pred = predictions['pisl_predictions']
            diffusion_pred = predictions['diffusion_predictions']
            fused_pred = predictions['fused_predictions']
            
            # 计算物理合理性分数
            pisl_physics_score = self._evaluate_physics_compliance(pisl_pred)
            diffusion_physics_score = self._evaluate_physics_compliance(diffusion_pred)
            fused_physics_score = self._evaluate_physics_compliance(fused_pred)
            
            # 计算平滑度分数
            pisl_smoothness = self._evaluate_smoothness(pisl_pred)
            diffusion_smoothness = self._evaluate_smoothness(diffusion_pred)
            fused_smoothness = self._evaluate_smoothness(fused_pred)
            
            return {
                'predictions': predictions,
                'physics_scores': {
                    'pisl': pisl_physics_score,
                    'diffusion': diffusion_physics_score,
                    'fused': fused_physics_score
                },
                'smoothness_scores': {
                    'pisl': pisl_smoothness,
                    'diffusion': diffusion_smoothness,
                    'fused': fused_smoothness
                }
            }
    
    def _evaluate_physics_compliance(self, predictions: torch.Tensor) -> torch.Tensor:
        """评估预测的物理合理性"""
        batch_size, seq_len, _ = predictions.shape
        pred_joints = predictions.view(batch_size, seq_len, self.n_joints, 3)
        
        # 计算速度
        dt = 1/30
        velocities = torch.diff(pred_joints, dim=1) / dt
        velocity_magnitude = torch.norm(velocities, dim=-1)
        
        # 物理合理性分数（速度在合理范围内的比例）
        reasonable_velocity = (velocity_magnitude < 5.0).float()
        physics_score = torch.mean(reasonable_velocity)
        
        return physics_score
    
    def _evaluate_smoothness(self, predictions: torch.Tensor) -> torch.Tensor:
        """评估预测的平滑度"""
        # 计算二阶差分（加速度的变化）
        second_diff = torch.diff(predictions, n=2, dim=1)
        smoothness_score = 1.0 / (1.0 + torch.mean(torch.norm(second_diff, dim=-1)))
        
        return smoothness_score


# 示例使用
if __name__ == "__main__":
    # 创建融合模型
    fusion_model = PISLHumanMACFusion(
        n_joints=22,
        input_dim=66,
        d_model=512,
        spline_degree=3,
        n_control_points=10,
        fusion_weight=0.5
    )
    
    # 示例数据
    batch_size = 2
    condition_len = 25
    target_len = 25
    input_dim = 66
    
    # 历史运动条件
    condition = torch.randn(batch_size, condition_len, input_dim)
    
    # 真实未来运动
    target = torch.randn(batch_size, target_len, input_dim)
    
    # 训练模式：计算损失
    losses = fusion_model.compute_loss(condition, target)
    print("训练损失:")
    for loss_name, loss_value in losses.items():
        print(f"  {loss_name}: {loss_value.item():.6f}")
    
    # 推理模式：预测
    predictions = fusion_model.predict(condition, target_len)
    print(f"\n预测结果形状: {predictions.shape}")
    
    # 分析预测质量
    analysis = fusion_model.analyze_predictions(condition, target_len)
    print(f"\n物理合理性分数:")
    for method, score in analysis['physics_scores'].items():
        print(f"  {method}: {score.item():.4f}")
    
    print(f"\n平滑度分数:")
    for method, score in analysis['smoothness_scores'].items():
        print(f"  {method}: {score.item():.4f}")