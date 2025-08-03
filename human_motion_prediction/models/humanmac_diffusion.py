"""
HumanMAC 扩散模型模块
基于扩散过程的人体动作生成和预测
结合PISL样条学习的运动学约束
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Union
from einops import rearrange, repeat


class SinusoidalPositionEmbedding(nn.Module):
    """正弦位置编码"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class MotionTransformerBlock(nn.Module):
    """运动Transformer块"""
    
    def __init__(self, 
                 d_model: int = 512,
                 n_heads: int = 8,
                 d_ff: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        
        self.self_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                x: torch.Tensor, 
                context: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.self_attention(x, x, x, attn_mask=mask)
        x = residual + self.dropout(attn_out)
        
        # Cross-attention (如果有context)
        if context is not None:
            residual = x
            x = self.norm2(x)
            attn_out, _ = self.cross_attention(x, context, context)
            x = residual + self.dropout(attn_out)
        
        # Feed forward
        residual = x
        x = self.norm3(x)
        ff_out = self.feed_forward(x)
        x = residual + ff_out
        
        return x


class MotionEncoder(nn.Module):
    """运动编码器"""
    
    def __init__(self,
                 input_dim: int = 66,  # 22 joints * 3 coords
                 d_model: int = 512,
                 n_layers: int = 6,
                 n_heads: int = 8,
                 d_ff: int = 2048,
                 max_seq_len: int = 1000,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(max_seq_len, d_model) * 0.02)
        
        self.transformer_blocks = nn.ModuleList([
            MotionTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim] 运动序列
            mask: [seq_len, seq_len] 注意力掩码
        Returns:
            encoded: [batch_size, seq_len, d_model] 编码后的特征
        """
        batch_size, seq_len, _ = x.shape
        
        # 投影到模型维度
        x = self.input_projection(x)
        
        # 添加位置编码
        x = x + self.pos_embedding[:seq_len].unsqueeze(0)
        x = self.dropout(x)
        
        # Transformer编码
        for block in self.transformer_blocks:
            x = block(x, mask=mask)
        
        return x


class MotionDecoder(nn.Module):
    """运动解码器"""
    
    def __init__(self,
                 output_dim: int = 66,  # 22 joints * 3 coords
                 d_model: int = 512,
                 n_layers: int = 6,
                 n_heads: int = 8,
                 d_ff: int = 2048,
                 max_seq_len: int = 1000,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.pos_embedding = nn.Parameter(torch.randn(max_seq_len, d_model) * 0.02)
        
        self.transformer_blocks = nn.ModuleList([
            MotionTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.output_projection = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                x: torch.Tensor, 
                context: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model] 噪声输入
            context: [batch_size, context_len, d_model] 历史运动上下文
            mask: 注意力掩码
        Returns:
            output: [batch_size, seq_len, output_dim] 预测的运动
        """
        batch_size, seq_len, _ = x.shape
        
        # 添加位置编码
        x = x + self.pos_embedding[:seq_len].unsqueeze(0)
        x = self.dropout(x)
        
        # Transformer解码
        for block in self.transformer_blocks:
            x = block(x, context=context, mask=mask)
        
        # 输出投影
        output = self.output_projection(x)
        
        return output


class HumanMACDiffusion(nn.Module):
    """
    HumanMAC扩散模型主体
    结合PISL样条学习的运动学约束
    """
    
    def __init__(self,
                 input_dim: int = 66,  # 22 joints * 3 coords
                 d_model: int = 512,
                 n_encoder_layers: int = 6,
                 n_decoder_layers: int = 6,
                 n_heads: int = 8,
                 d_ff: int = 2048,
                 max_seq_len: int = 1000,
                 dropout: float = 0.1,
                 n_timesteps: int = 1000,
                 beta_schedule: str = "cosine"):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_timesteps = n_timesteps
        
        # 时间步嵌入
        self.time_embedding = SinusoidalPositionEmbedding(d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        
        # 编码器和解码器
        self.encoder = MotionEncoder(
            input_dim, d_model, n_encoder_layers, n_heads, d_ff, max_seq_len, dropout
        )
        
        self.decoder = MotionDecoder(
            input_dim, d_model, n_decoder_layers, n_heads, d_ff, max_seq_len, dropout
        )
        
        # 噪声输入投影
        self.noise_projection = nn.Linear(input_dim, d_model)
        
        # 扩散过程参数
        self.register_buffer('betas', self._get_beta_schedule(beta_schedule, n_timesteps))
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', 
                           torch.sqrt(1 - self.alphas_cumprod))
        
        # PISL运动学约束嵌入
        self.kinematic_embedding = nn.Linear(d_model, d_model)
        
    def _get_beta_schedule(self, schedule: str, n_timesteps: int) -> torch.Tensor:
        """获取噪声调度"""
        if schedule == "linear":
            return torch.linspace(1e-4, 0.02, n_timesteps)
        elif schedule == "cosine":
            s = 0.008
            steps = n_timesteps + 1
            x = torch.linspace(0, n_timesteps, steps)
            alphas_cumprod = torch.cos(((x / n_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0, 0.999)
        else:
            raise ValueError(f"Unknown beta schedule: {schedule}")
    
    def forward(self, 
                x: torch.Tensor,
                timesteps: torch.Tensor,
                condition: torch.Tensor,
                kinematic_params: Optional[Dict] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [batch_size, seq_len, input_dim] 噪声输入
            timesteps: [batch_size] 时间步
            condition: [batch_size, condition_len, input_dim] 条件输入（历史运动）
            kinematic_params: PISL样条参数
        
        Returns:
            predicted_noise: [batch_size, seq_len, input_dim] 预测的噪声
        """
        batch_size, seq_len, _ = x.shape
        
        # 时间步嵌入
        time_emb = self.time_embedding(timesteps)  # [batch_size, d_model]
        time_emb = self.time_mlp(time_emb)  # [batch_size, d_model]
        
        # 编码条件输入（历史运动）
        condition_encoded = self.encoder(condition)  # [batch_size, condition_len, d_model]
        
        # 将时间嵌入添加到条件编码中
        time_emb_expanded = time_emb.unsqueeze(1).expand(-1, condition_encoded.shape[1], -1)
        condition_encoded = condition_encoded + time_emb_expanded
        
        # 如果有运动学参数，添加运动学约束
        if kinematic_params is not None:
            kinematic_emb = self._embed_kinematic_params(kinematic_params)
            condition_encoded = condition_encoded + kinematic_emb
        
        # 噪声输入投影
        x_projected = self.noise_projection(x)  # [batch_size, seq_len, d_model]
        
        # 添加时间嵌入到噪声输入
        time_emb_noise = time_emb.unsqueeze(1).expand(-1, seq_len, -1)
        x_projected = x_projected + time_emb_noise
        
        # 解码生成预测
        predicted_noise = self.decoder(x_projected, condition_encoded)
        
        return predicted_noise
    
    def _embed_kinematic_params(self, kinematic_params: Dict) -> torch.Tensor:
        """
        嵌入PISL运动学参数
        
        Args:
            kinematic_params: PISL样条参数字典
        
        Returns:
            kinematic_emb: [batch_size, seq_len, d_model] 运动学嵌入
        """
        # 简化实现：将控制点展平并投影
        control_points = kinematic_params['control_points']  # [n_joints, 3, n_control_points]
        
        # 展平控制点
        flattened = control_points.view(-1)  # [n_joints * 3 * n_control_points]
        
        # 投影到模型维度
        if flattened.shape[0] < self.d_model:
            # 如果维度不够，进行填充
            padding = torch.zeros(self.d_model - flattened.shape[0], device=flattened.device)
            flattened = torch.cat([flattened, padding])
        elif flattened.shape[0] > self.d_model:
            # 如果维度过多，进行截断
            flattened = flattened[:self.d_model]
        
        # 应用线性变换
        kinematic_emb = self.kinematic_embedding(flattened.unsqueeze(0))  # [1, d_model]
        
        return kinematic_emb.unsqueeze(1)  # [1, 1, d_model]
    
    def add_noise(self, x: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """添加噪声到干净数据"""
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise
    
    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """随机采样时间步"""
        return torch.randint(0, self.n_timesteps, (batch_size,), device=device)
    
    def compute_loss(self, 
                    x: torch.Tensor, 
                    condition: torch.Tensor,
                    kinematic_params: Optional[Dict] = None) -> torch.Tensor:
        """
        计算扩散损失
        
        Args:
            x: [batch_size, seq_len, input_dim] 真实运动数据
            condition: [batch_size, condition_len, input_dim] 条件输入
            kinematic_params: PISL样条参数
        
        Returns:
            loss: 扩散损失
        """
        batch_size = x.shape[0]
        device = x.device
        
        # 随机采样时间步
        timesteps = self.sample_timesteps(batch_size, device)
        
        # 生成随机噪声
        noise = torch.randn_like(x)
        
        # 添加噪声
        x_noisy = self.add_noise(x, noise, timesteps)
        
        # 预测噪声
        predicted_noise = self.forward(x_noisy, timesteps, condition, kinematic_params)
        
        # 计算MSE损失
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss
    
    @torch.no_grad()
    def sample(self, 
               shape: Tuple[int, int, int],
               condition: torch.Tensor,
               kinematic_params: Optional[Dict] = None,
               n_steps: Optional[int] = None) -> torch.Tensor:
        """
        DDPM采样生成运动序列
        
        Args:
            shape: (batch_size, seq_len, input_dim) 输出形状
            condition: [batch_size, condition_len, input_dim] 条件输入
            kinematic_params: PISL样条参数
            n_steps: 采样步数，默认使用全部时间步
        
        Returns:
            samples: [batch_size, seq_len, input_dim] 生成的运动序列
        """
        device = condition.device
        batch_size, seq_len, input_dim = shape
        
        if n_steps is None:
            n_steps = self.n_timesteps
        
        # 从纯噪声开始
        x = torch.randn(shape, device=device)
        
        # 逆向扩散过程
        timesteps = torch.linspace(self.n_timesteps - 1, 0, n_steps, dtype=torch.long, device=device)
        
        for i, t in enumerate(timesteps):
            t_batch = t.repeat(batch_size)
            
            # 预测噪声
            predicted_noise = self.forward(x, t_batch, condition, kinematic_params)
            
            # 计算去噪后的x
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            beta_t = self.betas[t]
            
            # 计算均值
            x_mean = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise)
            
            if t > 0:
                # 添加噪声（除了最后一步）
                alpha_cumprod_prev = self.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0)
                variance = beta_t * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t)
                noise = torch.randn_like(x)
                x = x_mean + torch.sqrt(variance) * noise
            else:
                x = x_mean
        
        return x
    
    @torch.no_grad()
    def ddim_sample(self,
                   shape: Tuple[int, int, int],
                   condition: torch.Tensor,
                   kinematic_params: Optional[Dict] = None,
                   n_steps: int = 50,
                   eta: float = 0.0) -> torch.Tensor:
        """
        DDIM采样（更快的采样方法）
        
        Args:
            shape: 输出形状
            condition: 条件输入
            kinematic_params: PISL样条参数
            n_steps: 采样步数
            eta: DDIM参数，0为确定性采样
        
        Returns:
            samples: 生成的运动序列
        """
        device = condition.device
        batch_size = shape[0]
        
        # 创建时间步序列
        timesteps = torch.linspace(self.n_timesteps - 1, 0, n_steps, dtype=torch.long, device=device)
        
        # 从纯噪声开始
        x = torch.randn(shape, device=device)
        
        for i, t in enumerate(timesteps):
            t_batch = t.repeat(batch_size)
            
            # 预测噪声
            predicted_noise = self.forward(x, t_batch, condition, kinematic_params)
            
            # DDIM更新
            alpha_cumprod_t = self.alphas_cumprod[t]
            alpha_cumprod_prev = self.alphas_cumprod[timesteps[i + 1]] if i < len(timesteps) - 1 else torch.tensor(1.0, device=device)
            
            # 预测x_0
            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
            
            # 计算方向
            direction = torch.sqrt(1 - alpha_cumprod_prev - eta ** 2 * (1 - alpha_cumprod_t)) * predicted_noise
            
            # 添加随机性
            if eta > 0 and i < len(timesteps) - 1:
                noise = torch.randn_like(x)
                direction += eta * torch.sqrt(1 - alpha_cumprod_t) * noise
            
            # 更新x
            x = torch.sqrt(alpha_cumprod_prev) * pred_x0 + direction
        
        return x


class MotionMasking:
    """运动序列掩码工具"""
    
    @staticmethod
    def create_causal_mask(seq_len: int) -> torch.Tensor:
        """创建因果掩码（下三角矩阵）"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        return mask.bool()
    
    @staticmethod
    def create_random_mask(seq_len: int, mask_ratio: float = 0.15) -> torch.Tensor:
        """创建随机掩码"""
        mask = torch.rand(seq_len) < mask_ratio
        return mask
    
    @staticmethod
    def apply_mask(x: torch.Tensor, mask: torch.Tensor, mask_value: float = 0.0) -> torch.Tensor:
        """应用掩码到输入序列"""
        masked_x = x.clone()
        masked_x[mask] = mask_value
        return masked_x


# 示例使用
if __name__ == "__main__":
    # 创建HumanMAC扩散模型
    model = HumanMACDiffusion(
        input_dim=66,  # 22 joints * 3 coordinates
        d_model=512,
        n_encoder_layers=6,
        n_decoder_layers=6,
        n_heads=8,
        n_timesteps=1000
    )
    
    # 示例数据
    batch_size = 4
    condition_len = 25  # 历史25帧
    pred_len = 25      # 预测25帧
    input_dim = 66
    
    # 历史运动条件
    condition = torch.randn(batch_size, condition_len, input_dim)
    
    # 真实未来运动
    target = torch.randn(batch_size, pred_len, input_dim)
    
    # 模拟PISL运动学参数
    kinematic_params = {
        'control_points': torch.randn(22, 3, 10),
        'knots': torch.linspace(0, 1, 14),
        'spline_degree': 3,
        'n_control_points': 10
    }
    
    # 计算训练损失
    loss = model.compute_loss(target, condition, kinematic_params)
    print(f"训练损失: {loss.item():.6f}")
    
    # 生成样本
    with torch.no_grad():
        samples = model.sample(
            shape=(batch_size, pred_len, input_dim),
            condition=condition,
            kinematic_params=kinematic_params
        )
        print(f"生成样本形状: {samples.shape}")
        
        # DDIM快速采样
        ddim_samples = model.ddim_sample(
            shape=(batch_size, pred_len, input_dim),
            condition=condition,
            kinematic_params=kinematic_params,
            n_steps=50
        )
        print(f"DDIM样本形状: {ddim_samples.shape}")